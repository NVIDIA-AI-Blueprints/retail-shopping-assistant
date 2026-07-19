"""
Performs quality testing given QA pairs, using an LLM.
"""
from openai import OpenAI
from typing import Dict, Sequence
from collections import Counter
from datetime import datetime, timezone
import os
import json
import yaml


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required for response_quality.py")
    return value


# Configuration
LLM_NAME = _required_env("JUDGE_MODEL")
BASE_URL = _required_env("JUDGE_BASE_URL")
API_KEY_ENV = _required_env("JUDGE_API_KEY_ENV")
LLM_CLIENT = OpenAI(
    base_url=BASE_URL,
    api_key=_required_env(API_KEY_ENV)
)


def _quality_output_dir(conversation: str, result_directory: str) -> str:
    return f"conversations/{conversation}/quality/{result_directory}"


def _format_prior_turns(prior_turns: Sequence[Dict[str, str]] | None) -> str:
    if not prior_turns:
        return ""

    lines = ["ACTUAL PRIOR CONVERSATION:"]
    for index, turn in enumerate(prior_turns, start=1):
        lines.extend(
            [
                f"Turn {index}",
                f"Shopper: {turn['shopper']}",
                f"Assistant: {turn['assistant']}",
            ]
        )
    return "\n".join(lines)


def _write_quality_summary(output_path: str, result_directory: str) -> None:
    summary_entries = []
    scores = []
    per_file = {}

    for filename in sorted(f for f in os.listdir(output_path) if f.endswith(".yaml")):
        with open(os.path.join(output_path, filename), "r") as result_file:
            entries = yaml.safe_load(result_file) or []
        file_scores = [int(entry["score"]) for entry in entries]
        if not file_scores:
            continue
        scores.extend(file_scores)
        per_file[filename] = {
            "count": len(file_scores),
            "average_score": sum(file_scores) / len(file_scores),
            "min_score": min(file_scores),
            "max_score": max(file_scores),
        }
        for entry in entries:
            summary_entries.append(
                {
                    "filename": filename,
                    "index": entry["index"],
                    "query": entry["query"],
                    "score": int(entry["score"]),
                    "justification": entry["justification"],
                }
            )

    if not scores:
        raise RuntimeError(f"No judge scores found in {output_path}")

    distribution = {str(score): count for score, count in sorted(Counter(scores).items())}
    overall_average = sum(scores) / len(scores)
    summary = {
        "result_directory": result_directory,
        "commit": os.environ.get("GITHUB_SHA", ""),
        "ref": os.environ.get("GITHUB_REF", ""),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "judge_model": LLM_NAME,
        "judge_base_url": BASE_URL,
        "count": len(scores),
        "overall_average": overall_average,
        "score_distribution": distribution,
        "per_file": per_file,
        "lowest_scoring_turns": sorted(
            summary_entries, key=lambda entry: (entry["score"], entry["filename"], entry["index"])
        )[:10],
    }

    with open(os.path.join(output_path, "quality_summary.json"), "w") as summary_file:
        json.dump(summary, summary_file, indent=2)

    markdown_lines = [
        f"# Quality Summary: {result_directory}",
        "",
        f"- Commit: `{summary['commit'] or 'local'}`",
        f"- Judge model: `{LLM_NAME}`",
        f"- Turns judged: {summary['count']}",
        f"- Overall average: {overall_average:.2f}/5",
        f"- Score distribution: {distribution}",
        "",
        "## Per Scenario",
        "",
    ]
    for filename, file_summary in per_file.items():
        markdown_lines.append(
            f"- `{filename}`: {file_summary['average_score']:.2f}/5 "
            f"({file_summary['count']} turns, min {file_summary['min_score']}, "
            f"max {file_summary['max_score']})"
        )
    markdown_lines.extend(["", "## Lowest Scoring Turns", ""])
    for entry in summary["lowest_scoring_turns"]:
        markdown_lines.append(
            f"- `{entry['filename']}` turn {entry['index']} score {entry['score']}: "
            f"{entry['query']}"
        )

    with open(os.path.join(output_path, "quality_summary.md"), "w") as summary_file:
        summary_file.write("\n".join(markdown_lines) + "\n")


def judge_test(
        query: str, 
        answer: str, 
        ideal_answer: str,
        verbose: bool = True,
        prior_turns: Sequence[Dict[str, str]] | None = None,
        ) -> Dict[str, str]:
    
    if verbose:
        print("judge_test() | Starting judgement.")

    history = _format_prior_turns(prior_turns)
    history_section = f"\n{history}\n" if history else ""
    prompt = f"""
You are an expert answer quality evaluator. Your task is to rate how well the RAG-generated answer answers the given question, compared to the ideal (reference) answer. 
Note that these responses may sometimes vary. For instance, if two answers list different, but similar products, that is fine. 

When prior turns are provided, the actual conversation history is authoritative for resolving references and shopper intent. The reference answer is guidance for expected quality and content, but it may contain assumptions that conflict with the real thread. It must not override the actual conversation history.

Consider the following criteria:
- Relevance to the question
- Completeness
- Clarity and coherence
- Consistency with the actual conversation history

Return a score from 1 to 5:
- 5 = Perfect: fully answers the question in the actual conversation context with strong clarity
- 4 = Good: sensible and contextually correct, with only minor omissions or clarity issues
- 3 = Acceptable: partially correct, but may be missing details or be slightly off-topic.
- 2 = Poor: mostly incorrect or irrelevant
- 1 = Unacceptable: completely wrong or nonsensical

Also provide a brief justification (1-2 sentences).
{history_section}

Question: {query}

Ideal Answer: {ideal_answer}

RAG Answer: {answer}
"""

    judge_function = {
        "type": "function",
        "function": {
            "name": "judge_function",
            "description": "Assess the quality of a response.",
            "parameters": {
                "type": "object",
                "properties": {
                    "judgement": {
                        "type": "integer",
                        "description": "The quality of the response in its actual conversation context.",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "The reason for giving the associated score."
                    }
                },
                "required": ["judgement", "reasoning"]
            }
        }
    }

    response = LLM_CLIENT.chat.completions.create(
        model=LLM_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful assistant trained to judge QA quality."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        tools=[judge_function],
        tool_choice="required"
    )

    parsed_output = json.loads(response.choices[0].message.tool_calls[0].function.arguments)

    res = {
        "score": parsed_output["judgement"],
        "justification": parsed_output["reasoning"]
    }

    if verbose:
        print("judge_test() | Finished judgement. Response: {res}")

    return res

if __name__ == "__main__":

    CONVERSATION = os.environ["TEST_PATH"]
    RESULT_DIRECTORY = os.environ.get("RESULT_DIRECTORY", "results")
    QUERY_DIR = f'conversations/{CONVERSATION}'
    RES_DIR = f'conversations/{CONVERSATION}/{RESULT_DIRECTORY}'
    OUTPUT_PATH = _quality_output_dir(CONVERSATION, RESULT_DIRECTORY)

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    query_files = sorted([f for f in os.listdir(QUERY_DIR) if f.endswith('.yaml')])
    res_files = sorted([f for f in os.listdir(RES_DIR) if f.endswith('.yaml')])

    assert query_files == res_files, "Mismatch between query and result filenames!"

    for filename in query_files:
        with open(os.path.join(QUERY_DIR, filename), 'r') as qf:
            query_data = yaml.safe_load(qf)
        with open(os.path.join(RES_DIR, filename), 'r') as rf:
            res_data = yaml.safe_load(rf)

        queries = query_data["queries"]
        ideal_answers = query_data["answers"]
        result_entries = res_data["results"]

        assert len(queries) == len(ideal_answers) == len(result_entries), f"Mismatch in QA counts in {filename}"

        results_per_file = []
        prior_turns = []

        for i, (query, ideal_answer, result_obj) in enumerate(zip(queries, ideal_answers, result_entries)):
            rag_answer = result_obj["response"]

            judgement = judge_test(
                query=query,
                answer=rag_answer,
                ideal_answer=ideal_answer,
                prior_turns=prior_turns,
            )

            result_entry = {
                "filename": filename,
                "index": i,
                "query": query,
                "ideal_answer": ideal_answer,
                "rag_output": rag_answer,
                "score": judgement['score'],
                "justification": judgement['justification'],
                "timing": result_obj.get("timing", {})
            }

            print(result_entry)
            results_per_file.append(result_entry)
            prior_turns.append({"shopper": query, "assistant": rag_answer})

        # Write YAML output per file
        with open(f"{OUTPUT_PATH}/{filename}", 'w') as out_file:
            yaml.dump(results_per_file, out_file, sort_keys=False, allow_unicode=True)

    _write_quality_summary(OUTPUT_PATH, RESULT_DIRECTORY)
