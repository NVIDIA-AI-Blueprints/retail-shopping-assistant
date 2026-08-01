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


def _format_current_catalog_evidence(
    product_evidence: Sequence[Dict] | None,
    *,
    max_chars: int = 12_000,
) -> str:
    """Render a bounded, ref-free catalog projection for the Judge."""

    if not product_evidence:
        return ""
    heading = (
        "CURRENT-TURN STRUCTURED CATALOG EVIDENCE "
        "(authoritative for product-specific facts):"
    )
    lines = [heading]
    for raw_record in product_evidence[:20]:
        if not isinstance(raw_record, dict):
            continue
        record = {
            key: raw_record[key]
            for key in (
                "product_name",
                "source_tool",
                "evidence_type",
                "facts",
                "search_scope",
            )
            if raw_record.get(key) not in (None, "", [], {})
        }
        if not record:
            continue
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        if len("\n".join((*lines, line))) > max_chars:
            lines.append("(additional structured evidence omitted)")
            break
        lines.append(line)
    return "\n".join(lines) if len(lines) > 1 else ""


def _validate_diagnostic_expectations(
    expectations: Dict | None,
    diagnostics: Dict | None,
    *,
    label: str = "turn",
) -> None:
    """Fail before judging when the live skill/tool trace violates the fixture."""

    expected = expectations or {}
    trace = diagnostics or {}
    skill_files = set(trace.get("skill_files_read") or [])
    tool_calls = trace.get("tool_calls") or []
    if not isinstance(tool_calls, list):
        raise AssertionError(
            f"{label}: agent_diagnostics.tool_calls must be a list"
        )

    normalized_calls = [call for call in tool_calls if isinstance(call, dict)]
    called_tools = {
        str(call.get("tool_name") or "") for call in normalized_calls
    }
    completed_tools = {
        str(call.get("tool_name") or "")
        for call in normalized_calls
        if call.get("status") == "completed"
    }

    for skill_name in expected.get("required_skills", []):
        path = f"/shopper/{skill_name}/SKILL.md"
        if path not in skill_files:
            raise AssertionError(f"{label}: missing required skill {skill_name}")
    for skill_name in expected.get("forbidden_skills", []):
        path = f"/shopper/{skill_name}/SKILL.md"
        if path in skill_files:
            raise AssertionError(f"{label}: forbidden skill selected {skill_name}")
    for tool_name in expected.get("required_tools", []):
        if tool_name not in completed_tools:
            raise AssertionError(
                f"{label}: required tool did not complete {tool_name}"
            )
    for tool_name in expected.get("forbidden_tools", []):
        if tool_name in called_tools:
            raise AssertionError(f"{label}: forbidden tool called {tool_name}")

    for tool_name, expected_count in (
        expected.get("tool_call_counts") or {}
    ).items():
        actual_count = sum(
            call.get("tool_name") == tool_name for call in normalized_calls
        )
        if actual_count != expected_count:
            raise AssertionError(
                f"{label}: expected {expected_count} {tool_name} calls, "
                f"found {actual_count}"
            )

    for call_expectation in expected.get("tool_call_expectations", []):
        if not isinstance(call_expectation, dict):
            raise AssertionError(
                f"{label}: tool_call_expectations entries must be mappings"
            )
        if not any(
            _diagnostic_subset_matches(call_expectation, call)
            for call in normalized_calls
        ):
            raise AssertionError(
                f"{label}: no tool call matched expectation "
                f"{json.dumps(call_expectation, sort_keys=True)}"
            )

    expected_detail_names = expected.get("required_product_detail_names")
    if expected_detail_names is not None:
        product_evidence = trace.get("product_evidence") or []
        if not isinstance(product_evidence, list):
            raise AssertionError(
                f"{label}: agent_diagnostics.product_evidence must be a list"
            )
        actual_detail_names = {
            str(record.get("product_name") or "")
            for record in product_evidence
            if isinstance(record, dict)
            and record.get("source_tool") == "get_product_details_tool"
            and record.get("evidence_type") == "product_detail"
            and record.get("product_name")
        }
        required_detail_names = {str(name) for name in expected_detail_names}
        if actual_detail_names != required_detail_names:
            raise AssertionError(
                f"{label}: expected product detail evidence "
                f"{sorted(required_detail_names)}, found "
                f"{sorted(actual_detail_names)}"
            )

    expected_compaction = expected.get("conversation_summary_compaction")
    if (
        expected_compaction is not None
        and trace.get("conversation_summary_compaction") != expected_compaction
    ):
        raise AssertionError(
            f"{label}: expected conversation summary compaction "
            f"{expected_compaction!r}, found "
            f"{trace.get('conversation_summary_compaction')!r}"
        )
    expected_projection = expected.get("conversation_summary_input_projection")
    if (
        expected_projection is not None
        and trace.get("conversation_summary_input_projection")
        != expected_projection
    ):
        raise AssertionError(
            f"{label}: expected conversation summary input projection "
            f"{expected_projection!r}, found "
            f"{trace.get('conversation_summary_input_projection')!r}"
        )


def _diagnostic_subset_matches(expected: object, actual: object) -> bool:
    """Match a fixture-owned structural subset without interpreting text."""

    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _diagnostic_subset_matches(value, actual[key])
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and all(
            any(_diagnostic_subset_matches(item, candidate) for candidate in actual)
            for item in expected
        )
    return expected == actual


def _preflight_diagnostic_expectations(
    query_dir: str,
    result_dir: str,
    filenames: Sequence[str],
    *,
    require_agent_diagnostics: bool = False,
) -> None:
    """Validate every trace before the first paid Judge request."""

    for filename in filenames:
        with open(os.path.join(query_dir, filename), "r") as query_file:
            query_data = yaml.safe_load(query_file) or {}
        with open(os.path.join(result_dir, filename), "r") as result_file:
            result_data = yaml.safe_load(result_file) or {}
        expectations = query_data.get("diagnostic_expectations") or [
            {} for _ in query_data.get("queries", [])
        ]
        result_entries = result_data.get("results") or []
        if len(expectations) != len(result_entries):
            raise AssertionError(
                f"Mismatch in diagnostic expectation counts in {filename}"
            )
        for index, (expected, result) in enumerate(
            zip(expectations, result_entries)
        ):
            diagnostics = result.get("agent_diagnostics")
            if require_agent_diagnostics and not diagnostics:
                raise AssertionError(
                    f"{filename} turn {index}: missing agent diagnostics; "
                    "restart the trusted evaluation service with "
                    "EXPOSE_AGENT_DIAGNOSTICS=true before running the paid Judge"
                )
            _validate_diagnostic_expectations(
                expected,
                diagnostics,
                label=f"{filename} turn {index}",
            )


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
        product_evidence: Sequence[Dict] | None = None,
        ) -> Dict[str, str]:
    
    if verbose:
        print("judge_test() | Starting judgement.")

    history = _format_prior_turns(prior_turns)
    history_section = f"\n{history}\n" if history else ""
    evidence = _format_current_catalog_evidence(product_evidence)
    evidence_section = f"\n{evidence}\n" if evidence else ""
    prompt = f"""
You are an expert answer quality evaluator. Your task is to rate how well the RAG-generated answer answers the given question, compared to the ideal (reference) answer. 
Note that these responses may sometimes vary. For instance, if two answers list different, but similar products, that is fine. 

When prior turns are provided, the actual conversation history is authoritative for resolving references and shopper intent. The reference answer is guidance for expected quality and content, but it may contain assumptions that conflict with the real thread. It must not override the actual conversation history.

When current-turn structured catalog evidence is provided, it is authoritative for product-specific facts in the generated answer. Do not mark a fact unconfirmed when that exact fact appears in this evidence.

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
{evidence_section}

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
    _preflight_diagnostic_expectations(
        QUERY_DIR,
        RES_DIR,
        query_files,
        require_agent_diagnostics=CONVERSATION == "shopping",
    )

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
                product_evidence=(
                    result_obj.get("agent_diagnostics") or {}
                ).get("product_evidence"),
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
