"""
Performs quality testing given QA pairs, using an LLM.
"""
from openai import OpenAI
from typing import Dict
import os
import json
import yaml

# Configuration
LLM_NAME = "nvdev/meta/llama-3.1-70b-instruct"
#LLM_NAME = "nvdev/nv-mistralai/mistral-nemo-12b-instruct"
EMBED_NAME = "nvdev/nvidia/nv-embedqa-e5-v5"
LLM_CLIENT = OpenAI(
    base_url= "https://integrate.api.nvidia.com/v1", #"http://pdx-tme-018:8000/v1", #"https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"]
)
EMBED_CLIENT = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"]
)

def judge_test(
        query: str, 
        answer: str, 
        ideal_answer: str,
        verbose: bool = True
        ) -> Dict[str, str]:
    
    if verbose:
        print("judge_test() | Starting judgement.")

    prompt = f"""/no_think
You are an expert answer quality evaluator. Your task is to rate how well the RAG-generated answer answers the given question, compared to the ideal (reference) answer. 
Note that these responses may sometimes vary. For instance, if two answers list different, but similar products, that is fine. 

Consider the following criteria:
- Relevance to the question
- Completeness
- Clarity and coherence

Return a score from 1 to 5:
- 5 = Perfect: matches the reference in content and clarity
- 4 = Good: Differences can be seen, but the jist is the same, e.g. a sensible response is still being made.
- 3 = Acceptable: partially correct, but may be missing details or be slightly off-topic.
- 2 = Poor: mostly incorrect or irrelevant
- 1 = Unacceptable: completely wrong or nonsensical

Also provide a brief justification (1-2 sentences).

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
                        "description": "The quality of the response given the ideal response.",
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
            {"role": "system", "content": "/no_think You are a helpful assistant trained to judge QA quality."},
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
    QUERY_DIR = f'conversations/{CONVERSATION}'
    RES_DIR = f'conversations/{CONVERSATION}/results'
    OUTPUT_PATH = f'conversations/{CONVERSATION}/judge'

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

        for i, (query, ideal_answer, result_obj) in enumerate(zip(queries, ideal_answers, result_entries)):
            rag_answer = result_obj["response"]

            judgement = judge_test(query=query, answer=rag_answer, ideal_answer=ideal_answer)

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

        # Write YAML output per file
        with open(f"{OUTPUT_PATH}/{filename}", 'w') as out_file:
            yaml.dump(results_per_file, out_file, sort_keys=False, allow_unicode=True)

