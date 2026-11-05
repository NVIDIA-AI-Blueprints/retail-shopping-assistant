# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score recorded replay conversations against rubrics with NeMo Evaluator.

Replay already ran each conversation against the agent, so nothing is sent to
the agent again: each recorded conversation is the "response" NeMo Evaluator
scores, and a judge model grades it once per rubric.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
from dataclasses import dataclass
from getpass import getpass
from pathlib import Path
from urllib.parse import urlsplit

import yaml
from helpers import REPO
from nemo_evaluator import ModelClient, run_evaluation
from nemo_evaluator.environments.base import SeedResult
from nemo_evaluator.environments.custom import benchmark, scorer
from nemo_evaluator.environments.registry import get_environment
from nemo_evaluator.scoring import ScorerInput
from nemo_evaluator.scoring.judge import JudgeScoringConfig, judge_score
from nemo_evaluator.solvers import SolveResult

SCRIPTS = REPO / "tests" / "evaluation" / "datasets" / "val" / "scripts"

_JUDGE_PROMPT = """You are grading one conversation between a shopper and the
shopping assistant of a fashion store.

What the conversation tests: {{instruction}}

The conversation. "Shown" lists the catalog facts of each product the system
displayed; "Cart" is the cart the system recorded after the turn. Both are
ground truth, not the assistant's claims:

{{response}}
{{reference_section}}
Grade only this rubric:

{rubric}

Respond with JSON only, "score" 1 for pass and 0 for fail:
{{{{"score": <1 or 0>, "reasoning": "<one or two sentences>"}}}}"""


def _goal(scenario_id: str) -> str:
    """The first paragraph of the scenario's `why`: what it protects."""
    path = next(SCRIPTS.rglob(f"{scenario_id}.yaml"), None)
    if path is None:
        return "A shopper talks to the assistant."
    why = yaml.safe_load(path.read_text()).get("why", "")
    return " ".join(why.strip().split("\n")[0].split())


_NOT_FACTS = {"catalog_text", "similarity", "taxonomy", "category", "subcategory"}


def _product(position: int, product: dict) -> str:
    facts = "; ".join(
        f"{key.replace('_', ' ')}: {', '.join(value) if isinstance(value, list) else value}"
        for key, value in (product.get("attributes") or {}).items()
        if key not in _NOT_FACTS
    )
    price = (product.get("price") or {}).get("amount", "?")
    return (
        f"  {position}. {product.get('display_name')}, ${price}. {facts}.\n"
        f"     {product.get('description', '')}"
    )


def _transcript(result: dict) -> str:
    lines = []
    for turn in result["turns"]:
        shown = [_product(i, p) for i, p in enumerate(turn["products"], 1)]
        cart = ", ".join(
            f"{line['item']} size {line.get('size', '-')} qty {line.get('amount', 1)}"
            for line in turn["cart"]
        )
        lines += [
            f"Turn {turn['index']}",
            f"Shopper: {turn['said']}",
            "Shown (catalog facts):" if shown else "Shown: none",
            *shown,
            f"Assistant: {turn['reply'].strip()}",
            f"Cart: {cart or 'empty'}",
            "",
        ]
    return "\n".join(lines)


def conversations(*run_dirs: Path) -> list[dict]:
    """One row per recorded conversation in the given replay result folders."""
    rows = []
    for run_dir in run_dirs:
        for path in sorted((run_dir / "raw").glob("*.json")):
            result = json.loads(path.read_text())
            rows.append(
                {
                    "scenario": result["id"],
                    "repeat": result["repeat"],
                    "replay": result["outcome"],
                    "goal": _goal(result["id"]),
                    "transcript": _transcript(result),
                }
            )
    return rows


@dataclass
class RecordedConversation:
    """A NeMo Evaluator solver that returns the recorded conversation."""

    async def solve(self, task: SeedResult) -> SolveResult:
        return SolveResult(response=task.metadata["transcript"])


def _verdict(result: dict) -> dict:
    if result.get("parse_error"):
        return {"verdict": "error", "reasoning": f"unreadable judge reply: {result['raw'][:200]}"}
    return {"verdict": "pass" if result["score"] >= 1 else "fail", "reasoning": result["reasoning"]}


def _rubric_scorer(rubrics: dict[str, str]):
    @scorer
    def score(sample: ScorerInput) -> dict:
        async def judge(client) -> dict:
            grades = {}
            for name, rubric in rubrics.items():
                template = _JUDGE_PROMPT.format(rubric=rubric.replace("{", "{{").replace("}", "}}"))
                grades[name] = _verdict(
                    await judge_score(
                        instruction=sample.metadata["goal"],
                        response=sample.response,
                        client=client,
                        config=JudgeScoringConfig(rubric_template=template, max_score=1),
                    )
                )
            passed = sum(g["verdict"] == "pass" for g in grades.values())
            return {"reward": passed / len(grades), "judge": grades}

        return {"correct": False, "needs_judge": True, "_judge_fn": judge}

    return score


def _agent_setting(name: str) -> str:
    return subprocess.run(
        ["docker", "exec", "orchestrator", "printenv", name], capture_output=True, text=True
    ).stdout.strip()


def judge_from_env() -> ModelClient:
    """The judge, from the first of these that sets a base URL:

    1. `JUDGE_MODEL_*` in this kernel, an override for this session.
    2. `JUDGE_MODEL_*` in the running orchestrator: the judge the deployment's
       env profile chose, however Jupyter was started.
    3. `LLM_*` in the running orchestrator, the agent's own model, for a
       deployment from before the judge was passed through.

    Each source's endpoint, model, and key are used together, so a key is never
    sent to an endpoint it is not for.
    """
    if os.environ.get("JUDGE_MODEL_BASE_URL"):
        base_url = os.environ["JUDGE_MODEL_BASE_URL"]
        model = os.environ["JUDGE_MODEL_NAME"]
        api_key = os.environ.get("JUDGE_MODEL_API_KEY") or os.environ.get("NVIDIA_API_KEY")
    elif base_url := _agent_setting("JUDGE_MODEL_BASE_URL"):
        model = _agent_setting("JUDGE_MODEL_NAME")
        api_key = _agent_setting("JUDGE_MODEL_API_KEY")
    else:
        base_url, model = _agent_setting("LLM_BASE_URL"), _agent_setting("LLM_MODEL")
        api_key = _agent_setting("LLM_API_KEY")
    try:
        socket.getaddrinfo(urlsplit(base_url).hostname, None)
    except (socket.gaierror, UnicodeError) as error:
        raise RuntimeError(
            f"The judge's host in {base_url!r} does not resolve here. A Compose service "
            "name such as local-llm resolves only inside Docker. Recreate the app from "
            "the shell that sourced your env profile (docker compose up -d), which passes "
            "the profile's JUDGE_MODEL_* to the orchestrator, or set JUDGE_MODEL_* in "
            "this kernel. tests/evaluation/README.md, Judge Commands, has the options."
        ) from error
    return ModelClient(
        base_url=base_url,
        model=model,
        api_key=api_key or getpass(f"API key for {base_url}: "),
        temperature=0,
        max_tokens=4096,
    )


async def grade_conversations(
    rows: list[dict],
    rubrics: dict[str, str],
    judge: ModelClient | None = None,
    name: str = "shopper-rubrics",
) -> list[dict]:
    """Grade every row on every rubric: pass, fail, or error, with the judge's reason."""
    judge = judge or judge_from_env()
    benchmark(name=name, dataset=lambda: rows, prompt="{goal}")(_rubric_scorer(rubrics))
    try:
        bundle = await run_evaluation(
            get_environment(name), RecordedConversation(), judge_client=judge, max_concurrent=2
        )
    finally:
        await judge.close()
    graded = []
    for result in sorted(bundle["_results"], key=lambda r: r["problem_idx"]):
        row = rows[result["problem_idx"]]
        grades = result["scoring_details"].get("judge", {})
        if "error" in grades:
            grades = {name: {"verdict": "error", "reasoning": grades["error"]} for name in rubrics}
        graded.append({**row, "grades": grades, "passed": result["reward"]})
    return graded
