"""Optional Judge phase for saved Challenger runs."""

from __future__ import annotations

from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Optional, Protocol
import argparse
import json
import sys

import yaml

try:  # Support both ``python -m src.judge`` and direct script execution.
    from .config import (
        ConfigError,
        EvalConfig,
        ModelRuntime,
        chat_completion_options,
        load_eval_config,
        resolve_model_runtime,
        write_yaml,
    )
    from .report import generate_report
except ImportError:  # pragma: no cover - exercised by direct CLI use.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.config import (  # type: ignore[no-redef]
        ConfigError,
        EvalConfig,
        ModelRuntime,
        chat_completion_options,
        load_eval_config,
        resolve_model_runtime,
        write_yaml,
    )
    from src.report import generate_report  # type: ignore[no-redef]


CRITERIA_KEYS = [
    "goal_completion",
    "relevance_helpfulness",
    "groundedness",
    "constraint_following",
    "multi_turn_context",
    "tool_state_correctness",
    "clarification_recovery",
    "safety_scope",
    "communication_quality",
    "style_composition_quality",
    "decision_boundary_quality",
]

JUDGE_SCENARIO_ATTEMPTS = 3


class Judge(Protocol):
    def judge_scenario(
        self,
        *,
        rules: str,
        run: Mapping[str, Any],
        scenario: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Return a structured scenario judgment."""


class OpenAICompatibleJudge:
    """Judge saved conversations with the configured OpenAI-compatible model."""

    def __init__(self, runtime: ModelRuntime) -> None:
        from openai import OpenAI

        self._runtime = runtime
        self._client = OpenAI(
            base_url=runtime.base_url,
            api_key=runtime.api_key or "not-needed",
            timeout=runtime.timeout_seconds,
        )

    def judge_scenario(
        self,
        *,
        rules: str,
        run: Mapping[str, Any],
        scenario: Mapping[str, Any],
    ) -> dict[str, Any]:
        prompt = _build_judge_prompt(rules=rules, run=run, scenario=scenario)
        last_error = "unknown Judge error"
        for _attempt in range(1, JUDGE_SCENARIO_ATTEMPTS + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._runtime.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a strict retail shopping assistant evaluation Judge. "
                                "Apply the supplied rules and return only valid JSON."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self._runtime.temperature,
                    max_tokens=self._runtime.max_tokens,
                    **chat_completion_options(self._runtime),
                )
                content = _message_text(response.choices[0].message)
                return _normalize_judgment(_parse_model_mapping(content))
            except Exception as exc:  # noqa: BLE001 - retry model/provider noncompliance.
                last_error = str(exc) or exc.__class__.__name__
        raise ValueError(
            f"{last_error} after {JUDGE_SCENARIO_ATTEMPTS} Judge attempts"
        )


def judge_run(
    config: EvalConfig,
    run_path: str | Path,
    *,
    judge: Optional[Judge] = None,
    require_enabled: bool = True,
) -> dict[str, Any]:
    """Apply the configured Judge to each scenario in ``run.yaml``."""

    path = Path(run_path)
    with path.open("r", encoding="utf-8") as handle:
        run = yaml.safe_load(handle) or {}
    if not isinstance(run, dict):
        raise ValueError(f"Run file must contain a YAML mapping: {path}")

    scenarios = run.get("scenarios", [])
    if not isinstance(scenarios, list):
        raise ValueError("Run file scenarios must be a list.")

    live_judge = judge
    if live_judge is None:
        if require_enabled and not config.judge_model.enabled:
            raise ConfigError("judge_model.enabled is false in eval_config.yaml.")
        runtime = resolve_model_runtime(config.judge_model, require=True)
        live_judge = OpenAICompatibleJudge(runtime)

    rules = _load_rules(config)
    scores: list[int] = []
    passes = 0
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            continue
        if _scenario_has_evaluation_error(scenario):
            judgment = _evaluation_error_judgment(scenario)
        else:
            try:
                judgment = live_judge.judge_scenario(
                    rules=rules, run=run, scenario=scenario
                )
            except Exception as exc:  # noqa: BLE001 - preserve partial judged run.
                judgment = _judge_error_judgment(exc)
        normalized = _normalize_judgment(judgment)
        scenario["judge"] = normalized
        scores.append(int(normalized["score"]))
        if normalized["pass"]:
            passes += 1
        _write_judged_run(path, run, scores=scores, passes=passes)

    _write_judged_run(path, run, scores=scores, passes=passes)
    generate_report(path)
    return run


def _scenario_has_evaluation_error(scenario: Mapping[str, Any]) -> bool:
    turns = scenario.get("turns", [])
    if scenario.get("error"):
        return True
    return not isinstance(turns, list) or not turns


def _evaluation_error_judgment(scenario: Mapping[str, Any]) -> dict[str, Any]:
    reason = str(scenario.get("error") or "scenario produced no target turns").strip()
    return {
        "score": 1,
        "pass": False,
        "reason": f"Evaluation incomplete: {reason}.",
        "criteria": {key: 1 for key in CRITERIA_KEYS},
        "critical_failures": [reason],
    }


def _judge_error_judgment(exc: Exception) -> dict[str, Any]:
    reason = f"Judge failed to score this scenario: {exc}"
    return {
        "score": 1,
        "pass": False,
        "reason": reason,
        "criteria": {key: 1 for key in CRITERIA_KEYS},
        "critical_failures": [reason],
    }


def _write_judged_run(path: Path, run: dict[str, Any], *, scores: list[int], passes: int) -> None:
    scenario_count = len(scores)
    run["judge_summary"] = {
        "scenario_count": scenario_count,
        "pass_count": passes,
        "fail_count": scenario_count - passes,
        "average_score": round(mean(scores), 2) if scores else None,
    }
    write_yaml(path, run)


def _load_rules(config: EvalConfig) -> str:
    rules_file = config.judge_model.rules_file or "judge_rules.md"
    rules_path = config.root / rules_file
    if not rules_path.exists():
        raise ConfigError(f"Judge rules file not found: {rules_path}")
    return rules_path.read_text(encoding="utf-8")


def _build_judge_prompt(
    *,
    rules: str,
    run: Mapping[str, Any],
    scenario: Mapping[str, Any],
) -> str:
    scenario_payload = _judge_scenario_payload(scenario)
    return f"""
Apply the Judge rules to this saved live conversation.

Judge rules:
{rules}

Run id: {run.get("run_id")}

Scenario and transcript:
{yaml.safe_dump(scenario_payload, sort_keys=False, allow_unicode=True)}

Return only JSON with this shape:
{{
  "score": 4,
  "pass": true,
  "reason": "One or two concise sentences.",
  "criteria": {{
    "goal_completion": 4,
    "relevance_helpfulness": 4,
    "groundedness": 4,
    "constraint_following": 4,
    "multi_turn_context": 4,
    "tool_state_correctness": 4,
    "clarification_recovery": 4,
    "safety_scope": 5,
    "communication_quality": 4,
    "style_composition_quality": 4,
    "decision_boundary_quality": 4
  }},
  "critical_failures": []
}}
"""


def _judge_scenario_payload(scenario: Mapping[str, Any]) -> dict[str, Any]:
    turns = []
    raw_turns = scenario.get("turns", [])
    if not isinstance(raw_turns, list):
        raw_turns = []
    for turn in raw_turns:
        if not isinstance(turn, Mapping):
            continue
        target = turn.get("target") if isinstance(turn.get("target"), Mapping) else {}
        turns.append(
            {
                "turn": turn.get("turn"),
                "shopper": turn.get("shopper"),
                "image_sent": turn.get("image_sent"),
                "assistant": target.get("response"),
                "assistant_error": target.get("error"),
                "returned_images": target.get("images"),
                "cart_after": target.get("cart"),
            }
        )

    return {
        "id": scenario.get("id"),
        "dataset": scenario.get("dataset"),
        "brief": scenario.get("brief"),
        "shopper_goal": scenario.get("shopper_goal"),
        "constraints": scenario.get("constraints"),
        "shopper_behavior": scenario.get("shopper_behavior"),
        "language_cues": scenario.get("language_cues"),
        "entry_mode": scenario.get("entry_mode"),
        "secondary_entry_pattern": scenario.get("secondary_entry_pattern"),
        "skill_focus": scenario.get("skill_focus"),
        "catalog_dependency": scenario.get("catalog_dependency"),
        "success_criteria": scenario.get("success_criteria"),
        "failure_modes": scenario.get("failure_modes"),
        "image_id": scenario.get("image_id"),
        "image_asset": scenario.get("image_asset"),
        "error": scenario.get("error"),
        "turns": turns,
    }


def _parse_model_mapping(content: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    stripped = content.strip()
    try:
        parsed, _ = decoder.raw_decode(stripped)
    except json.JSONDecodeError:
        object_start = stripped.find("{")
        if object_start >= 0:
            parsed, _ = decoder.raw_decode(stripped[object_start:])
        else:
            parsed = yaml.safe_load(content)
    if not isinstance(parsed, dict):
        raise ValueError("Judge model response must be a JSON/YAML object.")
    return parsed


def _message_text(message: Any) -> str:
    content = getattr(message, "content", None) or ""
    if content:
        return str(content)

    reasoning_content = getattr(message, "reasoning_content", None)
    if reasoning_content:
        return str(reasoning_content)

    if hasattr(message, "model_dump"):
        dumped = message.model_dump()
        if isinstance(dumped, Mapping):
            reasoning_content = dumped.get("reasoning_content")
            if reasoning_content:
                return str(reasoning_content)
    return ""


def _normalize_judgment(raw: Mapping[str, Any]) -> dict[str, Any]:
    score = int(raw.get("score", 1))
    score = max(1, min(5, score))
    critical_failures = raw.get("critical_failures", [])
    if not isinstance(critical_failures, list):
        critical_failures = [str(critical_failures)]
    criteria = raw.get("criteria", {})
    if not isinstance(criteria, Mapping):
        criteria = {}

    normalized_criteria = {}
    for key in CRITERIA_KEYS:
        value = criteria.get(key, score)
        try:
            normalized_criteria[key] = max(1, min(5, int(value)))
        except (TypeError, ValueError):
            normalized_criteria[key] = score

    passed = raw.get("pass", score >= 4 and not critical_failures)
    return {
        "score": score,
        "pass": bool(passed) and not critical_failures,
        "reason": str(raw.get("reason", "")).strip(),
        "criteria": normalized_criteria,
        "critical_failures": [str(item) for item in critical_failures],
    }


def _main() -> int:
    parser = argparse.ArgumentParser(description="Judge a saved Challenger run.")
    parser.add_argument(
        "run_path",
        nargs="?",
        help="Path to tests/evaluation/results/runs/<run_id>/run.yaml",
    )
    parser.add_argument("--config", default=None, help="Path to eval_config.yaml.")
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Judge the run referenced by results/latest.txt.",
    )
    parser.add_argument(
        "--enable-judge",
        action="store_true",
        help="Run the configured judge model for this command even when judge_model.enabled is false.",
    )
    args = parser.parse_args()
    if args.latest and args.run_path:
        parser.error("--latest cannot be combined with run_path.")
    if not args.latest and not args.run_path:
        parser.error("run_path is required unless --latest is provided.")

    config = load_eval_config(args.config)
    run_path = _resolve_run_path(config, args.run_path, latest=args.latest)
    judge_run(config, run_path, require_enabled=not args.enable_judge)
    print(f"Judged evaluation run: {run_path}")
    return 0


def _resolve_run_path(config: EvalConfig, run_path: Optional[str], *, latest: bool) -> Path:
    if not latest:
        if not run_path:
            raise ConfigError("run_path is required unless --latest is provided.")
        return Path(run_path)

    latest_path = config.root / "results" / "latest.txt"
    if not latest_path.exists():
        raise ConfigError(f"Latest run file not found: {latest_path}")
    run_id = latest_path.read_text(encoding="utf-8").strip()
    if not run_id:
        raise ConfigError(f"Latest run file is empty: {latest_path}")
    return config.root / "results" / "runs" / run_id / "run.yaml"


if __name__ == "__main__":
    raise SystemExit(_main())
