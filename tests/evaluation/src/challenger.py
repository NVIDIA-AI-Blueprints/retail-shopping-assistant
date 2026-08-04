"""Scenario-driven Challenger runner for live shopping-agent conversations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence
from urllib.parse import urlparse
import argparse
import base64
import hashlib
import json
import mimetypes
import random
import re
import shutil
import sys

import requests
import yaml

try:  # Support both ``python -m src.challenger`` and direct script execution.
    from .config import (
        ConfigError,
        EvalConfig,
        ModelRuntime,
        chat_completion_options,
        load_eval_config,
        resolve_model_runtime,
        snapshot_eval_config,
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
        snapshot_eval_config,
        write_yaml,
    )
    from src.report import generate_report  # type: ignore[no-redef]


@dataclass(frozen=True)
class ImageAsset:
    id: str
    metadata_path: Path
    file_path: Path
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ScenarioContext:
    dataset: str
    dataset_dir: Path
    scenario: dict[str, Any]
    image_asset: Optional[ImageAsset] = None


@dataclass(frozen=True)
class ShopperTurn:
    message: str
    goal_complete: bool = False


class ChallengerTurnError(RuntimeError):
    """Raised after Challenger turn generation retries are exhausted."""

    def __init__(self, message: str, errors: list[str]) -> None:
        super().__init__(message)
        self.errors = errors


class Challenger(Protocol):
    def next_turn(
        self,
        *,
        scenario: Mapping[str, Any],
        dataset: str,
        image_asset: Optional[ImageAsset],
        transcript: list[dict[str, Any]],
        turn_number: int,
        target_turns: int,
        min_turns: int,
    ) -> ShopperTurn:
        """Generate the next shopper turn."""


class TargetAgent(Protocol):
    def send_turn(self, *, user_id: int, query: str, image: str) -> dict[str, Any]:
        """Send one shopper turn to the target shopping assistant."""


CHALLENGER_TURN_ATTEMPTS = 3
MAX_SHOPPER_MESSAGE_CHARS = 1200
MAX_SHOPPER_MESSAGE_LINES = 8
MAX_PRODUCT_EVIDENCE_RECORDS = 24
MAX_PRODUCT_EVIDENCE_FIELDS = 40
MAX_PRODUCT_EVIDENCE_STRING_CHARS = 500
MAX_PRODUCT_EVIDENCE_SERIALIZED_CHARS = 32_000
MAX_PRODUCT_EVIDENCE_DEPTH = 4
MAX_CATALOG_SCOPE_OUTCOMES = 8
MAX_CATALOG_SCOPE_OUTCOMES_SERIALIZED_CHARS = 8_000
PRODUCT_EVIDENCE_FIELDS = frozenset(
    {
        "product_ref",
        "product_name",
        "source_tool",
        "evidence_type",
        "facts",
        "search_scope",
    }
)
PRODUCT_EVIDENCE_SOURCE_TYPES = {
    "search_catalog_tool": "search_result",
    "get_product_details_tool": "product_detail",
}
EVALUATION_METADATA_KEYS = frozenset(
    {
        "assistant_last",
        "catalog_dependency",
        "constraints",
        "conversation",
        "current_turn",
        "entry_mode",
        "failure_modes",
        "image_asset",
        "input_assets",
        "language_cues",
        "min_turns",
        "secondary_entry_pattern",
        "shopper_behavior",
        "shopper_goal",
        "skill_focus",
        "success_criteria",
        "target_turns",
        "turn_sequence",
    }
)


class OpenAICompatibleChallenger:
    """Generate shopper turns with the configured OpenAI-compatible model."""

    def __init__(self, runtime: ModelRuntime) -> None:
        from openai import OpenAI

        self._runtime = runtime
        self._client = OpenAI(
            base_url=runtime.base_url,
            api_key=runtime.api_key or "not-needed",
            timeout=runtime.timeout_seconds,
        )

    def next_turn(
        self,
        *,
        scenario: Mapping[str, Any],
        dataset: str,
        image_asset: Optional[ImageAsset],
        transcript: list[dict[str, Any]],
        turn_number: int,
        target_turns: int,
        min_turns: int,
    ) -> ShopperTurn:
        prompt = _build_challenger_prompt(
            scenario=scenario,
            dataset=dataset,
            image_asset=image_asset,
            transcript=transcript,
            turn_number=turn_number,
            target_turns=target_turns,
            min_turns=min_turns,
        )
        response = self._client.chat.completions.create(
            model=self._runtime.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Challenger shopper simulator for a retail "
                        "shopping assistant evaluation. Generate exactly one "
                        "realistic shopper message at a time. Return only valid JSON. "
                        "Do not include reasoning, analysis, markdown, or <think> tags."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self._runtime.temperature,
            max_tokens=self._runtime.max_tokens,
            **chat_completion_options(self._runtime),
        )
        content = _message_text(response.choices[0].message)
        return _parse_challenger_turn(content)


class TargetAgentClient:
    """HTTP client for the target Retail Shopping Assistant."""

    def __init__(self, config: EvalConfig) -> None:
        self._url = _join_url(config.target_agent.base_url, config.target_agent.endpoint)
        self._timeout = config.target_agent.timeout_seconds
        self._guardrails = config.target_agent.guardrails
        self._recorded_diagnostics = tuple(config.run.recorded_diagnostics)

    def send_turn(self, *, user_id: int, query: str, image: str) -> dict[str, Any]:
        payload = {
            "user_id": user_id,
            "query": query,
            "image": image,
            "image_bool": bool(image),
            "guardrails": self._guardrails,
        }
        response = requests.post(self._url, json=payload, timeout=self._timeout)
        response.raise_for_status()
        data = response.json()
        record: dict[str, Any] = {
            "status_code": response.status_code,
            "response": data.get("response", ""),
            "images": data.get("images", {}) or {},
            "cart": data.get("cart", {}) or {},
            "timings": data.get("timings", {}) or {},
        }
        record.update(_recorded_diagnostics(data, self._recorded_diagnostics))
        return record


def _recorded_diagnostics(
    data: Mapping[str, Any],
    fields: Sequence[str],
) -> dict[str, Any]:
    """Record the configured diagnostic fields from one target response.

    The runtime emits far more than was previously kept. Recording three fields
    meant a failing turn could not be explained afterwards: which skill was
    active, how many searches it spent, whether a call was rejected. Every
    diagnosis was reconstructed by reproducing the conversation live.

    A field the target did not send is recorded as absent rather than as an
    empty value, so "the runtime returned nothing" stays distinguishable from
    "the runtime was never asked".
    """

    diagnostics = data.get("agent_diagnostics")
    if not isinstance(diagnostics, Mapping):
        return {}
    return {name: diagnostics[name] for name in fields if name in diagnostics}


def _extract_product_evidence(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    diagnostics = data.get("agent_diagnostics")
    if not isinstance(diagnostics, Mapping):
        return []
    evidence = diagnostics.get("product_evidence")
    if not _valid_product_evidence_list(evidence):
        return []
    return evidence


def _extract_product_evidence_truncated(data: Mapping[str, Any]) -> bool:
    diagnostics = data.get("agent_diagnostics")
    if not isinstance(diagnostics, Mapping):
        return False
    evidence = diagnostics.get("product_evidence")
    if not _valid_product_evidence_list(evidence):
        return False
    truncated = diagnostics.get("product_evidence_truncated")
    return truncated if isinstance(truncated, bool) else False


def _extract_catalog_scope_outcomes(
    data: Mapping[str, Any],
) -> list[dict[str, Any]]:
    diagnostics = data.get("agent_diagnostics")
    if not isinstance(diagnostics, Mapping):
        return []
    outcomes = diagnostics.get("catalog_scope_outcomes")
    if (
        not isinstance(outcomes, list)
        or len(outcomes) > MAX_CATALOG_SCOPE_OUTCOMES
        or len(json.dumps(outcomes, sort_keys=True, default=str))
        > MAX_CATALOG_SCOPE_OUTCOMES_SERIALIZED_CHARS
    ):
        return []
    allowed_fields = {
        "outcome",
        "requested_product_type",
        "taxonomy",
        "confirmed_filters",
    }
    for outcome in outcomes:
        if (
            not isinstance(outcome, Mapping)
            or not set(outcome).issubset(allowed_fields)
            or outcome.get("outcome")
            not in {"no_direct_catalog_match", "zero_results"}
            or not _bounded_product_evidence_value(outcome)
        ):
            return []
    return outcomes


def _valid_product_evidence_list(evidence: Any) -> bool:
    if not isinstance(evidence, list) or len(evidence) > MAX_PRODUCT_EVIDENCE_RECORDS:
        return False
    if not all(_valid_product_evidence_record(record) for record in evidence):
        return False
    return len(json.dumps(evidence, sort_keys=True, default=str)) <= (
        MAX_PRODUCT_EVIDENCE_SERIALIZED_CHARS
    )


def _valid_product_evidence_record(record: Any) -> bool:
    if not isinstance(record, Mapping):
        return False
    if not set(record).issubset(PRODUCT_EVIDENCE_FIELDS):
        return False
    for field in ("product_ref", "product_name", "source_tool"):
        value = record.get(field)
        if (
            not isinstance(value, str)
            or not value
            or len(value) > MAX_PRODUCT_EVIDENCE_STRING_CHARS
        ):
            return False
    source_tool = record["source_tool"]
    evidence_type = record.get("evidence_type")
    if (
        not isinstance(evidence_type, str)
        or PRODUCT_EVIDENCE_SOURCE_TYPES.get(source_tool) != evidence_type
    ):
        return False
    facts = record.get("facts")
    if not isinstance(facts, Mapping) or len(facts) > MAX_PRODUCT_EVIDENCE_FIELDS:
        return False
    search_scope = record.get("search_scope")
    if evidence_type == "search_result":
        if not isinstance(search_scope, Mapping) or set(search_scope) != {
            "taxonomy",
            "confirmed_filters",
        }:
            return False
        if not all(isinstance(value, Mapping) for value in search_scope.values()):
            return False
    elif "search_scope" in record:
        return False
    return _bounded_product_evidence_value(record)


def _bounded_product_evidence_value(value: Any, *, depth: int = 0) -> bool:
    if depth > MAX_PRODUCT_EVIDENCE_DEPTH:
        return False
    if isinstance(value, str):
        return len(value) <= MAX_PRODUCT_EVIDENCE_STRING_CHARS
    if value is None or isinstance(value, (bool, int, float)):
        return True
    if isinstance(value, Mapping):
        return len(value) <= MAX_PRODUCT_EVIDENCE_FIELDS and all(
            isinstance(key, str)
            and len(key) <= MAX_PRODUCT_EVIDENCE_STRING_CHARS
            and _bounded_product_evidence_value(item, depth=depth + 1)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return len(value) <= MAX_PRODUCT_EVIDENCE_FIELDS and all(
            _bounded_product_evidence_value(item, depth=depth + 1)
            for item in value
        )
    return False


def load_scenario_contexts(
    config: EvalConfig,
    *,
    datasets: Optional[list[str]] = None,
    scenario_ids: Optional[set[str]] = None,
    scenario_limit_per_dataset: Optional[int] = None,
) -> list[ScenarioContext]:
    """Load scenario YAML and resolve image sidecars without duplicating them."""

    selected_datasets = datasets or config.run.datasets
    limit = (
        config.run.scenario_limit_per_dataset
        if scenario_limit_per_dataset is None
        else scenario_limit_per_dataset
    )
    rng = random.Random(config.run.random_seed)
    contexts: list[ScenarioContext] = []

    for dataset in selected_datasets:
        dataset_dir = config.root / "datasets" / dataset
        scenarios_path = dataset_dir / "scenarios.yaml"
        if not scenarios_path.exists():
            raise ConfigError(f"Scenario file not found: {scenarios_path}")
        with scenarios_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if data.get("version") != 1:
            raise ConfigError(f"Unsupported dataset version in {scenarios_path}")
        scenarios = data.get("scenarios")
        if not isinstance(scenarios, list):
            raise ConfigError(f"{scenarios_path} must contain a scenarios list.")

        if scenario_ids:
            scenarios = [
                scenario
                for scenario in scenarios
                if isinstance(scenario, Mapping) and scenario.get("id") in scenario_ids
            ]
        rng.shuffle(scenarios)
        if limit:
            scenarios = scenarios[:limit]

        for scenario in scenarios:
            if not isinstance(scenario, Mapping):
                raise ConfigError(f"Invalid scenario entry in {scenarios_path}")
            scenario_dict = dict(scenario)
            image_asset = None
            if "image_id" in scenario_dict:
                image_asset = load_image_asset(dataset_dir, str(scenario_dict["image_id"]))
            contexts.append(
                ScenarioContext(
                    dataset=dataset,
                    dataset_dir=dataset_dir,
                    scenario=scenario_dict,
                    image_asset=image_asset,
                )
            )

    if scenario_ids:
        found = {context.scenario["id"] for context in contexts}
        missing = sorted(scenario_ids - found)
        if missing:
            raise ConfigError("Scenario ids not found: " + ", ".join(missing))

    return contexts


def load_image_asset(dataset_dir: Path, image_id: str) -> ImageAsset:
    assets_dir = dataset_dir / "assets"
    matches: list[ImageAsset] = []
    for sidecar_path in sorted(assets_dir.glob("*.yaml")):
        with sidecar_path.open("r", encoding="utf-8") as handle:
            metadata = yaml.safe_load(handle) or {}
        if metadata.get("id") != image_id:
            continue
        file_name = metadata.get("file")
        if not isinstance(file_name, str) or not file_name:
            raise ConfigError(f"Image asset {sidecar_path} is missing file.")
        file_path = assets_dir / file_name
        if not file_path.exists():
            raise ConfigError(f"Image asset file not found: {file_path}")
        matches.append(
            ImageAsset(
                id=image_id,
                metadata_path=sidecar_path,
                file_path=file_path,
                metadata=dict(metadata),
            )
        )
    if not matches:
        raise ConfigError(f"Image asset id not found: {image_id}")
    if len(matches) > 1:
        raise ConfigError(f"Duplicate image asset id found: {image_id}")
    return matches[0]


def run_challenger(
    config: EvalConfig,
    *,
    datasets: Optional[list[str]] = None,
    scenario_ids: Optional[set[str]] = None,
    scenario_limit_per_dataset: Optional[int] = None,
    output_root: Optional[Path] = None,
    challenger: Optional[Challenger] = None,
    target: Optional[TargetAgent] = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run selected scenarios and persist a ``run.yaml`` plus reports."""

    contexts = load_scenario_contexts(
        config,
        datasets=datasets,
        scenario_ids=scenario_ids,
        scenario_limit_per_dataset=scenario_limit_per_dataset,
    )
    if dry_run:
        return {
            "scenario_count": len(contexts),
            "estimated_target_turns": sum(_target_turns(config, item.scenario) for item in contexts),
            "scenarios": [
                {"dataset": item.dataset, "id": item.scenario.get("id")} for item in contexts
            ],
        }

    if challenger is None:
        runtime = resolve_model_runtime(config.challenger_model, require=True)
        live_challenger = OpenAICompatibleChallenger(runtime)
    else:
        live_challenger = challenger
    live_target = target or TargetAgentClient(config)

    run_id = _new_run_id()
    results_root = output_root or config.root / "results"
    run_dir = results_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    run_record: dict[str, Any] = {
        "version": 1,
        "run_id": run_id,
        "started_at": _utc_now(),
        "completed_at": None,
        "config": snapshot_eval_config(config),
        "summary": {
            "scenario_count": len(contexts),
            "completed_scenarios": 0,
            "errored_scenarios": 0,
        },
        "scenarios": [],
    }
    run_path = run_dir / "run.yaml"
    write_yaml(run_path, run_record)

    completed = 0
    errored = 0
    for context in contexts:
        scenario_record = run_scenario(
            config=config,
            context=context,
            challenger=live_challenger,
            target=live_target,
            run_id=run_id,
            run_dir=run_dir,
        )
        run_record["scenarios"].append(scenario_record)
        if scenario_record.get("error"):
            errored += 1
        else:
            completed += 1
        run_record["summary"]["completed_scenarios"] = completed
        run_record["summary"]["errored_scenarios"] = errored
        write_yaml(run_path, run_record)

    run_record["completed_at"] = _utc_now()
    write_yaml(run_path, run_record)
    generate_report(run_path)
    return run_record


def run_scenario(
    *,
    config: EvalConfig,
    context: ScenarioContext,
    challenger: Challenger,
    target: TargetAgent,
    run_id: str,
    run_dir: Path,
) -> dict[str, Any]:
    scenario = context.scenario
    scenario_id = _required_scenario_id(scenario)
    target_turns = _target_turns(config, scenario)
    user_id = _scenario_user_id(run_id, scenario_id)
    image_asset = context.image_asset

    record: dict[str, Any] = {
        "id": scenario_id,
        "dataset": context.dataset,
        "user_id": user_id,
        "target_turns": target_turns,
        "brief": scenario.get("brief"),
        "shopper_goal": scenario.get("shopper_goal"),
        "constraints": scenario.get("constraints", []),
        "shopper_behavior": scenario.get("shopper_behavior", {}),
        "language_cues": scenario.get("language_cues", []),
        "entry_mode": scenario.get("entry_mode"),
        "secondary_entry_pattern": scenario.get("secondary_entry_pattern"),
        "skill_focus": scenario.get("skill_focus", []),
        "catalog_dependency": scenario.get("catalog_dependency"),
        "success_criteria": scenario.get("success_criteria", []),
        "failure_modes": scenario.get("failure_modes", []),
        "image_id": scenario.get("image_id"),
        "image_asset": _image_asset_record(image_asset),
        "input_assets": [],
        "turns": [],
        "error": None,
    }
    if image_asset:
        record["input_assets"].append(_copy_input_asset(image_asset, run_dir))

    transcript: list[dict[str, Any]] = []
    for turn_number in range(1, target_turns + 1):
        try:
            shopper_turn, retry_errors = _next_challenger_turn(
                challenger=challenger,
                scenario=scenario,
                dataset=context.dataset,
                image_asset=image_asset,
                transcript=transcript,
                turn_number=turn_number,
                target_turns=target_turns,
                min_turns=config.conversation.min_turns,
            )
            if retry_errors:
                record.setdefault("challenger_retry_errors", []).append(
                    {"turn": turn_number, "errors": retry_errors}
                )
        except ChallengerTurnError as exc:
            record["challenger_retry_errors"] = record.get(
                "challenger_retry_errors", []
            ) + [{"turn": turn_number, "errors": exc.errors}]
            if _can_stop_after_challenger_exhaustion(
                transcript=transcript,
                min_turns=config.conversation.min_turns,
            ):
                record["stopped_reason"] = "challenger_exhausted_after_partial_completion"
                break
            record["error"] = f"challenger_error: {exc}"
            break
        except Exception as exc:  # noqa: BLE001 - preserve run artifact.
            record["error"] = f"challenger_error: {exc}"
            break

        if (
            shopper_turn.goal_complete
            and config.conversation.stop_when_goal_complete
            and len(transcript) >= config.conversation.min_turns
        ):
            record["stopped_reason"] = "goal_complete"
            break

        if not shopper_turn.message:
            record["error"] = "challenger_error: empty shopper message before goal completion"
            break

        image_payload = _image_to_data_uri(image_asset.file_path) if image_asset and turn_number == 1 else ""
        turn_record: dict[str, Any] = {
            "turn": turn_number,
            "shopper": shopper_turn.message,
            "image_sent": bool(image_payload),
            "target": None,
            "returned_assets": [],
        }
        try:
            target_response = target.send_turn(
                user_id=user_id,
                query=shopper_turn.message,
                image=image_payload,
            )
            if config.run.save_returned_images:
                turn_record["returned_assets"] = _copy_returned_images(
                    target_response.get("images", {}),
                    run_dir=run_dir,
                    turn_number=turn_number,
                )
            turn_record["target"] = target_response
        except Exception as exc:  # noqa: BLE001 - preserve run artifact.
            turn_record["target"] = {"error": str(exc)}
            record["turns"].append(turn_record)
            record["error"] = f"target_error: {exc}"
            break

        record["turns"].append(turn_record)
        transcript.append(
            {
                "turn": turn_number,
                "shopper": shopper_turn.message,
                "assistant": target_response.get("response", ""),
                "images": target_response.get("images", {}),
                "cart": target_response.get("cart", {}),
            }
        )

    return record


def _can_stop_after_challenger_exhaustion(
    *,
    transcript: list[dict[str, Any]],
    min_turns: int,
) -> bool:
    return len(transcript) >= max(1, min_turns - 1)


def _next_challenger_turn(
    *,
    challenger: Challenger,
    scenario: Mapping[str, Any],
    dataset: str,
    image_asset: Optional[ImageAsset],
    transcript: list[dict[str, Any]],
    turn_number: int,
    target_turns: int,
    min_turns: int,
) -> tuple[ShopperTurn, list[str]]:
    retry_errors: list[str] = []
    last_error = "unknown Challenger error"
    for attempt in range(1, CHALLENGER_TURN_ATTEMPTS + 1):
        try:
            shopper_turn = challenger.next_turn(
                scenario=scenario,
                dataset=dataset,
                image_asset=image_asset,
                transcript=transcript,
                turn_number=turn_number,
                target_turns=target_turns,
                min_turns=min_turns,
            )
            if shopper_turn.message:
                _validate_shopper_message(shopper_turn.message)
                return shopper_turn, retry_errors
            if shopper_turn.goal_complete and len(transcript) >= min_turns:
                return shopper_turn, retry_errors
            if shopper_turn.goal_complete:
                last_error = (
                    "goal_complete returned before minimum turns with no shopper message "
                    f"({len(transcript)}/{min_turns} turns complete)"
                )
            else:
                last_error = "empty shopper message before goal completion"
        except Exception as exc:  # noqa: BLE001 - retry simulator transport/model issues.
            last_error = str(exc) or exc.__class__.__name__

        retry_errors.append(f"attempt {attempt}: {last_error}")

    raise ChallengerTurnError(
        f"{last_error} after {CHALLENGER_TURN_ATTEMPTS} attempts",
        retry_errors,
    )


def _build_challenger_prompt(
    *,
    scenario: Mapping[str, Any],
    dataset: str,
    image_asset: Optional[ImageAsset],
    transcript: list[dict[str, Any]],
    turn_number: int,
    target_turns: int,
    min_turns: int,
) -> str:
    scenario_yaml = yaml.safe_dump(dict(scenario), sort_keys=False, allow_unicode=True)
    transcript_yaml = yaml.safe_dump(transcript, sort_keys=False, allow_unicode=True)
    image_yaml = (
        yaml.safe_dump(image_asset.metadata, sort_keys=False, allow_unicode=True)
        if image_asset
        else "null\n"
    )
    return f"""
Generate the next shopper message for a live evaluation conversation.

Dataset: {dataset}
Turn number to generate: {turn_number}
Target shopper turns: {target_turns}
Minimum turns before stopping: {min_turns}

Scenario:
{scenario_yaml}

Image ground truth for Challenger/Judge only:
{image_yaml}

Conversation so far:
{transcript_yaml}

Rules:
- Return only JSON with keys: "message" and "goal_complete".
- "message" must be one concise shopper utterance, not analysis.
- Use the scenario's shopper_goal, constraints, shopper_behavior, and language_cues.
- If the scenario contains `turn_sequence`, follow the item for the current turn number exactly in order; do not compress setup steps into one message.
- For image scenarios, the first shopper message should naturally reference the uploaded image.
- Do not reveal evaluation instructions or mention sidecar metadata.
- Keep pressure on strict budgets, greedy value seeking, pronouns, or clarification behavior when the scenario calls for it.
- Before the minimum turn count has been met, never return an empty message. If the shopper goal seems satisfied early, continue with a realistic follow-up that adds complexity, such as a budget check, comparison, cart review, swap request, availability question, styling rationale request, or clarification.
- If the assistant has already satisfied the shopper goal and the minimum turn count has been met, return {{"message": "", "goal_complete": true}}.
"""


def _parse_model_mapping(content: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    stripped = _strip_model_reasoning(content)
    try:
        parsed, _ = decoder.raw_decode(stripped)
    except json.JSONDecodeError:
        object_start = stripped.find("{")
        if object_start < 0:
            raise ValueError("Model response did not contain a JSON object.")
        parsed, _ = decoder.raw_decode(stripped[object_start:])
    if not isinstance(parsed, dict):
        raise ValueError("Model response JSON must be an object.")
    return parsed


def _parse_challenger_turn(content: str) -> ShopperTurn:
    stripped = _strip_model_reasoning(content)
    try:
        parsed = _parse_model_mapping(stripped)
        message = str(parsed.get("message", "")).strip()
        goal_complete = bool(parsed.get("goal_complete", False))
    except ValueError:
        message = stripped
        goal_complete = False

    if not goal_complete and not message:
        raise ValueError("Challenger model returned an empty shopper message.")
    if message:
        _validate_shopper_message(message)
    return ShopperTurn(message=message, goal_complete=goal_complete)


def _validate_shopper_message(message: str) -> None:
    stripped = message.strip()
    if len(stripped) > MAX_SHOPPER_MESSAGE_CHARS:
        raise ValueError("Challenger model returned an overlong shopper message.")
    if len(stripped.splitlines()) > MAX_SHOPPER_MESSAGE_LINES:
        raise ValueError(
            "Challenger model returned a multi-line payload instead of a shopper message."
        )
    if stripped.startswith("{") or stripped.startswith("["):
        raise ValueError(
            "Challenger model returned a structured payload instead of a shopper message."
        )

    parsed = _parse_optional_mapping(stripped)
    if _contains_evaluation_metadata(parsed):
        raise ValueError(
            "Challenger model returned evaluation metadata instead of a shopper message."
        )


def _parse_optional_mapping(message: str) -> Any:
    try:
        return yaml.safe_load(message)
    except yaml.YAMLError:
        return None


def _contains_evaluation_metadata(value: Any) -> bool:
    if isinstance(value, Mapping):
        keys = {str(key) for key in value}
        if keys & EVALUATION_METADATA_KEYS:
            return True
        return any(_contains_evaluation_metadata(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_evaluation_metadata(item) for item in value)
    return False


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


def _strip_model_reasoning(content: str) -> str:
    stripped = content.strip()
    stripped = re.sub(r"<think\b[^>]*>.*?</think>", "", stripped, flags=re.IGNORECASE | re.DOTALL)
    last_close = stripped.lower().rfind("</think>")
    if last_close >= 0:
        stripped = stripped[last_close + len("</think>") :]
    open_match = re.search(r"<think\b[^>]*>", stripped, flags=re.IGNORECASE)
    if open_match:
        stripped = stripped[: open_match.start()]
    return stripped.strip()


def _target_turns(config: EvalConfig, scenario: Mapping[str, Any]) -> int:
    raw_turns = scenario.get("target_turns", config.conversation.default_turns)
    if not isinstance(raw_turns, int):
        raise ConfigError(f"Scenario {scenario.get('id')} target_turns must be an integer.")
    return min(config.conversation.max_turns, max(config.conversation.min_turns, raw_turns))


def _required_scenario_id(scenario: Mapping[str, Any]) -> str:
    scenario_id = scenario.get("id")
    if not isinstance(scenario_id, str) or not scenario_id:
        raise ConfigError("Scenario is missing id.")
    return scenario_id


def _scenario_user_id(run_id: str, scenario_id: str) -> int:
    digest = hashlib.sha256(f"{run_id}:{scenario_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 900000000 + 100000


def _new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _join_url(base_url: str, endpoint: str) -> str:
    return f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"


def _image_asset_record(image_asset: Optional[ImageAsset]) -> Optional[dict[str, Any]]:
    if not image_asset:
        return None
    return {
        "id": image_asset.id,
        "file": image_asset.metadata.get("file"),
        "metadata": image_asset.metadata,
    }


def _copy_input_asset(image_asset: ImageAsset, run_dir: Path) -> dict[str, Any]:
    destination = run_dir / "assets" / "input" / image_asset.file_path.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_asset.file_path, destination)
    return {
        "id": image_asset.id,
        "source": str(image_asset.file_path),
        "copied_to": _relative_to_run(destination, run_dir),
    }


def _copy_returned_images(
    images: Mapping[str, Any],
    *,
    run_dir: Path,
    turn_number: int,
) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    if not isinstance(images, Mapping):
        return copied

    for name, value in images.items():
        if not isinstance(value, str) or not value:
            continue
        safe_name = _safe_name(str(name))
        if value.startswith("data:image/"):
            copied.append(_copy_data_uri(value, safe_name, run_dir, turn_number))
            continue
        local_source = _resolve_local_image_source(value, run_dir)
        if local_source:
            destination = (
                run_dir
                / "assets"
                / "returned"
                / f"turn-{turn_number:02d}-{safe_name}{local_source.suffix}"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_source, destination)
            copied.append(
                {
                    "name": str(name),
                    "source": value,
                    "copied_to": _relative_to_run(destination, run_dir),
                }
            )
        else:
            copied.append({"name": str(name), "source": value, "copied_to": None})
    return copied


def _copy_data_uri(data_uri: str, safe_name: str, run_dir: Path, turn_number: int) -> dict[str, Any]:
    header, data = data_uri.split(",", 1)
    mime_type = header.removeprefix("data:").split(";")[0]
    extension = ".jpg" if mime_type == "image/jpeg" else mimetypes.guess_extension(mime_type) or ".img"
    destination = run_dir / "assets" / "returned" / f"turn-{turn_number:02d}-{safe_name}{extension}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(base64.b64decode(data))
    return {"name": safe_name, "source": "data_uri", "copied_to": _relative_to_run(destination, run_dir)}


def _resolve_local_image_source(value: str, run_dir: Path) -> Optional[Path]:
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"}:
        return None

    candidates: list[Path] = []
    raw_path = Path(value)
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        repo_root = run_dir.parents[4] if len(run_dir.parents) >= 5 else Path.cwd()
        stripped = value.lstrip("/")
        candidates.extend(
            [
                repo_root / stripped,
                repo_root / "shared" / "images" / Path(stripped).name,
                repo_root / "ui" / "public" / stripped,
            ]
        )
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _image_to_data_uri(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip().lower()).strip("-")
    return cleaned or "image"


def _relative_to_run(path: Path, run_dir: Path) -> str:
    return str(path.relative_to(run_dir))


def _main() -> int:
    parser = argparse.ArgumentParser(description="Run evaluation Challenger scenarios.")
    parser.add_argument("--config", default=None, help="Path to eval_config.yaml.")
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="Dataset to run. May be provided multiple times.",
    )
    parser.add_argument(
        "--scenario-id",
        action="append",
        dest="scenario_ids",
        help="Scenario id to run. May be provided multiple times.",
    )
    parser.add_argument(
        "--scenario-limit",
        type=int,
        default=None,
        help="Override scenario_limit_per_dataset.",
    )
    parser.add_argument(
        "--all-scenarios",
        action="store_true",
        help="Run every scenario in the selected dataset(s), ignoring scenario_limit_per_dataset.",
    )
    parser.add_argument("--output-root", default=None, help="Override results root directory.")
    parser.add_argument("--dry-run", action="store_true", help="Validate selection without API calls.")
    parser.add_argument("--judge", action="store_true", help="Run Judge after Challenger completes.")
    args = parser.parse_args()
    if args.all_scenarios and args.scenario_ids:
        parser.error("--all-scenarios cannot be combined with --scenario-id.")
    if args.scenario_limit is not None and args.scenario_limit < 0:
        parser.error("--scenario-limit must be zero or greater.")

    config = load_eval_config(args.config)
    output_root = Path(args.output_root).resolve() if args.output_root else None
    scenario_limit = 0 if args.all_scenarios else args.scenario_limit
    run_record = run_challenger(
        config,
        datasets=args.datasets,
        scenario_ids=set(args.scenario_ids) if args.scenario_ids else None,
        scenario_limit_per_dataset=scenario_limit,
        output_root=output_root,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print(yaml.safe_dump(run_record, sort_keys=False))
        return 0

    run_id = run_record["run_id"]
    run_path = (output_root or config.root / "results") / "runs" / run_id / "run.yaml"
    if args.judge or config.judge_model.enabled:
        try:
            from .judge import judge_run
        except ImportError:  # pragma: no cover - direct CLI use.
            from src.judge import judge_run  # type: ignore[no-redef]

        judge_run(config, run_path, require_enabled=not args.judge)
    print(f"Saved evaluation run: {run_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
