"""Configuration loading for Challenger and Judge runners."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional
import os

import yaml


EVAL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = EVAL_ROOT / "eval_config.yaml"


class ConfigError(ValueError):
    """Raised when evaluation configuration is missing or invalid."""


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    base_url_env: str
    model_env: str
    api_key_env: str
    base_url: Optional[str]
    model: Optional[str]
    api_key_required: bool
    disable_thinking: bool
    json_mode: bool
    temperature: float
    max_tokens: int
    timeout_seconds: int
    enabled: bool = True
    rules_file: Optional[str] = None


@dataclass(frozen=True)
class ModelRuntime:
    provider: str
    base_url: str
    model: str
    api_key: Optional[str]
    disable_thinking: bool
    json_mode: bool
    temperature: float
    max_tokens: int
    timeout_seconds: int


@dataclass(frozen=True)
class TargetAgentConfig:
    base_url: str
    endpoint: str
    timeout_seconds: int
    guardrails: bool


@dataclass(frozen=True)
class RunConfig:
    datasets: list[str]
    scenario_limit_per_dataset: int
    random_seed: int
    save_returned_images: bool
    send_asset_descriptions_to_agent: bool
    compare_to: Optional[str]


@dataclass(frozen=True)
class ConversationConfig:
    default_turns: int
    min_turns: int
    max_turns: int
    stop_when_goal_complete: bool


@dataclass(frozen=True)
class EvalConfig:
    version: int
    root: Path
    config_path: Path
    challenger_model: ModelConfig
    judge_model: ModelConfig
    target_agent: TargetAgentConfig
    run: RunConfig
    conversation: ConversationConfig


def load_eval_config(config_path: str | Path | None = None) -> EvalConfig:
    """Load and validate ``eval_config.yaml``."""

    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    path = path.expanduser().resolve()
    if not path.exists():
        raise ConfigError(f"Evaluation config not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, Mapping):
        raise ConfigError("Evaluation config must be a YAML mapping.")

    version = _as_int(raw.get("version"), "version")
    if version != 1:
        raise ConfigError(f"Unsupported evaluation config version: {version}")

    root = path.parent
    return EvalConfig(
        version=version,
        root=root,
        config_path=path,
        challenger_model=_load_model_config(raw, "challenger_model", default_enabled=True),
        judge_model=_load_model_config(raw, "judge_model", default_enabled=False),
        target_agent=_load_target_config(raw),
        run=_load_run_config(raw),
        conversation=_load_conversation_config(raw),
    )


def resolve_model_runtime(model_config: ModelConfig, *, require: bool = True) -> ModelRuntime | None:
    """Resolve a model config's environment references without logging secrets."""

    if not model_config.enabled and not require:
        return None

    if model_config.provider != "openai_compatible":
        raise ConfigError(f"Unsupported model provider: {model_config.provider}")

    base_url = _env_or_config_value(model_config.base_url_env, model_config.base_url)
    model = _env_or_config_value(model_config.model_env, model_config.model)
    api_key = os.environ.get(model_config.api_key_env, "").strip() or None

    missing = []
    if not base_url:
        missing.append(f"{model_config.base_url_env} or config base_url")
    if not model:
        missing.append(f"{model_config.model_env} or config model")
    if model_config.api_key_required and not api_key:
        missing.append(model_config.api_key_env)
    if missing:
        if not require:
            return None
        raise ConfigError(
            "Missing required model settings: " + ", ".join(sorted(missing))
        )

    return ModelRuntime(
        provider=model_config.provider,
        base_url=base_url,
        model=model,
        api_key=api_key,
        disable_thinking=model_config.disable_thinking,
        json_mode=model_config.json_mode,
        temperature=model_config.temperature,
        max_tokens=model_config.max_tokens,
        timeout_seconds=model_config.timeout_seconds,
    )


def chat_completion_options(runtime: ModelRuntime) -> dict[str, Any]:
    """Return optional OpenAI-compatible request controls for structured output."""

    options: dict[str, Any] = {}
    if runtime.json_mode:
        options["response_format"] = {"type": "json_object"}
    if runtime.disable_thinking:
        options["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    return options


def snapshot_eval_config(config: EvalConfig) -> dict[str, Any]:
    """Return a non-secret config snapshot suitable for committed run records."""

    return {
        "version": config.version,
        "config_path": str(config.config_path),
        "challenger_model": _snapshot_model(config.challenger_model),
        "judge_model": _snapshot_model(config.judge_model),
        "target_agent": {
            "base_url": config.target_agent.base_url,
            "endpoint": config.target_agent.endpoint,
            "timeout_seconds": config.target_agent.timeout_seconds,
            "guardrails": config.target_agent.guardrails,
        },
        "run": {
            "datasets": list(config.run.datasets),
            "scenario_limit_per_dataset": config.run.scenario_limit_per_dataset,
            "random_seed": config.run.random_seed,
            "save_returned_images": config.run.save_returned_images,
            "send_asset_descriptions_to_agent": config.run.send_asset_descriptions_to_agent,
            "compare_to": config.run.compare_to,
        },
        "conversation": {
            "default_turns": config.conversation.default_turns,
            "min_turns": config.conversation.min_turns,
            "max_turns": config.conversation.max_turns,
            "stop_when_goal_complete": config.conversation.stop_when_goal_complete,
        },
    }


def write_yaml(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def _load_model_config(
    raw: Mapping[str, Any], section: str, *, default_enabled: bool
) -> ModelConfig:
    data = _required_mapping(raw, section)
    return ModelConfig(
        enabled=_as_bool(data.get("enabled", default_enabled), f"{section}.enabled"),
        provider=_as_str(data.get("provider"), f"{section}.provider"),
        base_url_env=_as_str(data.get("base_url_env"), f"{section}.base_url_env"),
        model_env=_as_str(data.get("model_env"), f"{section}.model_env"),
        api_key_env=_as_str(data.get("api_key_env"), f"{section}.api_key_env"),
        base_url=_optional_str(data.get("base_url"), f"{section}.base_url"),
        model=_optional_str(data.get("model"), f"{section}.model"),
        api_key_required=_as_bool(
            data.get("api_key_required", False), f"{section}.api_key_required"
        ),
        disable_thinking=_as_bool(
            data.get("disable_thinking", False), f"{section}.disable_thinking"
        ),
        json_mode=_as_bool(data.get("json_mode", False), f"{section}.json_mode"),
        temperature=_as_float(data.get("temperature"), f"{section}.temperature"),
        max_tokens=_as_int(data.get("max_tokens"), f"{section}.max_tokens"),
        timeout_seconds=_as_int(
            data.get("timeout_seconds", 90), f"{section}.timeout_seconds"
        ),
        rules_file=_optional_str(data.get("rules_file"), f"{section}.rules_file"),
    )


def _load_target_config(raw: Mapping[str, Any]) -> TargetAgentConfig:
    data = _required_mapping(raw, "target_agent")
    return TargetAgentConfig(
        base_url=_as_str(data.get("base_url"), "target_agent.base_url").rstrip("/"),
        endpoint=_as_str(data.get("endpoint"), "target_agent.endpoint"),
        timeout_seconds=_as_int(data.get("timeout_seconds"), "target_agent.timeout_seconds"),
        guardrails=_as_bool(data.get("guardrails", True), "target_agent.guardrails"),
    )


def _load_run_config(raw: Mapping[str, Any]) -> RunConfig:
    data = _required_mapping(raw, "run")
    datasets = data.get("datasets")
    if not isinstance(datasets, list) or not all(isinstance(item, str) for item in datasets):
        raise ConfigError("run.datasets must be a list of dataset names.")
    return RunConfig(
        datasets=datasets,
        scenario_limit_per_dataset=_as_int(
            data.get("scenario_limit_per_dataset"), "run.scenario_limit_per_dataset"
        ),
        random_seed=_as_int(data.get("random_seed"), "run.random_seed"),
        save_returned_images=_as_bool(
            data.get("save_returned_images"), "run.save_returned_images"
        ),
        send_asset_descriptions_to_agent=_as_bool(
            data.get("send_asset_descriptions_to_agent"),
            "run.send_asset_descriptions_to_agent",
        ),
        compare_to=_optional_str(data.get("compare_to"), "run.compare_to"),
    )


def _load_conversation_config(raw: Mapping[str, Any]) -> ConversationConfig:
    data = _required_mapping(raw, "conversation")
    config = ConversationConfig(
        default_turns=_as_int(data.get("default_turns"), "conversation.default_turns"),
        min_turns=_as_int(data.get("min_turns"), "conversation.min_turns"),
        max_turns=_as_int(data.get("max_turns"), "conversation.max_turns"),
        stop_when_goal_complete=_as_bool(
            data.get("stop_when_goal_complete"), "conversation.stop_when_goal_complete"
        ),
    )
    if config.min_turns < 1:
        raise ConfigError("conversation.min_turns must be at least 1.")
    if config.max_turns < config.min_turns:
        raise ConfigError("conversation.max_turns must be greater than min_turns.")
    if not config.min_turns <= config.default_turns <= config.max_turns:
        raise ConfigError("conversation.default_turns must be within min/max turns.")
    return config


def _snapshot_model(model_config: ModelConfig) -> dict[str, Any]:
    return {
        "enabled": model_config.enabled,
        "provider": model_config.provider,
        "base_url_env": model_config.base_url_env,
        "model_env": model_config.model_env,
        "api_key_env": model_config.api_key_env,
        "base_url": model_config.base_url,
        "model": model_config.model,
        "api_key_required": model_config.api_key_required,
        "disable_thinking": model_config.disable_thinking,
        "json_mode": model_config.json_mode,
        "temperature": model_config.temperature,
        "max_tokens": model_config.max_tokens,
        "timeout_seconds": model_config.timeout_seconds,
        "rules_file": model_config.rules_file,
        "resolved": {
            "base_url": _env_or_config_value(
                model_config.base_url_env, model_config.base_url
            )
            or None,
            "model": _env_or_config_value(model_config.model_env, model_config.model)
            or None,
            "api_key_env_set": bool(os.environ.get(model_config.api_key_env, "").strip()),
        },
    }


def _env_or_config_value(env_name: str, config_value: Optional[str]) -> str:
    return os.environ.get(env_name, "").strip() or (config_value or "")


def _required_mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise ConfigError(f"{key} must be a YAML mapping.")
    return value


def _as_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field} must be a non-empty string.")
    return value.strip()


def _optional_str(value: Any, field: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError(f"{field} must be a string or null.")
    return value.strip() or None


def _as_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{field} must be an integer.")
    return value


def _as_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{field} must be a number.")
    return float(value)


def _as_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{field} must be true or false.")
    return value
