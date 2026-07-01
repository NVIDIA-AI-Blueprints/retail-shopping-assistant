from dataclasses import replace
from pathlib import Path
import sys

import pytest


EVAL_ROOT = Path(__file__).resolve().parents[2] / "evaluation"
sys.path.insert(0, str(EVAL_ROOT))

from src.config import ConfigError, load_eval_config, resolve_model_runtime, snapshot_eval_config


def test_load_eval_config_resolves_runtime_without_snapshot_secret(monkeypatch):
    monkeypatch.setenv("CHALLENGER_MODEL_BASE_URL", "http://model.example/v1")
    monkeypatch.setenv("CHALLENGER_MODEL_NAME", "shopper-model")
    monkeypatch.setenv("CHALLENGER_MODEL_API_KEY", "secret-token")

    config = load_eval_config()
    runtime = resolve_model_runtime(config.challenger_model)
    snapshot = snapshot_eval_config(config)

    assert runtime.model == "shopper-model"
    assert runtime.base_url == "http://model.example/v1"
    assert runtime.api_key == "secret-token"
    assert "secret-token" not in repr(snapshot)
    assert snapshot["challenger_model"]["resolved"]["api_key_env_set"] is True
    assert snapshot["challenger_model"]["disable_thinking"] is config.challenger_model.disable_thinking
    assert snapshot["challenger_model"]["json_mode"] is config.challenger_model.json_mode
    assert isinstance(config.target_agent.guardrails, bool)
    assert snapshot["target_agent"]["guardrails"] is config.target_agent.guardrails


def test_resolve_runtime_reports_env_var_names_not_values(monkeypatch):
    monkeypatch.delenv("CHALLENGER_MODEL_BASE_URL", raising=False)
    monkeypatch.setenv("CHALLENGER_MODEL_NAME", "shopper-model")
    monkeypatch.setenv("CHALLENGER_MODEL_API_KEY", "secret-token")

    config = load_eval_config()

    with pytest.raises(ConfigError) as exc_info:
        resolve_model_runtime(config.challenger_model)

    message = str(exc_info.value)
    assert "CHALLENGER_MODEL_BASE_URL" in message
    assert "secret-token" not in message


def test_resolve_runtime_allows_local_model_without_api_key(monkeypatch):
    monkeypatch.setenv("CHALLENGER_MODEL_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("CHALLENGER_MODEL_NAME", "local-shopper-model")
    monkeypatch.delenv("CHALLENGER_MODEL_API_KEY", raising=False)

    config = load_eval_config()
    runtime = resolve_model_runtime(config.challenger_model)
    snapshot = snapshot_eval_config(config)

    assert runtime.base_url == "http://localhost:8000/v1"
    assert runtime.model == "local-shopper-model"
    assert runtime.api_key is None
    assert snapshot["challenger_model"]["resolved"]["api_key_env_set"] is False


def test_resolve_runtime_uses_local_config_defaults_without_env_or_api_key(monkeypatch):
    monkeypatch.delenv("CHALLENGER_MODEL_BASE_URL", raising=False)
    monkeypatch.delenv("CHALLENGER_MODEL_NAME", raising=False)
    monkeypatch.delenv("CHALLENGER_MODEL_API_KEY", raising=False)

    config = load_eval_config()
    local_model = replace(
        config.challenger_model,
        base_url="http://127.0.0.1:8080/v1",
        model="local-served-model",
        api_key_required=False,
    )

    runtime = resolve_model_runtime(local_model)

    assert runtime.base_url == "http://127.0.0.1:8080/v1"
    assert runtime.model == "local-served-model"
    assert runtime.api_key is None


def test_resolve_runtime_env_overrides_config_defaults(monkeypatch):
    monkeypatch.setenv("CHALLENGER_MODEL_BASE_URL", "https://remote.example/v1")
    monkeypatch.setenv("CHALLENGER_MODEL_NAME", "remote-model")
    monkeypatch.delenv("CHALLENGER_MODEL_API_KEY", raising=False)

    config = load_eval_config()
    local_model = replace(
        config.challenger_model,
        base_url="http://127.0.0.1:8080/v1",
        model="local-served-model",
        api_key_required=False,
    )

    runtime = resolve_model_runtime(local_model)

    assert runtime.base_url == "https://remote.example/v1"
    assert runtime.model == "remote-model"


def test_resolve_runtime_can_require_api_key_for_cloud(monkeypatch):
    monkeypatch.setenv("CHALLENGER_MODEL_BASE_URL", "https://remote.example/v1")
    monkeypatch.setenv("CHALLENGER_MODEL_NAME", "remote-model")
    monkeypatch.delenv("CHALLENGER_MODEL_API_KEY", raising=False)

    config = load_eval_config()
    cloud_model = replace(config.challenger_model, api_key_required=True)

    with pytest.raises(ConfigError) as exc_info:
        resolve_model_runtime(cloud_model)

    assert "CHALLENGER_MODEL_API_KEY" in str(exc_info.value)
