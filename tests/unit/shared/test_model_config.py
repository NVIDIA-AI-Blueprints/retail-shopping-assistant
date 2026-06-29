from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from shared.model_config import (
    ModelConfigError,
    model_config_snapshot,
    resolve_model_config,
    validate_local_nim_env,
    validate_model_config,
)


def _write_models(root: Path) -> Path:
    config_root = root / "configs"
    config_root.mkdir()
    (config_root / "models.yaml").write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "local_nims": {
                    "required_env": ["NGC_API_KEY", "LOCAL_NIM_CACHE"],
                    "services": {
                        "nvclip": {
                            "compose_file": "docker-compose-nim-local.yaml",
                            "compose_service": "nvclip",
                            "base_url": "http://nvclip:8000/v1",
                            "model": "nvidia/nvclip",
                        }
                    },
                },
                "models": {
                    "app_llm": {
                        "source": "endpoint",
                        "base_url": "https://llm.example/v1",
                        "model": "llm",
                        "api_key_env": "LLM_API_KEY",
                    },
                    "image_embedding": {
                        "source": "local_nim",
                        "local_service": "nvclip",
                        "api_key_env": None,
                    },
                    "topic_control": {
                        "source": "disabled",
                    },
                },
            }
        )
    )
    return config_root


def test_resolves_endpoint_local_and_disabled_roles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_root = _write_models(tmp_path)
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    config = resolve_model_config(config_root=config_root)

    assert config.require("app_llm").base_url == "https://llm.example/v1"
    assert config.require("image_embedding").base_url == "http://nvclip:8000/v1"
    assert config.require("image_embedding").api_key_env is None
    assert config.get("topic_control").disabled is True
    assert config.required_local_nim_services == ("nvclip",)
    assert config.required_local_nim_env == ("NGC_API_KEY", "LOCAL_NIM_CACHE")


def test_validate_model_config_reports_missing_required_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_root = _write_models(tmp_path)
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    config = resolve_model_config(config_root=config_root)

    with pytest.raises(ModelConfigError, match="LLM_API_KEY"):
        validate_model_config(config, roles=("app_llm",))


def test_validate_model_config_reports_disabled_required_role(tmp_path: Path) -> None:
    config = resolve_model_config(config_root=_write_models(tmp_path))

    with pytest.raises(ModelConfigError, match="topic_control"):
        validate_model_config(config, roles=("topic_control",))


def test_validate_local_nim_env_only_when_local_services_are_used(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = resolve_model_config(config_root=_write_models(tmp_path))
    monkeypatch.delenv("NGC_API_KEY", raising=False)
    monkeypatch.delenv("LOCAL_NIM_CACHE", raising=False)

    with pytest.raises(ModelConfigError, match="NGC_API_KEY"):
        validate_local_nim_env(config)

    monkeypatch.setenv("NGC_API_KEY", "test-key")
    monkeypatch.setenv("LOCAL_NIM_CACHE", "/tmp/nim")
    validate_local_nim_env(config)


def test_snapshot_does_not_include_secret_value(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_root = _write_models(tmp_path)
    monkeypatch.setenv("LLM_API_KEY", "secret-value")

    snapshot = model_config_snapshot(resolve_model_config(config_root=config_root))

    assert "secret-value" not in repr(snapshot)
    assert snapshot["models"]["app_llm"]["api_key_present"] is True
