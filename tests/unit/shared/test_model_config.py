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


_SHIPPED = Path(__file__).resolve().parents[3] / "shared" / "configs"


@pytest.fixture
def no_model_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for key in ("LLM_BASE_URL", "LLM_MODEL", "VLM_BASE_URL", "VLM_MODEL"):
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def test_shipped_media_perception_is_the_app_llm_model(no_model_env: pytest.MonkeyPatch) -> None:
    config = resolve_model_config(config_root=_SHIPPED)

    app_llm, vlm = config.require("app_llm"), config.require("vlm")
    assert (vlm.base_url, vlm.model) == (app_llm.base_url, app_llm.model)
    assert vlm.api_key_env == "VLM_API_KEY"
    assert config.required_local_nim_services == ()


def test_media_perception_stays_put_when_the_app_llm_moves(
    no_model_env: pytest.MonkeyPatch,
) -> None:
    default = resolve_model_config(config_root=_SHIPPED).require("vlm")
    no_model_env.setenv("LLM_BASE_URL", "https://text-only.example/v1")
    no_model_env.setenv("LLM_MODEL", "text-only-model")

    vlm = resolve_model_config(config_root=_SHIPPED).require("vlm")

    assert (vlm.base_url, vlm.model) == (default.base_url, default.model)


def test_vlm_env_moves_media_perception(no_model_env: pytest.MonkeyPatch) -> None:
    no_model_env.setenv("VLM_BASE_URL", "https://vision.example/v1")
    no_model_env.setenv("VLM_MODEL", "vision-model")

    vlm = resolve_model_config(config_root=_SHIPPED).require("vlm")

    assert (vlm.base_url, vlm.model, vlm.api_key_env) == (
        "https://vision.example/v1",
        "vision-model",
        "VLM_API_KEY",
    )


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
