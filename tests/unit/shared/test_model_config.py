from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from shared.model_config import (
    ModelConfigError,
    guardrails_available,
    model_config_snapshot,
    resolve_model_config,
    validate_local_model_env,
    validate_model_config,
)


def _write_models(root: Path, app_llm: dict | None = None) -> Path:
    config_root = root / "configs"
    config_root.mkdir()
    (config_root / "models.yaml").write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "local_models": {
                    "required_env": ["HF_TOKEN", "HF_CACHE"],
                    "services": {
                        "local-embedding": {
                            "compose_file": "docker-compose-model-local.yaml",
                            "compose_service": "local-embedding",
                            "base_url": "http://local-embedding:8000/v1",
                            "model": "nvidia/Nemotron-3-Embed-1B-BF16",
                        }
                    },
                },
                "models": {
                    "app_llm": app_llm
                    or {
                        "source": "endpoint",
                        "base_url_env": "LLM_BASE_URL",
                        "model_env": "LLM_MODEL",
                        "api_key_env": "LLM_API_KEY",
                    },
                    "text_embedding": {
                        "source": "local_model",
                        "local_service": "local-embedding",
                        "base_url_env": "TEXT_EMBED_BASE_URL",
                        "model_env": "TEXT_EMBED_MODEL",
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


@pytest.fixture
def llm_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example/v1")
    monkeypatch.setenv("LLM_MODEL", "llm")
    return monkeypatch


def test_resolves_endpoint_local_and_disabled_roles(
    llm_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_root = _write_models(tmp_path)
    llm_env.setenv("LLM_API_KEY", "test-key")

    config = resolve_model_config(config_root=config_root)

    assert config.require("app_llm").base_url == "https://llm.example/v1"
    assert config.require("text_embedding").base_url == "http://local-embedding:8000/v1"
    assert config.require("text_embedding").api_key_env is None
    assert config.get("topic_control").disabled is True
    assert config.required_local_model_services == ("local-embedding",)
    assert config.required_local_model_env == ("HF_TOKEN", "HF_CACHE")


def test_a_local_role_is_not_redirected_by_the_hosted_env(
    model_endpoint_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = resolve_model_config(config_root=_write_models(tmp_path))

    embedding = config.require("text_embedding")
    assert (embedding.base_url, embedding.model) == (
        "http://local-embedding:8000/v1",
        "nvidia/Nemotron-3-Embed-1B-BF16",
    )


def test_endpoint_role_without_its_env_is_reported_by_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.setenv("LLM_MODEL", "llm")
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    config = resolve_model_config(config_root=_write_models(tmp_path))

    with pytest.raises(ModelConfigError, match=r"app_llm:LLM_BASE_URL\b"):
        validate_model_config(config, roles=("app_llm",))
    with pytest.raises(ModelConfigError, match=r"app_llm:LLM_BASE_URL\b"):
        config.require("app_llm")


@pytest.mark.parametrize("field", ["base_url", "model"])
def test_a_url_or_model_written_in_models_yaml_is_rejected(
    llm_env: pytest.MonkeyPatch, tmp_path: Path, field: str
) -> None:
    app_llm = {
        "source": "endpoint",
        "base_url_env": "LLM_BASE_URL",
        "model_env": "LLM_MODEL",
        "api_key_env": "LLM_API_KEY",
        field: "written-in-yaml",
    }

    with pytest.raises(ModelConfigError, match=f"models.app_llm.{field} is not read"):
        resolve_model_config(config_root=_write_models(tmp_path, app_llm=app_llm))


_SHIPPED = Path(__file__).resolve().parents[3] / "shared" / "configs"


def test_env_example_gives_every_shipped_role_an_endpoint(
    model_endpoint_env: pytest.MonkeyPatch,
) -> None:
    config = resolve_model_config(config_root=_SHIPPED)

    missing = {role: e.missing_env for role, e in config.models.items() if e.missing_env}
    assert missing == {}


def test_shipped_media_perception_is_the_app_llm_model(model_endpoint_env: pytest.MonkeyPatch) -> None:
    config = resolve_model_config(config_root=_SHIPPED)

    app_llm, vlm = config.require("app_llm"), config.require("vlm")
    assert (vlm.base_url, vlm.model) == (app_llm.base_url, app_llm.model)
    assert vlm.api_key_env == "VLM_API_KEY"
    assert config.required_local_model_services == ()


def test_shipped_default_roles_share_one_host(model_endpoint_env: pytest.MonkeyPatch) -> None:
    config = resolve_model_config(config_root=_SHIPPED)

    hosts = {config.require(role).base_url for role in ("app_llm", "vlm", "text_embedding")}
    assert len(hosts) == 1


def test_media_perception_stays_put_when_the_app_llm_moves(
    model_endpoint_env: pytest.MonkeyPatch,
) -> None:
    default = resolve_model_config(config_root=_SHIPPED).require("vlm")
    model_endpoint_env.setenv("LLM_BASE_URL", "https://text-only.example/v1")
    model_endpoint_env.setenv("LLM_MODEL", "text-only-model")

    vlm = resolve_model_config(config_root=_SHIPPED).require("vlm")

    assert (vlm.base_url, vlm.model) == (default.base_url, default.model)


def test_vlm_env_moves_media_perception(model_endpoint_env: pytest.MonkeyPatch) -> None:
    model_endpoint_env.setenv("VLM_BASE_URL", "https://vision.example/v1")
    model_endpoint_env.setenv("VLM_MODEL", "vision-model")

    vlm = resolve_model_config(config_root=_SHIPPED).require("vlm")

    assert (vlm.base_url, vlm.model, vlm.api_key_env) == (
        "https://vision.example/v1",
        "vision-model",
        "VLM_API_KEY",
    )


def test_validate_model_config_reports_missing_required_key(
    llm_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_root = _write_models(tmp_path)
    llm_env.delenv("LLM_API_KEY", raising=False)

    config = resolve_model_config(config_root=config_root)

    with pytest.raises(ModelConfigError, match="LLM_API_KEY"):
        validate_model_config(config, roles=("app_llm",))


def test_validate_model_config_reports_disabled_required_role(
    llm_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = resolve_model_config(config_root=_write_models(tmp_path))

    with pytest.raises(ModelConfigError, match="topic_control"):
        validate_model_config(config, roles=("topic_control",))


def test_unavailable_guardrails_need_no_guardrail_endpoint_or_key(
    model_endpoint_env: pytest.MonkeyPatch,
) -> None:
    for key in ("LLM_API_KEY", "VLM_API_KEY", "EMBED_API_KEY", "IMAGE_EMBED_API_KEY"):
        model_endpoint_env.setenv(key, "test-key")
    for key in ("RAIL_API_KEY", "MULTIMODAL_SAFETY_API_KEY", "RAILS_CONTENT_BASE_URL"):
        model_endpoint_env.delenv(key, raising=False)
    config = resolve_model_config(config_root=_SHIPPED)

    with pytest.raises(ModelConfigError, match="content_safety"):
        validate_model_config(config)

    model_endpoint_env.setenv("GUARDRAILS_AVAILABLE", "false")
    validate_model_config(config)
    assert model_config_snapshot(config)["guardrails_available"] is False


def test_guardrails_available_rejects_an_unknown_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GUARDRAILS_AVAILABLE", "ture")

    with pytest.raises(ModelConfigError, match="GUARDRAILS_AVAILABLE"):
        guardrails_available()


def test_validate_local_model_env_only_when_local_services_are_used(
    llm_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = resolve_model_config(config_root=_write_models(tmp_path))
    llm_env.delenv("HF_TOKEN", raising=False)
    llm_env.delenv("HF_CACHE", raising=False)

    with pytest.raises(ModelConfigError, match="HF_TOKEN"):
        validate_local_model_env(config)

    llm_env.setenv("HF_TOKEN", "test-token")
    llm_env.setenv("HF_CACHE", "/tmp/hf")
    validate_local_model_env(config)


def test_snapshot_does_not_include_secret_value(
    llm_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_root = _write_models(tmp_path)
    llm_env.setenv("LLM_API_KEY", "secret-value")

    snapshot = model_config_snapshot(resolve_model_config(config_root=config_root))

    assert "secret-value" not in repr(snapshot)
    assert snapshot["models"]["app_llm"]["api_key_present"] is True
