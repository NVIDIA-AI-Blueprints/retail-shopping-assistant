"""Model endpoint routing and deployment metadata.

Each model role is resolved independently. A role can point at an external
endpoint, a local NIM service that this repo can start, or be explicitly
disabled. Secrets are referenced by environment-variable name and are never
returned by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Mapping

import yaml


DEFAULT_CONFIG_ROOT = Path("/app/shared/configs")
MODEL_CONFIG_FILE_NAME = "models.yaml"
SOURCES = {"endpoint", "local_nim", "disabled"}


class ModelConfigError(ValueError):
    """Raised when model configuration is missing or invalid."""


@dataclass(frozen=True)
class ModelEndpoint:
    role: str
    source: str
    provider: str
    base_url: str | None
    model: str | None
    api_key_env: str | None
    api_key_present: bool
    local_service: str | None = None
    compose_file: str | None = None
    compose_service: str | None = None

    @property
    def disabled(self) -> bool:
        return self.source == "disabled"

    @property
    def api_key_required(self) -> bool:
        return self.api_key_env is not None


@dataclass(frozen=True)
class ResolvedModelConfig:
    models: dict[str, ModelEndpoint]
    required_local_nim_services: tuple[str, ...]
    required_local_nim_env: tuple[str, ...]

    def require(self, role: str) -> ModelEndpoint:
        try:
            endpoint = self.models[role]
        except KeyError as exc:
            raise ModelConfigError(f"models.yaml does not define role '{role}'.") from exc
        if endpoint.disabled:
            raise ModelConfigError(f"Model role '{role}' is disabled.")
        return endpoint

    def get(self, role: str) -> ModelEndpoint | None:
        return self.models.get(role)


def config_root_from_env() -> Path:
    return Path(os.environ.get("SHARED_CONFIG_ROOT", str(DEFAULT_CONFIG_ROOT)))


def resolve_model_config(
    *,
    config_root: str | Path | None = None,
) -> ResolvedModelConfig:
    root = Path(config_root) if config_root is not None else config_root_from_env()
    path = root / MODEL_CONFIG_FILE_NAME
    data = _load_yaml_mapping(path)

    version = data.get("version")
    if version != 1:
        raise ModelConfigError(f"Unsupported model config version in {path}: {version}")

    local_nims = _as_mapping(data.get("local_nims", {}), "local_nims")
    local_services = _as_mapping(local_nims.get("services", {}), "local_nims.services")
    required_local_nim_env = tuple(
        _as_str(value, "local_nims.required_env")
        for value in _as_list(local_nims.get("required_env", []), "local_nims.required_env")
    )

    raw_models = _as_mapping(data.get("models"), "models")
    models: dict[str, ModelEndpoint] = {}
    required_services: list[str] = []
    for role, raw_model in raw_models.items():
        if not isinstance(role, str):
            raise ModelConfigError("Model role names must be strings.")
        model_data = _as_mapping(raw_model, f"models.{role}")
        endpoint = _resolve_model(role, model_data, local_services)
        models[role] = endpoint
        if endpoint.compose_service:
            required_services.append(endpoint.compose_service)

    return ResolvedModelConfig(
        models=models,
        required_local_nim_services=tuple(dict.fromkeys(required_services)),
        required_local_nim_env=required_local_nim_env,
    )


def model_config_snapshot(config: ResolvedModelConfig) -> dict[str, Any]:
    """Return a non-secret dictionary suitable for CLI output."""

    return {
        "models": {
            role: {
                "source": endpoint.source,
                "provider": endpoint.provider,
                "base_url": endpoint.base_url,
                "model": endpoint.model,
                "api_key_env": endpoint.api_key_env,
                "api_key_required": endpoint.api_key_required,
                "api_key_present": endpoint.api_key_present,
                "local_service": endpoint.local_service,
                "compose_service": endpoint.compose_service,
            }
            for role, endpoint in config.models.items()
        },
        "required_local_nim_services": list(config.required_local_nim_services),
        "required_local_nim_env": list(config.required_local_nim_env),
    }


def validate_model_config(
    config: ResolvedModelConfig,
    roles: list[str] | tuple[str, ...] | None = None,
) -> None:
    selected_roles = roles or tuple(
        role for role, endpoint in config.models.items() if not endpoint.disabled
    )
    missing_keys = []
    disabled_roles = []
    for role in selected_roles:
        endpoint = config.get(role)
        if endpoint is None:
            raise ModelConfigError(f"models.yaml does not define role '{role}'.")
        if endpoint.disabled:
            disabled_roles.append(role)
            continue
        if endpoint.api_key_env and not endpoint.api_key_present:
            missing_keys.append(f"{role}:{endpoint.api_key_env}")

    if disabled_roles:
        raise ModelConfigError("Disabled required model roles: " + ", ".join(disabled_roles))
    if missing_keys:
        raise ModelConfigError(
            "Missing required API key environment variables: " + ", ".join(missing_keys)
        )


def validate_local_nim_env(config: ResolvedModelConfig) -> None:
    if not config.required_local_nim_services:
        return

    missing = [
        env_name
        for env_name in config.required_local_nim_env
        if not os.environ.get(env_name, "").strip()
    ]
    if missing:
        raise ModelConfigError(
            "Missing required local NIM environment variables: " + ", ".join(missing)
        )


def _resolve_model(
    role: str,
    data: Mapping[str, Any],
    local_services: Mapping[str, Any],
) -> ModelEndpoint:
    source = _as_str(data.get("source"), f"models.{role}.source")
    if source not in SOURCES:
        raise ModelConfigError(
            f"models.{role}.source must be one of: {', '.join(sorted(SOURCES))}."
        )

    provider = _as_str(
        data.get("provider", "openai_compatible"), f"models.{role}.provider"
    )
    api_key_env = _optional_str(data.get("api_key_env"), f"models.{role}.api_key_env")

    if source == "disabled":
        return ModelEndpoint(
            role=role,
            source=source,
            provider=provider,
            base_url=None,
            model=None,
            api_key_env=None,
            api_key_present=False,
        )

    service_data: Mapping[str, Any] = {}
    local_service = None
    compose_file = None
    compose_service = None
    if source == "local_nim":
        local_service = _as_str(data.get("local_service"), f"models.{role}.local_service")
        service_data = _as_mapping(
            local_services.get(local_service), f"local_nims.services.{local_service}"
        )
        compose_file = _as_str(
            service_data.get("compose_file"), f"local_nims.services.{local_service}.compose_file"
        )
        compose_service = _as_str(
            service_data.get("compose_service"),
            f"local_nims.services.{local_service}.compose_service",
        )

    base_url = _resolve_value(
        data,
        service_data,
        value_key="base_url",
        env_key="base_url_env",
        field=f"models.{role}.base_url",
    )
    model = _resolve_value(
        data,
        service_data,
        value_key="model",
        env_key="model_env",
        field=f"models.{role}.model",
    )
    api_key_present = bool(api_key_env and os.environ.get(api_key_env, "").strip())

    return ModelEndpoint(
        role=role,
        source=source,
        provider=provider,
        base_url=base_url,
        model=model,
        api_key_env=api_key_env,
        api_key_present=api_key_present,
        local_service=local_service,
        compose_file=compose_file,
        compose_service=compose_service,
    )


def _resolve_value(
    data: Mapping[str, Any],
    service_data: Mapping[str, Any],
    *,
    value_key: str,
    env_key: str,
    field: str,
) -> str:
    env_name = data.get(env_key)
    if env_name is not None:
        env_value = os.environ.get(_as_str(env_name, f"{field}_env"), "").strip()
        if env_value:
            return env_value

    if data.get(value_key) is not None:
        return _as_str(data.get(value_key), field)

    if service_data.get(value_key) is not None:
        return _as_str(service_data.get(value_key), field)

    raise ModelConfigError(f"Missing {field}: set {value_key} or {env_key}.")


def _load_yaml_mapping(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise ModelConfigError(f"Model config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return _as_mapping(data, str(path))


def _as_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ModelConfigError(f"{field} must be a mapping.")
    return value


def _as_list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ModelConfigError(f"{field} must be a list.")
    return value


def _as_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelConfigError(f"{field} must be a non-empty string.")
    return value.strip()


def _optional_str(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _as_str(value, field)
