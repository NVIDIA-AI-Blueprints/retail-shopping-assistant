"""Model endpoint routing and deployment metadata.

Each model role is resolved independently. A role can point at an external
endpoint, a locally deployed model that this repo can start, or be explicitly
disabled. An endpoint role's URL and model name come only from the environment
variables it names; the shipped values are in .env.example. Secrets are
referenced by environment-variable name and are never returned by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Mapping

import yaml


DEFAULT_CONFIG_ROOT = Path("/app/shared/configs")
MODEL_CONFIG_FILE_NAME = "models.yaml"
SOURCES = {"endpoint", "local_model", "disabled"}


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
    base_url_env: str | None = None
    model_env: str | None = None

    @property
    def disabled(self) -> bool:
        return self.source == "disabled"

    @property
    def api_key_required(self) -> bool:
        return self.api_key_env is not None

    @property
    def missing_env(self) -> tuple[str, ...]:
        """Variables that must be set before this role has an endpoint."""

        missing = []
        if self.base_url is None and self.base_url_env:
            missing.append(self.base_url_env)
        if self.model is None and self.model_env:
            missing.append(self.model_env)
        return tuple(missing)


@dataclass(frozen=True)
class ResolvedModelConfig:
    models: dict[str, ModelEndpoint]
    required_local_model_services: tuple[str, ...]
    required_local_model_env: tuple[str, ...]

    def require(self, role: str) -> ModelEndpoint:
        try:
            endpoint = self.models[role]
        except KeyError as exc:
            raise ModelConfigError(f"models.yaml does not define role '{role}'.") from exc
        if endpoint.disabled:
            raise ModelConfigError(f"Model role '{role}' is disabled.")
        if endpoint.missing_env:
            raise ModelConfigError(_missing_endpoint_message([role], self.models))
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

    local_models = _as_mapping(data.get("local_models", {}), "local_models")
    local_services = _as_mapping(local_models.get("services", {}), "local_models.services")
    required_local_model_env = tuple(
        _as_str(value, "local_models.required_env")
        for value in _as_list(local_models.get("required_env", []), "local_models.required_env")
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
        required_local_model_services=tuple(dict.fromkeys(required_services)),
        required_local_model_env=required_local_model_env,
    )


def model_config_snapshot(config: ResolvedModelConfig) -> dict[str, Any]:
    """Return a non-secret dictionary suitable for CLI output."""

    return {
        "models": {
            role: {
                "source": endpoint.source,
                "provider": endpoint.provider,
                "base_url": endpoint.base_url,
                "base_url_env": endpoint.base_url_env,
                "model": endpoint.model,
                "model_env": endpoint.model_env,
                "api_key_env": endpoint.api_key_env,
                "api_key_required": endpoint.api_key_required,
                "api_key_present": endpoint.api_key_present,
                "local_service": endpoint.local_service,
                "compose_service": endpoint.compose_service,
            }
            for role, endpoint in config.models.items()
        },
        "required_local_model_services": list(config.required_local_model_services),
        "required_local_model_env": list(config.required_local_model_env),
    }


def validate_model_config(
    config: ResolvedModelConfig,
    roles: list[str] | tuple[str, ...] | None = None,
) -> None:
    selected_roles = roles or tuple(
        role for role, endpoint in config.models.items() if not endpoint.disabled
    )
    missing_keys = []
    missing_endpoints = []
    disabled_roles = []
    for role in selected_roles:
        endpoint = config.get(role)
        if endpoint is None:
            raise ModelConfigError(f"models.yaml does not define role '{role}'.")
        if endpoint.disabled:
            disabled_roles.append(role)
            continue
        if endpoint.missing_env:
            missing_endpoints.append(role)
        if endpoint.api_key_env and not endpoint.api_key_present:
            missing_keys.append(f"{role}:{endpoint.api_key_env}")

    if disabled_roles:
        raise ModelConfigError("Disabled required model roles: " + ", ".join(disabled_roles))
    if missing_endpoints:
        raise ModelConfigError(_missing_endpoint_message(missing_endpoints, config.models))
    if missing_keys:
        raise ModelConfigError(
            "Missing required API key environment variables: " + ", ".join(missing_keys)
        )


def validate_local_model_env(config: ResolvedModelConfig) -> None:
    if not config.required_local_model_services:
        return

    missing = [
        env_name
        for env_name in config.required_local_model_env
        if not os.environ.get(env_name, "").strip()
    ]
    if missing:
        raise ModelConfigError(
            "Missing required local model environment variables: " + ", ".join(missing)
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
    if source == "local_model":
        local_service = _as_str(data.get("local_service"), f"models.{role}.local_service")
        service_data = _as_mapping(
            local_services.get(local_service), f"local_models.services.{local_service}"
        )
        compose_file = _as_str(
            service_data.get("compose_file"), f"local_models.services.{local_service}.compose_file"
        )
        compose_service = _as_str(
            service_data.get("compose_service"),
            f"local_models.services.{local_service}.compose_service",
        )

    for value_key, env_key in (("base_url", "base_url_env"), ("model", "model_env")):
        if value_key in data:
            raise ModelConfigError(
                f"models.{role}.{value_key} is not read: set the variable named by "
                f"{env_key} in your env profile instead (.env.example lists the defaults)."
            )

    base_url_env = model_env = None
    if source == "local_model":
        # The env profile always sets the hosted URLs, so they must not
        # redirect a role that was declared local.
        base_url = _as_str(
            service_data.get("base_url"), f"local_models.services.{local_service}.base_url"
        )
        model = _as_str(
            service_data.get("model"), f"local_models.services.{local_service}.model"
        )
    else:
        base_url_env = _as_str(data.get("base_url_env"), f"models.{role}.base_url_env")
        model_env = _as_str(data.get("model_env"), f"models.{role}.model_env")
        base_url = _env_value(base_url_env)
        model = _env_value(model_env)
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
        base_url_env=base_url_env,
        model_env=model_env,
    )


def _env_value(env_name: str | None) -> str | None:
    value = os.environ.get(env_name, "").strip() if env_name else ""
    return value or None


def _missing_endpoint_message(
    roles: list[str] | tuple[str, ...], models: Mapping[str, ModelEndpoint]
) -> str:
    details = ", ".join(f"{role}:{'+'.join(models[role].missing_env)}" for role in roles)
    return (
        "Missing model endpoint environment variables: "
        + details
        + ". Source .env.example, or a profile copied from it, before starting."
    )


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
