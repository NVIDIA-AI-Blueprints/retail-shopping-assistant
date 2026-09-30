# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Security-sensitive defaults in the standard Compose deployment."""

import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_memory_service_host_port_is_loopback_only() -> None:
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yaml").read_text())

    assert compose["services"]["memory-retriever"]["ports"] == [
        "127.0.0.1:8011:8011"
    ]


def _chain_server_config() -> dict:
    return yaml.safe_load(
        (REPO_ROOT / "shared/configs/chain_server/config.yaml").read_text()
    )


def test_agent_diagnostics_are_disabled_by_default() -> None:
    # Compose passes the variable through empty, so config.yaml decides.
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yaml").read_text())

    assert (
        "EXPOSE_AGENT_DIAGNOSTICS=${EXPOSE_AGENT_DIAGNOSTICS:-}"
        in compose["services"]["chain-server"]["environment"]
    )
    assert _chain_server_config()["expose_agent_diagnostics"] is False


def test_guardrail_settings_have_one_source() -> None:
    # Compose passes these through empty, so guardrails/src/rails.py decides.
    # A second copy here could drift below the coverage the service reports,
    # and modality coverage is the setting that decides whether an upload is
    # vetted at all.
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yaml").read_text())
    environment = compose["services"]["rails"]["environment"]

    for name in (
        "GUARDRAILS_SUPPORTED_MODALITIES",
        "MULTIMODAL_SAFETY_VIDEO_FPS",
        "GUARDRAILS_INPUT_EXECUTION_MODE",
        "GUARDRAILS_TIMEOUT_SECONDS",
    ):
        assert f"{name}=${{{name}:-}}" in environment


def test_weather_secret_is_disabled_and_scoped_to_chain_server() -> None:
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yaml").read_text())
    services = compose["services"]

    assert (
        "WEATHER_ENABLED=${WEATHER_ENABLED:-}"
        in services["chain-server"]["environment"]
    )
    assert _chain_server_config()["weather"]["enabled"] is False
    assert (
        "WEATHER_API_KEY=${WEATHER_API_KEY:-}"
        in services["chain-server"]["environment"]
    )

    for service_name, service in services.items():
        if service_name == "chain-server":
            continue
        environment = service.get("environment", [])
        assert not any(
            entry.startswith(("WEATHER_ENABLED=", "WEATHER_API_KEY="))
            for entry in environment
        )


def test_weather_environment_template_contains_no_secret() -> None:
    env_template = (REPO_ROOT / ".env.example").read_text()

    assert 'export WEATHER_ENABLED="${WEATHER_ENABLED:-}"' in env_template
    assert 'export WEATHER_API_KEY="${WEATHER_API_KEY:-}"' in env_template


def test_private_environment_profiles_are_gitignored() -> None:
    for profile in (".env", ".env.local", ".env.hosted", ".env.local-nim"):
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", "--no-index", profile],
            cwd=REPO_ROOT,
            check=False,
        )
        assert result.returncode == 0

    example = subprocess.run(
        ["git", "check-ignore", "--quiet", "--no-index", ".env.example"],
        cwd=REPO_ROOT,
        check=False,
    )
    assert example.returncode == 1
