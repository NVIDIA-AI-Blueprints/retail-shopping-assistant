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


def test_agent_diagnostics_are_disabled_by_default() -> None:
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yaml").read_text())

    assert (
        "EXPOSE_AGENT_DIAGNOSTICS=${EXPOSE_AGENT_DIAGNOSTICS:-false}"
        in compose["services"]["chain-server"]["environment"]
    )


def test_weather_secret_is_disabled_and_scoped_to_chain_server() -> None:
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yaml").read_text())
    services = compose["services"]

    assert (
        "WEATHER_ENABLED=${WEATHER_ENABLED:-false}"
        in services["chain-server"]["environment"]
    )
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

    assert 'export WEATHER_ENABLED="${WEATHER_ENABLED:-false}"' in env_template
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
