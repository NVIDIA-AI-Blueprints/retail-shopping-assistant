# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Security-sensitive defaults in the standard Compose deployment."""

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
