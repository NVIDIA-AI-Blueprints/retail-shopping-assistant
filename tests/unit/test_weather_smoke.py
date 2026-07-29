# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline checks for the explicit redacted weather smoke command."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/weather_smoke.py"


def _run_smoke(overrides: dict[str, str]) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    for name in (
        "WEATHER_API_KEY",
        "WEATHER_SMOKE_LOCATION",
        "WEATHER_SMOKE_DATE",
        "WEATHER_SMOKE_START_DATE",
        "WEATHER_SMOKE_END_DATE",
    ):
        environment.pop(name, None)
    environment.update(overrides)
    return subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_disabled_smoke_makes_no_request_and_redacts_inputs() -> None:
    result = _run_smoke(
        {
            "WEATHER_ENABLED": "false",
            "WEATHER_SMOKE_LOCATION": "NYC, NY",
            "WEATHER_SMOKE_DATE": "2026-07-28",
        }
    )

    assert result.returncode == 1
    output = json.loads(result.stdout)
    assert output["outcome"] == "weather_disabled"
    assert output["mode"] == "single_date"
    assert output["window_days"] == 1
    assert "NYC, NY" not in result.stdout + result.stderr
    assert "2026-07-28" not in result.stdout + result.stderr


def test_enabled_smoke_without_key_fails_before_transport() -> None:
    result = _run_smoke(
        {
            "WEATHER_ENABLED": "true",
            "WEATHER_SMOKE_LOCATION": "NYC, NY",
        }
    )

    assert result.returncode == 1
    output = json.loads(result.stdout)
    assert output["outcome"] == "weather_config_invalid"
    assert output["schema_valid"] is True
    assert "NYC, NY" not in result.stdout + result.stderr


def test_invalid_config_and_request_use_distinct_failure_codes() -> None:
    invalid_config = _run_smoke(
        {
            "WEATHER_ENABLED": "sometimes",
            "WEATHER_SMOKE_LOCATION": "NYC, NY",
        }
    )
    invalid_request = _run_smoke(
        {
            "WEATHER_ENABLED": "false",
            "WEATHER_SMOKE_LOCATION": "NYC, NY",
            "WEATHER_SMOKE_DATE": "next week",
        }
    )

    assert json.loads(invalid_config.stdout)["outcome"] == "weather_config_invalid"
    assert json.loads(invalid_request.stdout)["outcome"] == "weather_request_invalid"
    combined = (
        invalid_config.stdout
        + invalid_config.stderr
        + invalid_request.stdout
        + invalid_request.stderr
    )
    assert "NYC, NY" not in combined
    assert "next week" not in combined
