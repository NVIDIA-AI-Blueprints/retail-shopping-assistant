# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the directly constructible, deliberately unregistered wrapper."""

from __future__ import annotations

import json
from pathlib import Path

from chain_server.src.tool_policy import SHOPPING_TOOL_POLICIES
from chain_server.src.weather import WeatherRequest, weather_failure
from chain_server.src.weather_tool import get_weather_forecast_tool


REPO_ROOT = Path(__file__).resolve().parents[3]


class RecordingClient:
    def __init__(self) -> None:
        self.requests: list[WeatherRequest] = []

    def get_forecast(self, request: WeatherRequest):
        self.requests.append(request)
        return weather_failure("weather_disabled")


def test_tool_has_the_closed_name_schema_and_direct_result() -> None:
    client = RecordingClient()
    tool = get_weather_forecast_tool(client)

    result = tool.invoke(
        {
            "location": "98101",
            "start_date": "2026-07-28",
            "end_date": "2026-07-29",
        }
    )

    assert tool.name == "get_weather_forecast_tool"
    assert tool.return_direct is False
    assert set(tool.args) == {"location", "date", "start_date", "end_date"}
    assert result["code"] == "weather_disabled"
    assert client.requests == [
        WeatherRequest(
            location="98101",
            start_date="2026-07-28",
            end_date="2026-07-29",
        )
    ]


def test_tool_validation_returns_only_the_sanitized_typed_failure() -> None:
    client = RecordingClient()
    tool = get_weather_forecast_tool(client)

    result = tool.invoke(
        {
            "location": "Seattle 98101",
            "date": "next week",
            "extra": "must not enter the contract",
        }
    )

    parsed = json.loads(result)
    assert parsed == {
        "ok": False,
        "code": "weather_request_invalid",
        "message": "The weather request is invalid.",
        "retryable": False,
    }
    assert client.requests == []
    assert "Seattle" not in result
    assert "next week" not in result


def test_weather_tool_remains_absent_from_every_serving_registration_surface() -> None:
    assert "get_weather_forecast_tool" not in SHOPPING_TOOL_POLICIES

    runtime = (REPO_ROOT / "chain_server/src/deepagents_runtime.py").read_text()
    policy = (REPO_ROOT / "chain_server/src/tool_policy.py").read_text()
    skill_files = list(
        (REPO_ROOT / "chain_server/skills/shopper").glob("*/SKILL.md")
    )

    assert "get_weather_forecast_tool" not in runtime
    assert "get_weather_forecast_tool" not in policy
    assert skill_files
    assert all(
        "get_weather_forecast_tool" not in path.read_text()
        for path in skill_files
    )
