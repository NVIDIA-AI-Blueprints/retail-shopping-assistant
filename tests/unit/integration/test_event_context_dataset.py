# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = (
    REPO_ROOT / "tests" / "integration" / "conversations" / "event_context"
)


def test_event_context_dataset_is_a_focused_location_weather_gate() -> None:
    files = sorted(DATASET_ROOT.glob("conv_*.yaml"))
    assert [path.name for path in files] == [
        "conv_guest_location.yaml",
        "conv_profile_saved_location.yaml",
        "conv_profile_shop_now.yaml",
        "conv_profile_weather.yaml",
        "conv_weather_isolation.yaml",
    ]

    conversations = [
        yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in files
    ]

    assert sum(len(item["queries"]) for item in conversations) == 10
    assert all(
        len(item["queries"])
        == len(item["answers"])
        == len(item["diagnostic_expectations"])
        for item in conversations
    )
    assert conversations[0].get("shopper_profile_id") is None
    assert conversations[1]["shopper_profile_id"] == "shopper_alex"
    assert conversations[2]["shopper_profile_id"] == "shopper_jordan"
    assert conversations[3]["shopper_profile_id"] == "shopper_alex"
    assert conversations[4].get("shopper_profile_id") is None

    golden_text = " ".join(
        answer
        for conversation in conversations
        for answer in conversation["answers"]
    ).lower()
    assert "tentative local-event candidate" in golden_text
    assert "shopper's usual area" in golden_text
    assert "ask the destination directly" in golden_text
    assert (
        "explicit cancun ceremony-and-reception beach-on-sand setting overrides"
        in golden_text
    )
    assert "guest mode" in golden_text
    assert "do not claim a live forecast" in golden_text
    assert "event time, product role, or preferences" in golden_text
    assert "wind, breeze" in golden_text
    assert "not as a request for a complete multi-role look" in golden_text
    assert "run one catalog search for one useful core role" in golden_text
    assert "never a bare where, city, or destination question" in golden_text
    assert "fulfillment of the prior event-context question" in golden_text
    assert "run no new catalog search" in golden_text
    assert "next calendar monday-through-sunday range" in golden_text
    assert "provider-resolved forecast location" in golden_text
    assert "supported event zip and venue but no event date" in golden_text
    assert "resolve \"tomorrow\" against the server date" in golden_text
    assert "weather data provided by visual crossing" in golden_text
    assert "forecasts can change" in golden_text
    assert "current explicit shopper-provided zip overrides" in golden_text
    assert "do not reuse the prior forecast" in golden_text
    assert "do not activate event context" in golden_text

    expectations = [
        expected
        for conversation in conversations
        for expected in conversation["diagnostic_expectations"]
    ]
    assert sum(
        expected["weather_tool_calls"]
        for expected in expectations
    ) == 4
    assert all(
        "get_weather_forecast_tool"
        in (
            expected.get("required_tools", [])
            + expected.get("forbidden_tools", [])
        )
        for expected in expectations
    )
    assert conversations[2]["diagnostic_expectations"][0][
        "tool_call_counts"
    ] == {"search_catalog_tool": 1}
    assert conversations[2]["diagnostic_expectations"][1][
        "tool_call_counts"
    ] == {
        "get_weather_forecast_tool": 1,
        "search_catalog_tool": 0,
    }
