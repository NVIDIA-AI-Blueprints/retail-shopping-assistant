# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = (
    REPO_ROOT
    / "tests"
    / "integration"
    / "conversations"
    / "event_context_weather_guidance"
)


def test_weather_guidance_dataset_is_the_thin_six_turn_gate() -> None:
    files = sorted(DATASET_ROOT.glob("conv_*.yaml"))
    assert [path.name for path in files] == [
        "conv_profile_cancun_venue.yaml",
        "conv_profile_nyc_relative_weekday.yaml",
    ]

    conversations = [
        yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in files
    ]
    assert all(
        conversation["shopper_profile_id"] == "shopper_jordan"
        for conversation in conversations
    )
    assert all(
        len(conversation["queries"])
        == len(conversation["answers"])
        == len(conversation["diagnostic_expectations"])
        == 3
        for conversation in conversations
    )

    golden_text = " ".join(
        answer
        for conversation in conversations
        for answer in conversation["answers"]
    ).lower()
    assert "do not infer a beach" in golden_text
    assert "ask only for the event date" in golden_text
    assert "friday next week" in golden_text
    assert "exact prior candidates" in golden_text
    assert "product-agnostic" in golden_text
    assert "performance-proven" in golden_text

    for conversation in conversations:
        first, second, third = conversation["diagnostic_expectations"]
        assert first["tool_call_counts"] == {
            "search_catalog_tool": 1,
            "get_weather_forecast_tool": 0,
        }
        assert second["tool_call_counts"] == {
            "search_catalog_tool": 0,
            "get_weather_forecast_tool": 0,
        }
        assert third["tool_call_counts"] == {
            "search_catalog_tool": 0,
            "get_weather_forecast_tool": 1,
        }
        assert "get_product_details_tool" in second["forbidden_tools"]
        assert "resolve_conversation_products_tool" in third[
            "forbidden_tools"
        ]
