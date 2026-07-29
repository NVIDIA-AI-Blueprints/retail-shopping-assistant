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
    / "event_context_comparison"
)


def test_event_context_comparison_dataset_is_one_three_turn_gate() -> None:
    files = sorted(DATASET_ROOT.glob("conv_*.yaml"))
    assert [path.name for path in files] == [
        "conv_prior_product_weather.yaml",
    ]

    conversation = yaml.safe_load(files[0].read_text(encoding="utf-8"))
    assert conversation["shopper_profile_id"] == "shopper_jordan"
    assert (
        len(conversation["queries"])
        == len(conversation["answers"])
        == len(conversation["diagnostic_expectations"])
        == 3
    )

    comparison = conversation["diagnostic_expectations"][2]
    assert [
        item["required_business_sequence"]
        for item in conversation["diagnostic_expectations"]
    ] == [
        ["search_catalog_tool"],
        ["get_weather_forecast_tool"],
        [
            "resolve_conversation_products_tool",
            "get_product_details_tool",
            "get_product_details_tool",
        ],
    ]
    assert [
        item["required_event_context_next_question"]
        for item in conversation["diagnostic_expectations"]
    ] == ["event_location", "none", "none"]
    assert conversation["diagnostic_expectations"][1][
        "required_weather_trace"
    ] == {
        "request_shape": "relative_range",
        "location_source": "shopper_provided_location",
        "provider_input": "location_query",
        "outcome": "success",
    }
    assert comparison["required_tools"] == [
        "resolve_conversation_products_tool",
        "get_product_details_tool",
    ]
    assert comparison["tool_call_counts"] == {
        "resolve_conversation_products_tool": 1,
        "get_product_details_tool": 2,
        "search_catalog_tool": 0,
        "get_weather_forecast_tool": 0,
    }
    assert comparison["required_product_detail_names"] == [
        "Intricate Lace Gown",
        "Wavy Hem Satin Dress",
    ]
    assert "search_catalog_tool" in comparison["forbidden_tools"]
    assert "get_weather_forecast_tool" in comparison["forbidden_tools"]
    assert comparison["required_response_phrases"] == [
        "Intricate Lace Gown",
        "Wavy Hem Satin Dress",
    ]
    assert "Previously shown options still in play" in (
        comparison["forbidden_response_phrases"]
    )
