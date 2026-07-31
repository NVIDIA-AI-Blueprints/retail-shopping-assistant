# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET = (
    REPO_ROOT
    / "tests"
    / "integration"
    / "conversations"
    / "event_context_response_routing"
    / "conv_weather_without_venue.yaml"
)


def test_weather_without_venue_dataset_freezes_response_boundary() -> None:
    conversation = yaml.safe_load(DATASET.read_text(encoding="utf-8"))

    assert len(conversation["queries"]) == len(conversation["answers"]) == 1
    expectation = conversation["diagnostic_expectations"][0]
    assert expectation["required_event_context_next_question"] == "none"
    assert expectation["weather_tool_calls"] == 1
    assert expectation["required_weather_scope"] == {
        "scope_revision": 0,
        "location_action": "set",
        "window_action": "set",
        "location_source": "shopper_provided_location",
        "location_supplied": True,
        "date_supplied": True,
    }
    assert "Based on your venue detail" in expectation[
        "forbidden_response_phrases"
    ]
