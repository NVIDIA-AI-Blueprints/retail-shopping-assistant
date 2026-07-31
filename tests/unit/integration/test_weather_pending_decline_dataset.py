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
    / "weather_pending_decline"
    / "conv_pending_decline.yaml"
)


def test_pending_decline_dataset_freezes_source_bound_consumption() -> None:
    conversation = yaml.safe_load(DATASET.read_text(encoding="utf-8"))

    assert conversation["shopper_profile_id"] == "shopper_jordan"
    assert (
        len(conversation["queries"])
        == len(conversation["answers"])
        == len(conversation["diagnostic_expectations"])
        == 3
    )
    initial, declined, product_work = conversation["diagnostic_expectations"]
    assert initial["required_event_context_next_question"] == "event_location"
    assert declined["required_event_context_next_question"] == "none"
    assert declined["required_weather_scope"] == {
        "scope_revision": 1,
        "location_action": "unavailable",
        "window_action": "retain",
        "location_source": None,
        "location_supplied": False,
        "date_supplied": False,
    }
    assert declined["weather_tool_calls"] == 0
    assert "event-context" in product_work["forbidden_skills"]
    assert product_work["required_business_sequence"] == ["search_catalog_tool"]
