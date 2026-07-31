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
    / "weather_pending_withdrawal"
    / "conv_pending_withdrawal.yaml"
)


def test_pending_withdrawal_dataset_freezes_the_authority_regression() -> None:
    conversation = yaml.safe_load(DATASET.read_text(encoding="utf-8"))

    assert conversation["shopper_profile_id"] == "shopper_jordan"
    assert (
        len(conversation["queries"])
        == len(conversation["answers"])
        == len(conversation["diagnostic_expectations"])
        == 3
    )
    first, withdrawal, completion = conversation["diagnostic_expectations"]
    assert first["required_weather_scope"]["window_action"] == "set"
    assert withdrawal["required_weather_scope"] == {
        "scope_revision": 1,
        "location_action": "set",
        "window_action": "unavailable",
        "location_source": "shopper_provided_location",
        "location_supplied": True,
        "date_supplied": False,
    }
    assert withdrawal["required_event_context_next_question"] == "none"
    assert withdrawal["weather_tool_calls"] == 0
    assert conversation["queries"][2] == "Saturday next week."
    assert completion["required_weather_scope"]["location_action"] == "retain"
    assert completion["required_weather_scope"]["window_action"] == "set"
    assert completion["required_weather_trace"]["request_shape"] == (
        "relative_exact_date"
    )
    assert completion["weather_tool_calls"] == 1
