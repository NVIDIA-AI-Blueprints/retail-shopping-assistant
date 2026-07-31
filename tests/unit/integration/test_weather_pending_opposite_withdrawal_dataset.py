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
    / "weather_pending_opposite_withdrawal"
    / "conv_opposite_withdrawal.yaml"
)


def test_opposite_withdrawal_dataset_freezes_pending_retirement() -> None:
    conversation = yaml.safe_load(DATASET.read_text(encoding="utf-8"))

    assert conversation["shopper_profile_id"] == "shopper_jordan"
    assert (
        len(conversation["queries"])
        == len(conversation["answers"])
        == len(conversation["diagnostic_expectations"])
        == 4
    )
    initial, withdrawal, reopened, completed = conversation[
        "diagnostic_expectations"
    ]
    assert initial["required_event_context_next_question"] == "event_location"
    assert withdrawal["required_weather_scope"] == {
        "scope_revision": 1,
        "location_action": "clear",
        "window_action": "unavailable",
        "location_source": None,
        "location_supplied": False,
        "date_supplied": False,
    }
    assert withdrawal["required_event_context_next_question"] == "none"
    assert withdrawal["weather_tool_calls"] == 0
    assert reopened["required_weather_scope"]["location_action"] == "clear"
    assert reopened["required_weather_scope"]["window_action"] == "set"
    assert reopened["required_event_context_next_question"] == "event_location"
    assert reopened["weather_tool_calls"] == 0
    assert completed["required_weather_scope"]["location_action"] == "set"
    assert completed["required_weather_scope"]["window_action"] == "retain"
    assert completed["weather_tool_calls"] == 1
