# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = (
    REPO_ROOT / "tests" / "integration" / "conversations" / "event_context"
)


def test_event_context_dataset_is_a_four_turn_profile_and_guest_gate() -> None:
    files = sorted(DATASET_ROOT.glob("conv_*.yaml"))
    assert [path.name for path in files] == [
        "conv_guest_location.yaml",
        "conv_profile_saved_location.yaml",
        "conv_profile_shop_now.yaml",
    ]

    conversations = [
        yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in files
    ]

    assert sum(len(item["queries"]) for item in conversations) == 4
    assert all(
        len(item["queries"]) == len(item["answers"])
        for item in conversations
    )
    assert conversations[0].get("shopper_profile_id") is None
    assert conversations[1]["shopper_profile_id"] == "shopper_alex"
    assert conversations[2]["shopper_profile_id"] == "shopper_alex"

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
    assert "grounded dress candidates" in golden_text
