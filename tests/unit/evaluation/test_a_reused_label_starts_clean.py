# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A second run under the same label does not resume the first one's shopper.

The memory service keeps carts and dialogue across runs. While the ids were
derived from the label alone, rerunning `2026-09-30-journeys` after two
rate-limited attempts sent all 25 journeys to the shoppers those attempts had
left behind: J01 opened on turn 1 with two pairs of Jade Suede Heels already in
the cart and nineteen turns of an earlier wedding in memory, and the run
reported 10 of 25 as if the agent had regressed.
"""

from __future__ import annotations

import re

from tests.evaluation.src.replay import scenario_identity


def test_a_reused_label_gets_a_new_shopper_and_conversation() -> None:
    first = scenario_identity("nightly", "0a1b2c3d", "J01_wedding_abroad", 0)
    second = scenario_identity("nightly", "4e5f6a7b", "J01_wedding_abroad", 0)

    assert first["user_id"] != second["user_id"]
    assert first["conversation_id"] != second["conversation_id"]
    assert first["cart_id"] != second["cart_id"]


def test_ids_are_distinct_within_a_run() -> None:
    ids = {
        scenario_identity("nightly", "0a1b2c3d", scenario, repeat)["cart_id"]
        for scenario in ("J01_wedding_abroad", "J02_video_look_full")
        for repeat in range(3)
    }

    assert len(ids) == 6


def test_trace_capture_still_finds_the_label_and_the_journey() -> None:
    """notebook/trace_capture.py selects by `<label>-` and reads `-(J\\d+)_`."""

    identity = scenario_identity("trace-bbd2499d", "0a1b2c3d", "J01_wedding_abroad", 0)

    assert identity["session_id"] == identity["conversation_id"]
    assert identity["conversation_id"].startswith("trace-bbd2499d-")
    assert re.search(r"-(J\d+)_", identity["conversation_id"])[1] == "J01"  # type: ignore[index]
