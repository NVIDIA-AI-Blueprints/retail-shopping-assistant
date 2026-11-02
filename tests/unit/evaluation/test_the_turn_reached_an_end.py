# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A scenario cannot pass while one of its turns never answered.

Both of these runs reported J01 as passing, with 24 green checks:

    2026-09-17__shown-in-order        turn 17, recursion_limit, 21 repeats
    2026-09-17__judge-stop-honoured   turn 17, agent_timeout, 0 products

The scripted checks read the final reply, and "This request took too long to
complete. Please retry." satisfied them. So three separate retry loops lived
inside a green suite, and each was found by reading transcripts rather than by
the suite failing.

This check is unscripted on purpose: the scenario author cannot be expected to
write an expectation for a failure mode they have not met yet.
"""

from __future__ import annotations

from tests.evaluation.src.replay import TurnResult, _the_turn_reached_an_end


def _turn(**overrides: object) -> TurnResult:
    fields: dict[str, object] = {
        "index": 17,
        "said": "it's going to snow when we get back, what should I wear",
        "reply": "This request took too long to complete. Please retry.",
        "products": [],
        "cart": [],
        "tools": ["search_catalog_tool"],
        "seconds": 135.1,
    }
    return TurnResult(**(fields | overrides))  # type: ignore[arg-type]


def test_a_turn_that_ran_out_of_steps_fails() -> None:
    """The 2026-09-17 shown-in-order run, which reported 24 passing checks."""

    checks = _the_turn_reached_an_end(
        _turn(ended="recursion_limit", repeated={"search_catalog_tool": 21})
    )

    assert [(c.name, c.outcome) for c in checks] == [("turn_completed", "fail")]
    assert "recursion_limit" in checks[0].detail
    assert "21 identical repeated calls" in checks[0].detail


def test_a_turn_that_ran_out_of_time_fails() -> None:
    """The judge-stop-honoured run, where the same turn showed no products."""

    checks = _the_turn_reached_an_end(
        _turn(ended="agent_timeout", repeated={"search_catalog_tool": 9})
    )

    assert checks[0].outcome == "fail"
    assert "agent_timeout" in checks[0].detail


def test_the_repeat_count_is_left_out_when_there_were_none() -> None:
    """A turn can exhaust the clock without looping, and should say only that."""

    checks = _the_turn_reached_an_end(_turn(ended="agent_timeout"))

    assert checks[0].detail == "ended on agent_timeout"


def test_a_completed_turn_adds_no_check() -> None:
    """Silence on the ordinary path, so the suite's count does not move."""

    assert _the_turn_reached_an_end(_turn(ended="completed")) == []


def test_a_turn_with_no_recorded_ending_adds_no_check() -> None:
    """Absent diagnostics are unknown, not failure."""

    assert _the_turn_reached_an_end(_turn()) == []


def test_a_grounding_error_is_not_counted_as_running_out() -> None:
    """Seen on turn 15. A real error, but the turn answered with 22 products.

    Kept out because this check is about a turn that never reached an answer.
    Folding other failures in here would make one check mean several things.
    """

    assert _the_turn_reached_an_end(_turn(ended="grounding_error")) == []
