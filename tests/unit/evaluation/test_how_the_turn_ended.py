# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A failed turn has to say why, and over the shapes the runtime really emits.

The report used to record tool *names* only. A turn that hit the graph's
recursion limit therefore read as an ordinary quality failure: 24 names, a
generic apology, and no sign that the same rejected argument had been sent 22
times. The reason existed -- the runtime computes `final_termination_reason`
-- and was dropped before it reached the artifact.

The first attempt at reading it back took `rejected_tool_calls` for a list of
call objects and asked each one for its `tool_name`. It holds sequence numbers.
Every unit test passed, because none of them fed it a real diagnostic, and
J01 turn 1 died with `AttributeError: 'int' object has no attribute 'get'`.
So these cases are written against the shapes `_collect_agent_diagnostics`
actually produces, which `test_deepagents_observability.py` fixes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tests.evaluation.src.replay import (
    TurnResult,
    _how_the_turn_ended,
    write_transcript,
)

#: The shape asserted by `test_tool_trace_records_native_repair_scope_rejection`:
#: a rejection's name and reason live on the `tool_calls` entry, and
#: `rejected_tool_calls` carries that entry's sequence number.
_A_REJECTED_CALL: dict[str, Any] = {
    "tool_calls": [
        {
            "sequence": 1,
            "tool_name": "search_catalog_tool",
            "arguments": {"requested_product_type": "tote_bags"},
            "status": "rejected",
            "rejection_reason": "repair_scope_changed",
        }
    ],
    "rejected_tool_calls": [1],
    "final_termination_reason": "completed",
}


def test_a_rejection_is_reported_with_its_reason() -> None:
    assert _how_the_turn_ended(_A_REJECTED_CALL)["rejected"] == [
        "search_catalog_tool (repair_scope_changed)"
    ]


def test_sequence_numbers_are_never_read_as_calls() -> None:
    """The bug that killed J01 turn 1: `[1]` is not a list of call objects."""

    ended = _how_the_turn_ended(
        {"rejected_tool_calls": [1, 2, 3], "tool_calls": []}
    )
    assert ended["rejected"] == []
    assert ended["repeated"] == {}


def test_the_termination_reason_is_carried_through() -> None:
    assert (
        _how_the_turn_ended({"final_termination_reason": "recursion_limit"})[
            "ended"
        ]
        == "recursion_limit"
    )


def test_one_argument_sent_twice_is_counted_once_as_a_repeat() -> None:
    """A name count cannot tell 22 identical calls from 22 questions."""

    same = {
        "tool_name": "resolve_conversation_products_tool",
        "arguments": {"references": '[{"reference_id": "first_sweater"'},
    }
    ended = _how_the_turn_ended({"tool_calls": [same, dict(same), dict(same)]})
    assert ended["repeated"] == {"resolve_conversation_products_tool": 2}


def test_differing_arguments_are_not_repeats() -> None:
    ended = _how_the_turn_ended(
        {
            "tool_calls": [
                {"tool_name": "search_catalog_tool", "arguments": {"q": "boots"}},
                {"tool_name": "search_catalog_tool", "arguments": {"q": "belts"}},
            ]
        }
    )
    assert ended["repeated"] == {}


def test_absent_diagnostics_do_not_raise() -> None:
    """A turn that never reached the agent still has to be recorded."""

    assert _how_the_turn_ended({}) == {"ended": "", "rejected": [], "repeated": {}}


def _transcript_of(turn: TurnResult, tmp_path: Path) -> str:
    """Render one turn the way a run does: through `vars(turn)`.

    Going through the dataclass is the point. A field the stream carries and
    `TurnResult` does not declare is dropped between the two, which is how
    these three were computed and never printed.
    """

    path = tmp_path / "transcript.md"
    write_transcript(
        path,
        {"why": ""},
        {
            "id": "J02_video_look_full",
            "covers": ["references"],
            "identity": {"conversation_id": "c1"},
            "turns": [vars(turn) | {"checks": []}],
        },
        "test-build",
    )
    return path.read_text()


def _turn(**overrides: Any) -> TurnResult:
    fields: dict[str, Any] = {
        "index": 2,
        "said": "do you have that first one in a size 6",
        "reply": "I encountered an error while helping with your shopping request.",
        "products": [],
        "cart": [],
        "tools": ["resolve_conversation_products_tool"],
        "seconds": 50.1,
    }
    return TurnResult(**(fields | overrides))


def test_the_transcript_states_an_abnormal_ending(tmp_path: Path) -> None:
    """What the 2026-09-17 run would have said without opening a database."""

    written = _transcript_of(
        _turn(
            ended="recursion_limit",
            repeated={"resolve_conversation_products_tool": 21},
        ),
        tmp_path,
    )

    assert "ended: **recursion_limit**" in written
    assert "identical repeats: resolve_conversation_products_tool x21" in written


def test_the_transcript_stays_quiet_on_an_ordinary_turn(tmp_path: Path) -> None:
    """A line on every turn saying "completed" is a line readers learn to skip."""

    written = _transcript_of(_turn(ended="completed"), tmp_path)

    assert "ended:" not in written
    assert "identical repeats:" not in written
    assert "rejected:" not in written
