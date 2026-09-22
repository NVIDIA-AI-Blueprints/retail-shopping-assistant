# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A tool asked the same thing twice in one turn answers from the first ask.

Nothing used to stop it. Asked which of four dresses was better value for
weekly wear, the model ran one evidence block three times -- availability for
the same four refs with the same invented size hints, promotions with no
arguments at all, then the same product details -- with every previous result
still in its prompt, which grew 13.3k to 17.5k as it went. It emitted no
reasoning between the repeats. The turn ended at 19 tool calls and 95 seconds
because the product-detail cap fired, not because the model was done.
"""

from __future__ import annotations

from chain_server.src.control_signals import SIGNALS_KEY, ControlSignal
from chain_server.src.turn_scope import TurnScope


def test_the_first_ask_is_not_a_repeat() -> None:
    scope = TurnScope()

    assert scope.answer_already_given("check_active_promotions_tool", "") is None


def test_the_same_ask_again_is_answered_from_the_first() -> None:
    scope = TurnScope()
    scope.remember_answer(
        "check_active_promotions_tool",
        "",
        "ACTIVE PROMOTIONS: NO",
    )

    held = scope.answer_already_given("check_active_promotions_tool", "")

    assert isinstance(held, str)
    assert "ALREADY_ANSWERED_THIS_TURN" in held
    # The answer itself comes back, so a model that genuinely lost track of it
    # is not left with a scolding and no data.
    assert "ACTIVE PROMOTIONS: NO" in held


def test_the_third_identical_ask_stops_tool_use() -> None:
    """Saying "you already asked" is itself something to loop against."""

    scope = TurnScope()
    scope.remember_answer("check_product_availability_tool", "[]", "AVAILABILITY")

    scope.answer_already_given("check_product_availability_tool", "[]")
    third = scope.answer_already_given("check_product_availability_tool", "[]")

    assert isinstance(third, tuple)
    text, artifact = third
    assert "STOP_TOOL_USE" in text
    assert artifact[SIGNALS_KEY] == [str(ControlSignal.STOP_TOOL_USE)]


def test_different_arguments_are_a_different_question() -> None:
    scope = TurnScope()
    scope.remember_answer("get_product_details_tool", "ref_a", "DETAILS A")

    assert scope.answer_already_given("get_product_details_tool", "ref_b") is None


def test_one_turn_s_answers_do_not_reach_the_next() -> None:
    """Scope is per turn, so a price read last turn is read again this turn."""

    scope = TurnScope()
    scope.remember_answer("get_product_details_tool", "ref_a", "DETAILS A")

    assert TurnScope().answer_already_given(
        "get_product_details_tool",
        "ref_a",
    ) is None
