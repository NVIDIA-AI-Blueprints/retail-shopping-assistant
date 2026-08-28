# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""J06 t9: "add the Jade Suede Heels in a 6", then "actually make those a 7".

It resolved "those" to a dress from eight turns earlier. Nothing was missing --
the heels were the line directly above the pronoun in the conversation lane,
the newest showing in the index, and a line in the cart. Three copies of the
answer and the model derived the referent from none of them.

So the runtime derives it and states the result. Not a fourth copy of the
conversation: the resolution of it.
"""

from __future__ import annotations

from types import SimpleNamespace

from chain_server.src.turn_support import format_most_recent_subject


def _state(sets):
    return SimpleNamespace(historical_product_sets=sets)


_HEELS = {
    "turn_seq": 7,
    "products": [
        {"ref": "h1", "name": "Jade Suede Heels"},
        {"ref": "h2", "name": "Charming Cognac Heels"},
    ],
}
_DRESSES = {
    "turn_seq": 1,
    "products": [{"ref": "d1", "name": "Black Satin Lace-Up Dress"}],
}


def test_the_newest_showing_is_the_subject() -> None:
    text = format_most_recent_subject(_state([_DRESSES, _HEELS]))

    assert "Jade Suede Heels" in text
    assert "Black Satin Lace-Up Dress" not in text
    assert "turn 7" in text


def test_it_says_what_a_bare_pronoun_means() -> None:
    text = format_most_recent_subject(_state([_HEELS]))

    assert "MOST RECENT SUBJECT" in text
    assert '"those"' in text


def test_order_in_the_list_does_not_decide_it() -> None:
    """Newest by turn, not by position in the list."""

    assert "Jade Suede Heels" in format_most_recent_subject(
        _state([_HEELS, _DRESSES])
    )


def test_nothing_shown_yet_says_nothing() -> None:
    """An opening turn gains no section to ignore."""

    assert format_most_recent_subject(_state([])) == ""
    assert format_most_recent_subject(SimpleNamespace()) == ""


def test_a_malformed_entry_is_skipped_rather_than_crashing() -> None:
    assert format_most_recent_subject(_state(["nonsense", _HEELS]))


def test_the_lane_actually_reaches_the_prompt() -> None:
    """Through _build_user_message, not the formatter.

    Removing the call passed every test above. That wiring gap has now caught
    me six times this month, so the assertion is on the message the model gets.
    """

    from chain_server.src.agenttypes import Cart
    from chain_server.src.deepagents_runtime import DeepAgentsRuntime

    state = SimpleNamespace(
        query="actually make those a 7",
        image=None,
        media=[],
        media_analysis="",
        cart=Cart(),
        context="",
        shopper_context=None,
        wearer_audience=[],
        historical_product_sets=[_HEELS],
    )
    identity = SimpleNamespace(
        request_id="r1", session_id="s1", conversation_id="c1", cart_id="cart1"
    )

    message = DeepAgentsRuntime._build_user_message(None, state, identity)

    assert "MOST RECENT SUBJECT" in message
    assert "Jade Suede Heels" in message


def test_an_opening_turn_has_no_empty_section() -> None:
    """A heading with nothing under it is noise the model learns to skip."""

    from chain_server.src.agenttypes import Cart
    from chain_server.src.deepagents_runtime import DeepAgentsRuntime

    state = SimpleNamespace(
        query="show me black dresses",
        image=None,
        media=[],
        media_analysis="",
        cart=Cart(),
        context="",
        shopper_context=None,
        wearer_audience=[],
        historical_product_sets=[],
    )
    identity = SimpleNamespace(
        request_id="r1", session_id="s1", conversation_id="c1", cart_id="cart1"
    )

    assert "MOST RECENT SUBJECT" not in DeepAgentsRuntime._build_user_message(
        None, state, identity
    )
