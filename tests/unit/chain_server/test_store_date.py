# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The turn carries a date, because the model does not reliably have one.

Three identical asks for today's date: one answered "Today is August 6, 2026",
two answered "I don't have access to your local date/time from here". Arriving
one turn in three is worse than never arriving -- the shopper meets a different
assistant each time, and "a wedding next week" cannot be planned around.
"""

from __future__ import annotations

from datetime import datetime, timezone

from chain_server.src.response_format import _format_store_date


def _block() -> str:
    return _format_store_date(datetime(2026, 8, 6, 12, 0, tzinfo=timezone.utc))


def test_the_date_is_stated_with_its_weekday() -> None:
    """A weekday is what makes "this weekend" resolvable at all."""

    block = _block()

    assert "2026-08-06" in block
    assert "Thursday" in block


def test_relative_dates_resolve_against_it_and_are_said_out_loud() -> None:
    """Saying the worked-out dates is what lets the shopper catch an error."""

    block = _block()

    assert "next " in block and "this weekend" in block
    assert "Say the calendar dates you worked" in block


def test_a_date_establishes_nothing_but_the_date() -> None:
    """The guard that keeps this from becoming licence to invent.

    A date implies a season only if you also assume a hemisphere and an
    outdoor event. August is winter in half the world. Without this clause the
    date becomes exactly the inference `_SHOPPER_CONTEXT_SYSTEM_RULES` forbids.
    """

    block = _block()

    assert "This establishes nothing else" in block
    assert "not the weather, not a season" in block
    assert "what anyone wears at this time of" in block


def test_the_shoppers_own_date_may_differ() -> None:
    """UTC is the store's date, not a claim about where the shopper is."""

    block = _block()

    assert "store's current date" in block
    assert "may differ" in block
