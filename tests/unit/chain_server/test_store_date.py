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


def test_a_date_says_nothing_about_where_the_shopper_is() -> None:
    """The guard that matters, and the freedom that does not threaten it.

    Knowing the date must never become "so it is cold where you are" -- that
    is the inference `_SHOPPER_CONTEXT_SYSTEM_RULES` forbids, and August is
    winter in half the world anyway.

    The date block says only this. What the assistant may say about a place
    the shopper named lives in the agent prompt, where it applies whether or
    not a forecast tool exists -- one rule, one channel.
    """

    block = _block()

    assert "does not say where the shopper is" in block
    assert "location, weather and season never follow from it" in block


def test_the_shoppers_own_date_may_differ() -> None:
    """UTC is the store's date, not a claim about where the shopper is."""

    block = _block()

    assert "store's current date" in block
    assert "may differ" in block
