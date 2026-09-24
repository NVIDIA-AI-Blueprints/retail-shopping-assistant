# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sections of the per-turn input the agent and grounding editor read."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from .agenttypes import ShopperContext


def _format_store_date(now: datetime | None = None) -> str:
    """Give the turn a date, because the model does not reliably have one.

    Measured three identical asks: one answered "Today is August 6, 2026",
    two answered "I don't have access to your local date/time". A date that
    arrives one turn in three is worse than none, because the shopper gets a
    different assistant each time.

    Deliberately narrow. A date says when the shop is, and nothing else: not
    where the shopper is, not the weather, not a season -- August is winter in
    half the world and irrelevant indoors. Without that clause a date becomes
    the licence to invent exactly the facts the shopper-context rules forbid.
    """

    stamp = (now or datetime.now(UTC)).astimezone(UTC)
    return (
        "TODAY (store's current date, server-resolved):\n"
        f"{stamp:%Y-%m-%d}, a {stamp:%A}, UTC\n"
        "Resolve relative dates the shopper mentions against this -- next "
        "week, this weekend, in two weeks. Say the calendar dates you worked "
        "out so they can correct you. The shopper's own date may differ if "
        "they are far from UTC.\n"
        "This says when the shop is. It does not say where the shopper is, "
        "and their location, weather and season never follow from it.\n"
        "END TODAY"
    )


def _format_shopper_context(context: ShopperContext | None) -> str:
    if context is None:
        return ""
    # The saved ZIP is deliberately absent. Every use of it is forbidden --
    # it is not proof of location, weather, or a product requirement, and the
    # weather slice that would give it a use is dormant. Showing the model a
    # fact and then forbidding every use of it is an invitation, not a
    # safeguard. It stays on the profile record and the picker; it returns
    # here when weather tooling defines what may be concluded from it.
    return (
        "SHOPPER CONTEXT (server-resolved; soft guidance only):\n"
        f"shopper_type: {context.shopper_type}\n"
        f"behavior: {context.behavior}\n"
        "END SHOPPER CONTEXT"
    )


def _format_wearer_audience(audience: list[str] | None) -> str:
    """Say who the last named item was for, without scoping anything.

    A wearer is a property of the item they were named for, not of the
    conversation: after "shades for hubby", "show me some heels" must not be
    scoped to mens.

    The two errors are not the same size. Carrying it wrongly costs the
    shopper the whole result set, silently, with no way to see why. Forgetting
    it costs one question. So the value is reported and the turn decides:
    audience scopes a search only when the turn itself names the person.
    """

    if not audience:
        return ""
    values = ", ".join(sorted(str(value) for value in audience))
    return (
        "SHOPPING FOR (context for reading this turn; not a scope by itself):\n"
        f"audience: {values}\n"
        "The last item the shopper named a person for was for this audience. "
        "Decide this turn's audience from this turn's own words. If they "
        "refer to that person again, including by pronoun -- \"he also needs "
        "a bag\", \"something for her\" -- filter to the values that suit "
        "them. If this turn refers to nobody, send no audience filter at all, "
        "however obviously the person is still around. You may ask whether "
        "they are still shopping for the same person.\n"
        "END SHOPPING FOR"
    )


def _format_retrieved_images(retrieved: dict[str, str] | None) -> str:
    if not retrieved:
        return "(none)"
    return "\n".join(f"- {name}: image available" for name in retrieved)


def _format_media_summary(media: list[dict[str, Any]]) -> str:
    if not media:
        return "(none)"
    counts: dict[str, int] = {}
    for item in media:
        media_type = str(item.get("type") or "unknown")
        counts[media_type] = counts.get(media_type, 0) + 1
    return ", ".join(f"{count} {media_type}(s)" for media_type, count in sorted(counts.items()))


def format_most_recent_subject(state: Any) -> str:
    """Name what the conversation is about now, so a pronoun has an anchor.

    "Add the Jade Suede Heels in a 6", then "actually make those a 7", resolved
    to a dress from eight turns earlier. Nothing was missing: the heels were
    the line directly above the pronoun in the conversation lane, the newest
    showing in the index, and a line in the cart. The model had to derive the
    referent from three places and derived it wrongly.

    So the runtime derives it and states what it got. This is not a fourth copy
    of the conversation -- it is the resolution of it, which is the part that
    was going wrong. What just happened is state, not interpretation.

    Most recent first: what the last turn did to the cart, then what it showed.
    Silent when there is neither, so an opening turn gains nothing to ignore.
    """

    # Only the newest showing for now. What the previous turn did to the cart
    # is computed at the end of a turn for the grounding editor and never
    # carried into the next one, so there is no field to read here yet -- and a
    # line that is always empty is dead code pretending to be a feature.
    lines: list[str] = []
    sets = [
        entry
        for entry in (getattr(state, "historical_product_sets", None) or [])
        if isinstance(entry, dict) and isinstance(entry.get("products"), list)
    ]
    if sets:
        newest = max(sets, key=lambda entry: entry.get("turn_seq") or 0)
        shown = [
            str(item.get("name"))
            for item in newest["products"][:4]
            if isinstance(item, dict) and item.get("name")
        ]
        if shown:
            lines.append(
                f"last shown (turn {newest.get('turn_seq')}): " + "; ".join(shown)
            )
    if not lines:
        return ""
    body = "\n".join(f"- {line}" for line in lines)
    return (
        "MOST RECENT SUBJECT (what the conversation is about right now):\n"
        f"{body}\n"
        'A bare pronoun -- "those", "it", "them", "that one" -- means something '
        "here unless the shopper names another product. Resolve it here first, "
        "and look further back only if nothing here fits."
    )
