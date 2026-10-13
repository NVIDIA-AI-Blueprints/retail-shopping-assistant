# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a turn streams to the browser besides its reply text."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping, Sequence
from typing import Any

from ..agenttypes import State
from ..media_perception import summarize_media_analysis

logger = logging.getLogger(__name__)


def _emit_media_progress(on_progress: Any, state: State) -> None:
    """Send what the vision model saw, if anything and if anyone is listening.

    A guarded turn gets the stage only. Perception runs before the output rail,
    so the observations have not been vetted yet, and this message reaches the
    shopper seconds before the one that would have vetted them.

    An unguarded turn is not withholding anything, and nothing here would vet
    it later, so it sends the card as it always did.

    Failing here must never cost a turn: this is a progress message, and a turn
    that answered correctly but could not describe its own perception step is
    still a turn that answered correctly.
    """

    if on_progress is None or not getattr(state, "media", None):
        return
    try:
        if state.guardrails:
            message = {
                "type": "progress",
                "payload": {"stage": "media_perception", "status": "completed"},
            }
        else:
            summary = summarize_media_analysis(state.media_analysis or "")
            if not summary:
                return
            message = {"type": "media_analysis", "payload": summary}
        on_progress(json.dumps({**message, "timestamp": time.time()}))
    except Exception as exc:  # noqa: BLE001 - progress never breaks a turn.
        logger.warning("Could not emit media progress: %s", type(exc).__name__)


def the_showing(
    products: Sequence[Any],
    groups: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """This turn's products as the shopper sees them: a list of headed groups.

    A showing is not a queue. Asked for dresses and shoes, the shopper reads
    two headed lists and counts from one inside each, so "the second shoes" is
    a question the structure can answer. Stored flat, it was eight products in
    a row and that question had no answer at all.

    Numbering restarts inside each group, because that is how the shopper
    counts. Everything downstream -- the screen, the reply, resolution -- is
    handed this same structure, so none of them has to derive an order and
    none of them can derive a different one.

    Products no group claims are one unheaded group at the end, which is also
    how a conversation recorded before groups existed reads back.
    """

    held = {
        str(product.get("product_id") or ""): product
        for product in products
        if isinstance(product, Mapping) and str(product.get("product_id") or "")
    }
    shown: list[dict[str, Any]] = []
    placed: set[str] = set()
    for group in groups or []:
        members = []
        for product_id in group.get("product_ids") or []:
            product = held.get(str(product_id))
            if product is not None and str(product_id) not in placed:
                members.append(product)
                placed.add(str(product_id))
        if members:
            shown.append(
                {
                    "heading": _a_heading_that_is_not_a_product(
                        str(group.get("heading") or ""), products
                    ),
                    "products": _numbered_within_the_group(members),
                }
            )
    unclaimed = [
        product
        for product in products
        if isinstance(product, Mapping)
        and str(product.get("product_id") or "") not in placed
    ]
    if unclaimed:
        shown.append(
            {"heading": "", "products": _numbered_within_the_group(unclaimed)}
        )
    return shown


def _streamed_in_group_order(
    showing: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """The showing flattened for the wire, every product naming its group."""

    return [
        {**product, "group": group.get("heading") or ""}
        for group in showing
        for product in group.get("products") or []
    ]


def _a_heading_that_is_not_a_product(heading: str, products: Sequence[Any]) -> str:
    """This heading, unless it is the name of a product being shown under it.

    Asked to add one tote by name, the model sends that name as the product
    type, and the group of four totes comes out headed "Ombre Canvas Tote Bag"
    -- three of which are not that. An equality check against the showing's own
    products, not a word list: the data to settle it is already in hand.
    """

    wanted = _normalized_display_name(heading)
    if not wanted:
        return ""
    for product in products:
        if not isinstance(product, Mapping):
            continue
        if _normalized_display_name(str(product.get("display_name") or "")) == wanted:
            return ""
    return heading


def _normalized_display_name(name: str) -> str:
    return " ".join(name.split()).casefold()


def _numbered_within_the_group(
    products: Sequence[Any],
) -> list[dict[str, Any]]:
    """The group's products, each carrying the place it holds under its heading.

    So the client renders a given order instead of deriving one. It had been
    deriving one: the chat row was built from a name-keyed image map, the
    product list was matched back into it by display name, and the panel kept
    its own ordering state. Three mechanisms standing in for a number.

    Stamped here rather than on `state.product_results` because that list is
    re-parsed as `ProductSummary`, which forbids unknown fields. It belongs at
    this boundary regardless: where a product sits on a screen is a fact about
    how this turn was presented, not a fact about the product.
    """

    return [
        {**product, "position": position}
        for position, product in enumerate(products, 1)
        if isinstance(product, dict)
    ]
