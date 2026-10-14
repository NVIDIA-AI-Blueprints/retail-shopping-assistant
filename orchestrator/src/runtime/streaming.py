# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a turn streams to the browser besides its reply text."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from threading import Lock
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware

from ..agenttypes import State
from ..media_perception import summarize_media_analysis

logger = logging.getLogger(__name__)

#: What the shopper sees while a tool runs. Tool names are internal and must
#: never reach the browser; a tool missing here shows nothing new.
_TOOL_PROGRESS_LABELS = {
    "search_catalog_tool": "Searching the catalog",
    "get_product_details_tool": "Checking product details",
    "resolve_conversation_products_tool": "Looking back at earlier products",
    "describe_catalog_tool": "Checking what the store carries",
    "get_cart_tool": "Checking your cart",
    "view_cart_total_tool": "Checking your cart",
    "add_cart_items_tool": "Updating your cart",
    "remove_cart_item_tool": "Updating your cart",
    "update_cart_items_tool": "Updating your cart",
    "get_store_policy_tool": "Checking store policies",
    "check_product_availability_tool": "Checking availability",
    "check_active_promotions_tool": "Checking current promotions",
    "get_weather_forecast_tool": "Checking the weather forecast",
}
WRITING_REPLY_LABEL = "Writing your reply"
CHECKING_MESSAGE_LABEL = "Checking your message"
CHECKING_REPLY_LABEL = "Checking the reply"


def media_progress_label(media: Sequence[Mapping[str, Any]]) -> str:
    """What perception is looking at, as the shopper would say it."""

    kinds = {str(item.get("type", "")) for item in media}
    if kinds == {"video"}:
        return "Looking at your video"
    if kinds == {"image"}:
        return "Looking at your photo" if len(media) == 1 else "Looking at your photos"
    return "Looking at what you shared"


def _emit_progress_label(on_progress: Any, label: str) -> None:
    """Tell the browser what the turn is doing now. Never fails the turn."""

    if on_progress is None:
        return
    try:
        on_progress(
            json.dumps(
                {
                    "type": "progress",
                    "payload": {"stage": "activity", "label": label},
                    "timestamp": time.time(),
                }
            )
        )
    except Exception as exc:  # noqa: BLE001 - progress never breaks a turn.
        logger.warning("Could not emit progress: %s", type(exc).__name__)


class ToolProgressMiddleware(AgentMiddleware):
    """Announce each allowed tool call to the browser before it runs.

    Observes only: the request and the tool's result pass through unchanged.
    Tools can run on worker threads, so every message is handed to the event
    loop that owns ``on_progress``.
    """

    def __init__(self, on_progress: Callable[[str], Any]) -> None:
        super().__init__()
        self._on_progress = on_progress
        self._loop = asyncio.get_running_loop()
        self._lock = Lock()
        self._last_label: str | None = None

    def _announce(self, request: Any) -> None:
        tool_call = getattr(request, "tool_call", None) or {}
        label = _TOOL_PROGRESS_LABELS.get(str(tool_call.get("name", "")))
        if label is None:
            return
        with self._lock:
            if label == self._last_label:
                return
            self._last_label = label
        # A closed loop means the turn is already over.
        with contextlib.suppress(RuntimeError):
            self._loop.call_soon_threadsafe(_emit_progress_label, self._on_progress, label)

    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        self._announce(request)
        return handler(request)

    async def awrap_tool_call(
        self, request: Any, handler: Callable[[Any], Awaitable[Any]]
    ) -> Any:
        self._announce(request)
        return await handler(request)


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
