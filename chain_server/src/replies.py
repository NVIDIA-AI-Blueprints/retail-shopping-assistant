# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checks and receipts applied to the turn's final reply."""

from __future__ import annotations

import json
from typing import Any

from .agenttypes import Cart, State
from .cart_format import _format_cart
from .message_shape import (
    _current_turn_messages,
    _message_type,
    _result_messages,
    _value,
)
from .tools.evidence import evidence_of
from .tools.skill_gate import SKILL_ACTIVATION_TOOL_NAME


def _products_found_receipt(state: Any) -> str:
    """Answer from the products this turn actually found, or "" if none.

    A turn that fetched a Cancun forecast, had one search refused and its retry
    succeed, then ran out of budget before writing anything, told the shopper
    "I could not complete that shopping request. Please try again." The work was
    done and thrown away, and the shopper was asked to pay for it twice.

    This is not a reply the assistant composed -- it names what was found and
    nothing more, because everything that would need judgement is exactly what
    there was no budget left to do.
    """

    products = [
        record
        for record in (getattr(state, "product_results", None) or [])
        if isinstance(record, dict) and record.get("display_name")
    ]
    if not products:
        return ""
    lines = ["Here is what I found before I ran out of time on this request:"]
    seen: set[str] = set()
    for record in products:
        name = str(record.get("display_name"))
        if name in seen:
            continue
        seen.add(name)
        price = record.get("price")
        amount = (
            f" -- {price.get('amount')} {price.get('currency')}"
            if isinstance(price, dict) and price.get("amount") is not None
            else ""
        )
        lines.append(f"- {name}{amount}")
        if len(seen) >= 6:
            break
    lines.append(
        "Ask me about any of these, or say what to change and I will search "
        "again."
    )
    return "\n".join(lines)


def _committed_effect_receipt(
    effects: list[dict[str, Any]],
    cart: Cart | None,
) -> str:
    """Tell the shopper exactly what was committed before the turn failed."""

    lines = [
        "Something went wrong finishing that request, but a cart change was "
        "already applied:",
        "",
    ]
    for effect in effects:
        operation = str(effect.get("operation") or "changed")
        target = str(
            effect.get("product_id") or effect.get("cart_line_id") or "an item"
        )
        quantity = effect.get("quantity")
        detail = f" (quantity {quantity})" if isinstance(quantity, int) else ""
        lines.append(f"- {operation}: {target}{detail}")
    lines.append("")
    if cart is not None:
        lines.append(_format_cart(cart))
        lines.append("")
    lines.append(
        "Please review your cart before retrying so the change is not applied "
        "twice."
    )
    return "\n".join(lines)


def _has_grounding_authority(state: State, current_evidence: str) -> bool:
    """Return whether this turn has any authority to check a draft against.

    Every turn hydrates memory lanes before the model runs. Gating the grounding
    editor on current-turn *tool* evidence alone discards that hydrated context:
    a follow-up or styling turn grounded in the historical product index or the
    authoritative cart would skip grounding entirely, leaving the draft free to
    assert product facts nothing supports.

    Dialogue is deliberately excluded. It establishes shopper intent, never
    product, policy, inventory, or cart fact, so it is not something a product
    claim can be checked against.
    """

    return bool(
        current_evidence
        or state.historical_product_sets
        or state.cart.contents
    )


def _has_search_only_tool_evidence(result: Any, *, request_id: str) -> bool:
    """Return whether current-turn commerce evidence contains only searches."""

    tool_names: list[str] = []
    has_search_result = False
    for message in _current_turn_messages(_result_messages(result), request_id):
        if _message_type(message) != "tool":
            continue
        name = str(_value(message, "name") or "")
        returned_results = (evidence_of(message) or {}).get("outcome") == "results"
        if not name and returned_results:
            name = "search_catalog_tool"
        if name == SKILL_ACTIVATION_TOOL_NAME:
            continue
        tool_names.append(name)
        if name == "search_catalog_tool" and returned_results:
            has_search_result = True
    return (
        has_search_result
        and set(tool_names) == {"search_catalog_tool"}
    )


def _media_failure_response(media_analysis: str) -> str:
    detail = "video/image understanding is unavailable for this turn"
    try:
        parsed = json.loads(media_analysis)
    except json.JSONDecodeError:
        parsed = {}
    if isinstance(parsed, dict):
        summary = str(parsed.get("summary") or "").strip()
        if summary:
            detail = _clean_media_failure_detail(summary)

    return (
        f"I could not analyze the attached media because {detail}. "
        "Please describe the item in text, such as color, silhouette, material, "
        "and any visible details, and I can search the catalog from that description."
    )


def _clean_media_failure_detail(summary: str) -> str:
    detail = summary.strip()
    for prefix in (
        "Media was attached, but ",
        "Video was attached, but ",
        "Image was attached, but ",
    ):
        if detail.startswith(prefix):
            detail = detail[len(prefix):]
            break
    if detail:
        detail = detail[0].lower() + detail[1:]
    return detail.rstrip(". ") or "video/image understanding is unavailable for this turn"


def _in_presentation_order(
    products: list[dict[str, Any]],
    reply: str,
    groups: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """The shown products, ordered as the reply presents them.

    The cards and the words are the same list to a shopper, so "the second one"
    has to mean one product. Left alone, the cards would follow the catalog's
    ranking and the sentences whatever the model wrote.

    The order is settled once, here, where the reply and the products are both
    in hand -- so every consumer downstream renders one order rather than
    each choosing its own.

    Within a group, never across them. A shopper asked for dresses and shoes
    sees two headed lists, and a reply that discusses a shoe before finishing
    with the dresses would otherwise lift that shoe into the dresses. The
    groups keep the order they were asked for; only the products inside one
    are sorted by where the reply names them.
    """

    if not groups:
        return _as_the_reply_names_them(products, reply)
    held: dict[str, dict[str, Any]] = {
        str(product.get("product_id") or ""): product
        for product in products
        if str(product.get("product_id") or "")
    }
    ordered: list[dict[str, Any]] = []
    placed: set[str] = set()
    for group in groups:
        members = []
        for product_id in group.get("product_ids") or []:
            product = held.get(str(product_id))
            if product is not None and str(product_id) not in placed:
                members.append(product)
                placed.add(str(product_id))
        ordered.extend(_as_the_reply_names_them(members, reply))
    # A product no group claimed still belongs on the screen. The name lookup
    # puts products in front of the shopper without going through a scope, so
    # this is not the empty case it looks like.
    ordered.extend(
        product
        for product in products
        if str(product.get("product_id") or "") not in placed
    )
    return ordered


def _as_the_reply_names_them(
    products: list[dict[str, Any]],
    reply: str,
) -> list[dict[str, Any]]:
    """One list, sorted by where the reply first names each product.

    This looks for the exact display names the service itself produced. It reads
    nothing else out of the reply, and decides nothing but sequence: a product
    the reply never names keeps its ranking, after the ones it does.
    """

    if not products or not reply:
        return products
    mentioned: list[tuple[int, int, dict[str, Any]]] = []
    unmentioned: list[tuple[int, dict[str, Any]]] = []
    for rank, product in enumerate(products):
        name = str(product.get("display_name") or "")
        at = reply.find(name) if name else -1
        if at >= 0:
            mentioned.append((at, rank, product))
        else:
            unmentioned.append((rank, product))
    # Rank breaks ties, so two products named in the same breath keep the
    # catalog's order between them.
    mentioned.sort(key=lambda item: (item[0], item[1]))
    return [product for _at, _rank, product in mentioned] + [
        product for _rank, product in unmentioned
    ]


def _images_in_product_order(
    images: dict[str, str],
    products: list[dict[str, Any]],
) -> dict[str, str]:
    """The image map, following the product order, keeping every entry.

    The cards render from this map, so it has to agree with the list beside it.
    Anything it holds that the products do not name is kept at the end rather
    than dropped: it was shown, and losing it would remove a card rather than
    move one.
    """

    if not images:
        return images
    named = [
        str(product.get("display_name") or "")
        for product in products
        if str(product.get("display_name") or "") in images
    ]
    seen = set(named)
    return {
        **{name: images[name] for name in named},
        **{name: url for name, url in images.items() if name not in seen},
    }
