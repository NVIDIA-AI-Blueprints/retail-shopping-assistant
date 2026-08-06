# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rendering for model-visible and shopper-visible text.

These functions turn already-decided data into the exact strings the model and
the shopper read. They decide nothing: no retrieval, no policy, no control flow,
no state. That is what makes them safe to hold apart from the runtime, and worth
holding apart, because their output is a contract -- several are asserted
byte-for-byte by the evidence tests, so a wording change is a visible diff rather
than a quietly different string.

Moved verbatim from ``deepagents_runtime.py``. Nothing here was edited during the
move; the runtime imports these names back, so behaviour is unchanged.
"""

from __future__ import annotations

from datetime import datetime, timezone

from typing import Any
import json

from shared.commerce_contracts import (
    Cart as CommerceCart,
    CartMutationResult,
    CheckActivePromotionsResult,
    CheckProductAvailabilityResult,
    GetStorePolicyResult,
    ProductSummary,
)

from .agenttypes import Cart, ShopperContext

_SEARCH_FILTER_EVIDENCE_PREFIX = "SEARCH_FILTER_EVIDENCE:"

_SEARCH_TAXONOMY_EVIDENCE_PREFIX = "SEARCH_TAXONOMY_EVIDENCE:"

_SEARCH_DIRECTION_EVIDENCE_PREFIX = "SEARCH_DIRECTION_EVIDENCE:"

_SEARCH_GUIDANCE_EVIDENCE_PREFIX = "SEARCH_GUIDANCE_EVIDENCE:"

_SEARCH_SCOPE_RELATION_EVIDENCE_PREFIX = "SEARCH_SCOPE_RELATION_EVIDENCE:"

_CATALOG_SCOPE_OUTCOME_PREFIX = "CATALOG_SCOPE_OUTCOME:"

_PRODUCT_DETAIL_GROUNDING_NOTE = (
    "PRODUCT_DETAIL_GROUNDING_NOTE: This detail result exposes only "
    "the fields shown below. Material, care, dimensions, closures, fit, "
    "sizing, colorways, and outdoor performance are unavailable unless explicitly "
    "listed. Do not infer them from product names or prior marketing text."
)


def _format_search_group(
    group: dict[str, Any],
    products: list[dict[str, Any]],
    *,
    index: int,
) -> list[str]:
    """Format one search group's bounded guidance and verified products."""

    taxonomy = group.get("taxonomy") or {}
    values = taxonomy.get("subcategory") or taxonomy.get("category") or []
    label = " / ".join(str(value).replace("_", " ") for value in values)
    title = label.title() if label else f"Product group {index}"
    lines = [f"**{title}**", "", "General guidance (not product-specific facts):"]
    lines.extend((str(group["guidance"]), "", "Catalog candidates:", ""))
    for product in products:
        parts = [f"**{product['name']}**"]
        if product.get("price"):
            parts.append(str(product["price"]))
        if product.get("category"):
            parts.append(str(product["category"]).replace("_", " "))
        lines.append("- " + " — ".join(parts))
    lines.append("")
    return lines


def _format_filter_statement(name: str, value: Any) -> str:
    label = name.replace("_", " ")
    if isinstance(value, list):
        values = [str(item).replace("_", " ") for item in value]
        if len(values) == 1:
            return f"{label} is {values[0]}"
        if values:
            return f"{label} is one of {', '.join(values)}"
        return ""
    if isinstance(value, dict):
        bounds = []
        if value.get("min") is not None:
            bounds.append(f"minimum {value['min']}")
        if value.get("max") is not None:
            bounds.append(f"maximum {value['max']}")
        return f"{label} {' and '.join(bounds)}" if bounds else ""
    return f"{label} is {value}"


def _format_search_filter_evidence(filters: dict[str, Any]) -> str:
    """Format canonical hard filters proven by a successful search."""

    return (
        f"{_SEARCH_FILTER_EVIDENCE_PREFIX} "
        + json.dumps(filters, sort_keys=True, default=str)
    )


def _format_search_direction_evidence(semantic_query: str) -> str:
    """Record the model-authored preference used for successful ranking."""

    return (
        f"{_SEARCH_DIRECTION_EVIDENCE_PREFIX} "
        + json.dumps(semantic_query, ensure_ascii=False)
    )


def _format_search_guidance_evidence(shopper_guidance: str) -> str:
    """Record bounded product-agnostic guidance authored before retrieval."""

    return (
        f"{_SEARCH_GUIDANCE_EVIDENCE_PREFIX} "
        + json.dumps({"text": shopper_guidance.strip()}, ensure_ascii=False)
    )


def _format_search_taxonomy_evidence(taxonomy: dict[str, Any]) -> str:
    """Format the advertised taxonomy scope used by a successful search."""

    return (
        f"{_SEARCH_TAXONOMY_EVIDENCE_PREFIX} "
        + json.dumps(taxonomy, sort_keys=True, default=str)
    )


def _format_search_scope_relation_evidence(
    *,
    requested_product_type: str,
    advertised_category: str,
) -> str:
    """Record a model-selected advertised parent for honest response framing."""

    return (
        f"{_SEARCH_SCOPE_RELATION_EVIDENCE_PREFIX} "
        + json.dumps(
            {
                "relation": "model_selected_parent_category",
                "requested_product_type": requested_product_type,
                "advertised_category": advertised_category,
            },
            sort_keys=True,
        )
    )


def _format_search_composed_role_evidence(
    *,
    requested_product_type: str,
    role_advertised_types: list[str],
) -> str:
    """Record that the model, not the shopper, proposed this role.

    Same envelope as the parent-category relation above, because both answer
    the same question for the composer: how does the noun in the evidence
    relate to what the shopper actually said?
    """

    return (
        f"{_SEARCH_SCOPE_RELATION_EVIDENCE_PREFIX} "
        + json.dumps(
            {
                "relation": "model_composed_role",
                "requested_product_type": requested_product_type,
                "role_advertised_types": sorted(role_advertised_types),
            },
            sort_keys=True,
        )
    )


def _format_catalog_scope_outcome(outcome: dict[str, Any]) -> str:
    """Format one bounded non-product catalog outcome for diagnostics."""

    return (
        f"{_CATALOG_SCOPE_OUTCOME_PREFIX} "
        + json.dumps(outcome, sort_keys=True, default=str)
    )


def _format_product_record(record: dict[str, Any]) -> str:
    lines = [
        f"PRODUCT_REF: {record['product_ref']}",
        f"NAME: {record['name']}",
    ]
    if record.get("category"):
        lines.append(f"CATEGORY: {record['category']}")
    if record.get("price"):
        lines.append(f"PRICE: {record['price']}")
    if record.get("image_url"):
        lines.append(f"IMAGE_URL: {record['image_url']}")
    attributes = record.get("attributes") or {}
    if attributes:
        lines.append("CONFIRMED_ATTRIBUTES:")
        lines.extend(
            f"- {name.replace('_', ' ')}: {value}"
            for name, value in attributes.items()
        )
    lines.append(
        "DETAILS: Any attribute not listed above is not carried by this search "
        "result. Read it with get_product_details_tool and this PRODUCT_REF "
        "before stating it; absence here is not evidence that it is unknown."
    )
    return "\n".join(lines)


def _format_product_detail_record(record: dict[str, Any]) -> str:
    lines = [
        _PRODUCT_DETAIL_GROUNDING_NOTE,
        f"PRODUCT_REF: {record['product_ref']}",
        f"NAME: {record['name']}",
    ]
    if record.get("category"):
        lines.append(f"CATEGORY: {record['category']}")
    if record.get("brand"):
        lines.append(f"BRAND: {record['brand']}")
    if record.get("price"):
        lines.append(f"PRICE: {record['price']}")
    if record.get("image_url"):
        lines.append(f"IMAGE_URL: {record['image_url']}")
    if record.get("details"):
        lines.append("DETAILS:")
        lines.extend(f"- {detail}" for detail in record["details"])
    else:
        lines.append("NO_ADDITIONAL_STRUCTURED_DETAILS")
    return "\n".join(lines)


def _format_detail_value(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    if isinstance(value, dict):
        return ", ".join(
            f"{key}={value[key]}" for key in sorted(value)
        )
    return str(value)


def _format_product_refs(products: list[ProductSummary]) -> str:
    return ", ".join(
        f"{product.display_name} (PRODUCT_REF: {product.product_id})"
        for product in products
    )


def _format_cart_add_result(added: list[str], failed: list[str], cart: Cart) -> str:
    lines = ["CART_ADD_RESULT"]
    if added:
        lines.append("Added:")
        lines.extend(added)
    if failed:
        lines.append("Failed:")
        lines.extend(failed)
    lines.append("Current cart:")
    lines.append(_format_cart_lines(cart))
    lines.append("Cart total:")
    lines.append(_format_cart_total(cart))
    return "\n".join(lines)


def _format_cart_lines(cart: Cart | CommerceCart) -> str:
    if isinstance(cart, CommerceCart):
        if not cart.lines:
            return "  (cart is empty)"
        lines = [
            f"  {line.cart_line_id} | {line.display_name} | qty {line.quantity}"
            + (
                f" | {line.unit_price.currency} {line.unit_price.amount:.2f}"
                if line.unit_price
                else ""
            )
            for line in cart.lines
        ]
        if cart.subtotal:
            lines.append(
                f"  SUBTOTAL: {cart.subtotal.currency} {cart.subtotal.amount:.2f}"
            )
        return "\n".join(lines)

    if not cart.contents:
        return "(empty)"
    lines = []
    for item in cart.contents:
        price = item.get("price")
        suffix = ""
        if price is not None:
            try:
                suffix = f" @ ${float(price):.2f}"
            except (TypeError, ValueError):
                suffix = ""
        cart_line_id = item.get("cart_line_id") or item.get("item", "")
        lines.append(
            f"- CART_LINE_ID: {cart_line_id} | "
            f"{item.get('amount', 1)} x {item.get('item', '')}{suffix}"
        )
    return "\n".join(lines)


def _format_cart(cart: Cart) -> str:
    return _format_cart_lines(cart)


def _format_cart_remove_result(
    result: CartMutationResult,
    *,
    fallback: str,
) -> str:
    if not result.ok:
        return result.error.message if result.error else "Cart remove failed."
    message = result.message or fallback
    if result.cart is not None:
        return "\n".join(
            [message, "Current cart:", _format_cart_lines(result.cart)]
        )
    return message


def _format_update_cart_result(
    result: CartMutationResult,
    cart: Cart | CommerceCart | None = None,
) -> str:
    if not result.ok:
        message = result.error.message if result.error else "unknown error"
        return f"CART UPDATE FAILED: {message}"
    lines = ["CART UPDATED"]
    if result.changed_line:
        lines.append(
            f"  {result.changed_line.display_name} → "
            f"qty {result.changed_line.quantity}"
        )
    active_cart = cart if cart is not None else result.cart
    if active_cart is not None:
        lines.append(_format_cart_lines(active_cart))
    return "\n".join(lines)


def _format_policy_result(result: GetStorePolicyResult) -> str:
    if not result.ok or result.policy is None:
        message = result.error.message if result.error else "unknown error"
        return f"POLICY NOT AVAILABLE: {message}"
    policy = result.policy
    return f"STORE POLICY — {policy.title}\n{policy.body}"


def _format_availability_result(result: CheckProductAvailabilityResult) -> str:
    return f"AVAILABILITY ({result.product_ref}): {result.message}"


def _format_promotions_result(result: CheckActivePromotionsResult) -> str:
    status = "YES" if result.active else "NO"
    return f"ACTIVE PROMOTIONS: {status}\n{result.message}"


def _format_cart_total(cart: Cart) -> str:
    if not cart.contents:
        return "Your cart is empty, so the total is $0.00."
    subtotal = 0.0
    missing = []
    lines = []
    for item in cart.contents:
        name = item.get("item", "")
        amount = int(item.get("amount") or 0)
        price = item.get("price")
        if price is None:
            missing.append(name)
            lines.append(f"- {amount} x {name}: price unavailable")
            continue
        line_total = float(price) * amount
        subtotal += line_total
        lines.append(f"- {amount} x {name} @ ${float(price):.2f} = ${line_total:.2f}")
    total = f"Cart total: ${subtotal:.2f}"
    if missing:
        total += f" excluding items without cached prices: {', '.join(missing)}"
    return "\n".join(lines + [total])


#: One event, one date: a turn never needs many forecasts, and each is a paid
#: external call.
WEATHER_CALLS_PER_TURN = 2

WEATHER_NO_DATE = (
    "WEATHER_NEEDS_A_DATE: no forecast was fetched, because no date was given "
    "and today is not what the shopper is dressing for. Ask them, as part of a "
    "styling question rather than as a request for a parameter, and show a "
    "grounded starting point in the same reply."
)

def weather_call_needs_a_date(
    date: Any, start_date: Any, end_date: Any
) -> bool:
    """Whether this call would silently forecast today instead of the event.

    Extracted from the tool closure so it can be tested. Left inside, deleting
    it entirely kept all 1089 tests passing -- the third time today that a
    constant was asserted while the enforcement sat somewhere nothing could
    reach.
    """

    return date is None and start_date is None and end_date is None


WEATHER_BUDGET_EXHAUSTED = (
    "WEATHER_UNAVAILABLE: this turn has already looked up the forecast it is "
    "allowed to. Use what you have and style the occasion; do not guess the "
    "weather."
)

def claim_weather_call(scope: Any) -> bool:
    """Take one of this turn's forecast calls, or report that none are left.

    Extracted from the tool closure so the budget can be tested. It could not
    be before, and deleting the check entirely left all 1081 tests passing --
    a paid external call with no enforced ceiling and nothing to notice.
    """

    with scope.weather_lock:
        if scope.weather_calls >= WEATHER_CALLS_PER_TURN:
            return False
        scope.weather_calls += 1
        return True


def _format_weather_result(result: Any) -> str:
    """Render a forecast, or an honest failure, as evidence for the reply.

    Failures are the common case, not the edge: most event shopping happens
    more than fifteen days ahead, and the horizon is fifteen days. So every
    failure says the same thing -- say plainly that the forecast is not
    available, then style the occasion -- because a turn that cannot see the
    weather is still a turn that can dress someone.
    """

    if not getattr(result, "ok", False):
        # One sentence, because no forecast is the ordinary state rather than
        # an event. Everything about how to behave without one already lives
        # in the agent prompt, where it applies whether or not this tool
        # exists at all.
        return (
            "WEATHER_UNAVAILABLE: "
            + str(getattr(result, "message", "No forecast is available."))
            + " Do not repeat this code. Style the occasion."
        )
    lines = [
        "WEATHER_EVIDENCE: a live forecast for the place and dates below. It "
        "supports what the conditions will be, and nothing about any product. "
        "It never makes an item warm, waterproof or suitable -- say what the "
        "weather is, then reason about the outfit as styling judgement.",
        f"PLACE_RESOLVED_BY_PROVIDER: {getattr(result, 'resolved_location', '')}",
    ]
    for day in getattr(result, "days", []) or []:
        parts = [f"{day.date.isoformat()}: {day.condition}"]
        if day.temperature_low_f is not None and day.temperature_high_f is not None:
            parts.append(
                f"{day.temperature_low_f:.0f}-{day.temperature_high_f:.0f}F"
            )
        parts.append(f"precipitation {day.precipitation_probability_pct:.0f}%")
        if day.precipitation_types:
            parts.append("as " + ", ".join(day.precipitation_types))
        lines.append("  " + "; ".join(parts))
    lines.append(
        "SAY_WHICH_PLACE: name the place the provider resolved, so a shopper "
        "who meant a different one can correct you."
    )
    # The provider's own label and link travel with the data, so the terms are
    # met by whatever provider answered rather than by a constant here.
    attribution = getattr(result, "attribution", None)
    if attribution is not None:
        lines.append(
            f"REQUIRED_ATTRIBUTION: include \"{attribution.label}\" and the "
            f"link {attribution.url} wherever you use this. A forecast is an "
            "estimate, not a guarantee, and never a safety warning."
        )
    return "\n".join(lines)


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

    stamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
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

    This used to read "keep filtering to this audience while it still
    applies", which made a wearer a property of the conversation rather than
    of the item they were named for. "Shades for hubby" then scoped every
    later search: "show me some heels" came back empty in a shop full of
    heels, because they are all womens and the carried audience was not.

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
    lines = []
    for name, image_url in retrieved.items():
        lines.append(f"- {name}: image available")
    return "\n".join(lines)


def _format_media_summary(media: list[dict[str, Any]]) -> str:
    if not media:
        return "(none)"
    counts: dict[str, int] = {}
    for item in media:
        media_type = str(item.get("type") or "unknown")
        counts[media_type] = counts.get(media_type, 0) + 1
    return ", ".join(f"{count} {media_type}(s)" for media_type, count in sorted(counts.items()))
