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

import json
from datetime import UTC, datetime
from datetime import date as CalendarDate
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from shared.commerce_contracts import (
    Cart as CommerceCart,
)
from shared.commerce_contracts import (
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


def _format_words_this_catalog_cannot_filter_on(
    set_aside: dict[str, list[str]],
) -> str:
    """Say which words were ranked on rather than filtered by, and what that means.

    A filter this catalog does not advertise is dropped rather than refused,
    and the words stay in the query where the index can rank on them. That is
    a weaker promise than a filter: a colour filter guarantees every result is
    that colour, ranking only makes them likelier to be near it. The
    difference is the shopper's to know about, so it is said here rather than
    left for them to find in a product page.
    """

    if not set_aside:
        return ""
    return (
        "SEARCH_WORDS_RANKED_NOT_FILTERED: "
        + json.dumps(set_aside, sort_keys=True, default=str)
        + " -- this catalog does not advertise these, so they could not be "
        "filters. The results were ranked on the words instead, which does "
        "not guarantee any of them match. Check the results against what was "
        "asked for, and if none of them is it, say so plainly rather than "
        "offering the nearest thing as though it were."
    )


def _format_colour_words_read_as_advertised_ones(
    colours_mapped: dict[str, list[str]],
) -> str:
    """Say which colour word was read as which advertised colours.

    The shopper is owed this both ways. They did not get the word they said,
    and what they did get is a real filter rather than a ranking -- so unlike
    the note above, every result here genuinely is one of these colours. Saying
    it in the same register as the results lets the reply pass it on plainly
    instead of implying the shade was an exact match.
    """

    if not colours_mapped:
        return ""
    return (
        "SEARCH_COLOUR_READ_AS: "
        + json.dumps(colours_mapped, sort_keys=True, default=str)
        + " -- this catalog does not list these colour words, so the closest "
        "colours it does list were filtered on instead. Every result really "
        "is one of those colours. Name the colour the shopper is seeing "
        "rather than implying it is the word they used."
    )


def _format_excluded_near_miss(near_miss: Any) -> str:
    """Say what the filter removed, so a question about the filter can be answered.

    A filtered search can confirm that something fits and can never report
    that it does not. Asked whether a $169.99 bracelet was inside a $150
    budget, a turn searched bracelets under the $110.01 still unspent; the
    bracelet was retrieved, ranked, and dropped by the filter, and its absence
    from the results was read as absence from the shop. The shopper was told
    this catalog does not stock a product it sells.

    It is evidence and not a result. It fails the search it came from, so it
    is not offered, not counted and not shown -- it is here to be *told*
    about, which is what was asked for.
    """

    if near_miss is None:
        return ""
    name = str(getattr(near_miss, "display_name", "") or "").strip()
    if not name:
        return ""
    payload: dict[str, Any] = {"display_name": name}
    price = getattr(near_miss, "price", None)
    amount = getattr(price, "amount", None)
    if amount is not None:
        # The number, not the Money object. The price is the whole point of
        # this line -- it is what the shopper asked about -- and serialising
        # the wrapper puts a currency field between them and the answer.
        payload["price"] = amount
    return (
        "EXCLUDED BY A FILTER ON THIS SEARCH: "
        + json.dumps(payload, sort_keys=True, default=str)
        + " -- this product exists in the catalog and does not satisfy the "
        "filter, which is why it is not among the results. It is not an "
        "option and must not be offered or listed as one. State it only to "
        "answer what was asked: never say this shop has no such product."
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
    advertised_subcategories: list[str] | None = None,
) -> str:
    """Record a model-selected advertised parent for honest response framing.

    The parent alone does not say whether the shopper's kind is here. Asked to
    match a look containing jeans, the model chose `apparel` -- true of every
    garment ever made -- and ranked dresses against "dark blue straight-leg
    jeans". What settles it is the list of subcategories that parent actually
    holds: blouses, camisoles, dresses, jumpsuits, skirts, sweaters. None is a
    trouser, and that is a fact rather than a judgement, so it travels with the
    relation instead of being left to be inferred.
    """

    payload = {
        "relation": "model_selected_parent_category",
        "requested_product_type": requested_product_type,
        "advertised_category": advertised_category,
    }
    if advertised_subcategories:
        payload["advertised_subcategories"] = list(advertised_subcategories)
    return (
        f"{_SEARCH_SCOPE_RELATION_EVIDENCE_PREFIX} "
        + json.dumps(payload, sort_keys=True)
    )


def _format_search_unadvertised_type_evidence(
    *,
    requested_product_type: str,
    searched_types: list[str],
    advertised_subcategories: list[str] | None = None,
) -> str:
    """Record that the shopper's product type is not one this catalog lists.

    Shown a video with jeans in it, the model searched `subcategory: skirts`
    and presented what came back as the answer. Nothing said the shopper's word
    had been swapped for another, because the disclosure only fired when a bare
    parent category was chosen -- a scope naming real subcategories looked like
    a direct search for exactly what was asked for.

    Whether the swap is sound is a judgement no check here can make: a pump
    really is a heel, and a jean really is not a skirt. What is certain, and
    checkable, is that the shopper's type is not advertised. So it is recorded,
    and the reply has to own it.
    """

    payload = {
        "relation": "model_selected_advertised_types",
        "requested_product_type": requested_product_type,
        "requested_type_is_advertised": False,
        "searched_types": list(searched_types),
    }
    if advertised_subcategories:
        payload["advertised_subcategories"] = list(advertised_subcategories)
    return (
        f"{_SEARCH_SCOPE_RELATION_EVIDENCE_PREFIX} "
        + json.dumps(payload, sort_keys=True)
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


#: Said once per search result, not once per product. It is a fact about the
#: search, identical for every hit, and thirteen hits repeated it thirteen
#: times -- 2,665 of one result's 13,346 characters, more than the product
#: facts themselves, re-sent on every later model call of the turn.
SEARCH_RESULT_ATTRIBUTE_LIMIT_NOTE = (
    "DETAILS: Any attribute not listed under a product above is not carried by "
    "this search result. Read it with get_product_details_tool and that "
    "PRODUCT_REF before stating it; absence here is not evidence that it is "
    "unknown."
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
    attributes = record.get("attributes") or {}
    if attributes:
        lines.append("CONFIRMED_ATTRIBUTES:")
        lines.extend(
            f"- {name.replace('_', ' ')}: {value}"
            for name, value in attributes.items()
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


def _format_cart_add_result(
    added: list[str],
    failed: list[str],
    cart: Cart,
    ready: list[str] | None = None,
) -> str:
    """Report an add, including what was established but not written.

    The add is all or nothing, so one unanswered item holds back the rest. That
    is deliberate. What was not deliberate is that the held-back items vanished
    from the result: a shopper who gave a correct size for the boots and a
    letter size for the sweater was asked for both again, because nothing told
    the model the boots were already settled.
    """

    lines = ["CART_ADD_RESULT"]
    if added:
        lines.append("Added:")
        lines.extend(added)
    if failed:
        lines.append("Failed:")
        lines.extend(failed)
    if ready:
        lines.append(
            "Established, not added -- the add is all or nothing, so these are "
            "waiting on the item above. Do not ask for these again:"
        )
        lines.extend(ready)
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
            + (f" | size {line.size}" if line.size else "")
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
        # The size is what makes two lines of one dress make sense to a
        # shopper reading their own cart back.
        size = item.get("size")
        size_text = f" (size {size})" if size else ""
        lines.append(
            f"- CART_LINE_ID: {cart_line_id} | "
            f"{item.get('amount', 1)} x {item.get('item', '')}{size_text}{suffix}"
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


def _format_size_change_result(
    *,
    display_name: str,
    from_size: str,
    to_size: str,
    quantity: int,
    cart: Cart | CommerceCart | None,
    old_line_removed: bool,
    old_line_id: str,
) -> str:
    """Report a size change as the one change it is, or say what is left over.

    The add runs before the remove, so that a failure between them leaves the
    shopper an extra line rather than nothing. That is also why there are two
    reports: on the unhappy path the cart really does hold both sizes, and the
    turn has to say so and carry the id that finishes the job, rather than
    announce a replacement that only half happened.
    """

    held = from_size or "onesize"
    if old_line_removed:
        lines = [
            f"CART SIZE CHANGED: {display_name} is now qty {quantity}, size "
            f"{to_size}. The size {held} line was removed."
        ]
    else:
        lines = [
            f"CART SIZE PARTIALLY CHANGED: size {to_size} was added for "
            f"{display_name}, but the size {held} line could not be removed, "
            "so the cart holds both. Remove it with remove_cart_item_tool "
            f"using CART_LINE_ID {old_line_id}. Tell the shopper what the "
            "cart actually holds, not what was asked for."
        ]
    if cart is not None:
        lines.append(_format_cart_lines(cart))
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

class WeatherForecastInput(BaseModel):
    """The agent-facing shape of a forecast request.

    Separate from `WeatherRequest` for one reason: the field is called `city`.
    Twice the prose form of this rule was ignored -- "going to Italy tomorrow"
    called the tool 5/5 and forecast "Italia" at 74-101F as though a country
    had one temperature -- and a rule the model reads is weaker than a
    parameter it has to fill. `location` invites any place; `city` does not.
    """

    model_config = ConfigDict(extra="forbid")

    city: str = Field(
        ...,
        min_length=1,
        max_length=120,
        description=(
            "One city, town or postal code -- Cancun, Napa CA, 94558. Never a "
            "country, region or coastline: they have no single weather, so ask "
            "the shopper which city instead of calling."
        ),
    )
    #: Where the place came from, as a parameter rather than a rule, for the
    #: reason above: the prose form said "the shopper named a CITY" without
    #: saying when. "It's going to snow when we get back" names no place, so
    #: the assistant reached for the wedding city of an earlier turn and
    #: answered a shopper describing snow with the forecast for Rome at 77-97F.
    #:
    #: This asked for the current turn's words and nothing else, which is one
    #: turn narrower than the bug. Two things made Rome wrong and neither was
    #: the age of the citation: the shopper had said what the conditions would
    #: be, and "when we get back" is home rather than the city of the trip.
    #: Meanwhile "will I need a jacket in the evening", nine turns into
    #: planning one trip to one city, names no place either -- and was refused,
    #: then answered with invented weather, which is the outcome the refusal
    #: exists to prevent. So the field asks which place they are asking about
    #: now, and names the two things that disqualify a carried-over one.
    #:
    #: Nothing to quote is the signal. Ask which place they mean.
    shopper_words_naming_the_place: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description=(
            "Quote the shopper's own words naming the place they are asking "
            "about NOW. If they have already said what the weather will be, "
            "they are the authority on their own trip, no forecast is needed "
            "at all, and this is not a call to make -- dress what they told "
            "you. Usually from THIS turn. Words from an earlier turn count "
            "only when this turn carries that same trip forward and puts no "
            "other place in play: nine turns into planning one trip, \"will "
            "I need a jacket in the evening\" is asking about that city. What "
            "never counts is a place they have moved off -- \"when we get "
            "back\" is home, not the city of the trip, and the trip's "
            "forecast contradicts them; served exactly that, the assistant "
            "recommended a satin dress and ballet flats for snow. If you "
            "cannot tell which place they mean, there is nothing to quote: "
            "ask which place they mean instead, and do not describe "
            "conditions you have not fetched."
        ),
    )
    date: CalendarDate | None = Field(
        default=None, description="One exact ISO date, resolved against TODAY."
    )
    start_date: CalendarDate | None = Field(
        default=None, description="Inclusive ISO range start; use with end_date."
    )
    end_date: CalendarDate | None = Field(
        default=None, description="Inclusive ISO range end; use with start_date."
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
        "STILL_SHOW_THE_CLOTHES: a forecast is not an answer on its own. "
        "Search and show real pieces in the same reply. Live, a forecast turn "
        "returned weather and generic advice with nothing to buy, three times "
        "out of three -- a shop that talked about the weather and forgot to "
        "sell anything.",
        f"PLACE_RESOLVED_BY_PROVIDER: {getattr(result, 'resolved_location', '')}",
    ]
    for day in getattr(result, "days", []) or []:
        parts = [f"{day.date.isoformat()}: {day.condition}"]
        if day.temperature_low_f is not None and day.temperature_high_f is not None:
            parts.append(
                f"{day.temperature_low_f:.0f}-{day.temperature_high_f:.0f}F"
            )
        if day.precipitation_probability_pct is not None:
            parts.append(
                f"precipitation {day.precipitation_probability_pct:.0f}%"
            )
        if day.precipitation_types:
            parts.append("as " + ", ".join(day.precipitation_types))
        lines.append("  " + "; ".join(parts))
    lines.append(
        "SAY_WHICH_PLACE: open by naming the place these numbers are for, as "
        "the place you chose to look up rather than as settled fact, and "
        "invite the correction in the same breath -- \"using Rome for the "
        "forecast; say if you meant somewhere else\". Naming it is not enough "
        "on its own: a shopper who said only \"Italy\" never chose the city "
        "you picked, and live, one who meant Florence was given Rome's "
        "forecast at 73-101F as though it were where they would be. If what "
        "came back is broader than a town -- a country or a region, like "
        "Italia or Toscana -- then these numbers describe that whole area and "
        "nowhere in particular: say so plainly, and ask which city they will "
        "be in before dressing them for the weather."
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
    return "\n".join(f"- {name}: image available" for name in retrieved)


def _format_media_summary(media: list[dict[str, Any]]) -> str:
    if not media:
        return "(none)"
    counts: dict[str, int] = {}
    for item in media:
        media_type = str(item.get("type") or "unknown")
        counts[media_type] = counts.get(media_type, 0) + 1
    return ", ".join(f"{count} {media_type}(s)" for media_type, count in sorted(counts.items()))

def _cart_line_key(line: dict) -> tuple:
    """Identity of a cart line for comparison: what a shopper would call
    'the same line' -- the product and the size, not the opaque line id."""
    return (
        str(line.get("item") or line.get("display_name") or ""),
        str(line.get("size") or ""),
    )


def format_cart_change(before: Cart | None, after: Cart | None) -> str:
    """State what this turn did to the cart, as a fact.

    The editor was already told not to claim a cart action absent from CURRENT
    CART, and it still passed "I've added the tote bag back" on a turn where the
    add failed and the cart was unchanged. A prohibition left it comparing two
    lists and judging; this hands it the answer. Computed from the two
    snapshots, so it cannot disagree with the cart.
    """

    if before is None or after is None:
        return "not known for this turn"
    b: dict[tuple, int] = {}
    for line in getattr(before, "contents", []) or []:
        k = _cart_line_key(line)
        b[k] = b.get(k, 0) + int(line.get("amount") or 0)
    a: dict[tuple, int] = {}
    for line in getattr(after, "contents", []) or []:
        k = _cart_line_key(line)
        a[k] = a.get(k, 0) + int(line.get("amount") or 0)
    changes: list[str] = []
    for k in sorted(set(a) | set(b), key=lambda x: (x[0], x[1])):
        name, size = k
        label = f"{name}" + (f" (size {size})" if size else "")
        delta = a.get(k, 0) - b.get(k, 0)
        if delta > 0:
            changes.append(f"- added {label} x{delta}")
        elif delta < 0:
            changes.append(f"- removed {label} x{-delta}")
    if not changes:
        return (
            "NOTHING CHANGED. No item was added, removed, or altered this "
            "turn. Do not tell the shopper otherwise, whatever the draft says."
        )
    return "\n".join(changes)


def format_catalog_shape(capabilities: Any) -> str:
    """What the shop holds, read off the published capabilities.

    A shopper asking "what's the most expensive thing you have" is asking about
    the shop, not about a product, and nothing could answer it. Measured over
    five runs the assistant searched one department and reported its ceiling as
    the catalog's -- a $189.99 purse, a $199.99 crossbody -- in a shop that
    reaches $269.99, or refused outright. Every wrong answer was a category
    ceiling, because a search is the only instrument it had.

    Counts and price ranges per category are published and were never exposed.
    This states them and nothing else: no ranking, no filters, no judgement,
    so it cannot become a second way to search.
    """

    taxonomy = getattr(capabilities, "taxonomy", None)
    categories = getattr(taxonomy, "categories", None) or {}
    if not categories:
        return "CATALOG_SHAPE: the catalog publishes no taxonomy."

    def _money(value: Any) -> str:
        return f"{float(value):.2f}".rstrip("0").rstrip(".") if value is not None else "?"

    lows: list[float] = []
    highs: list[float] = []
    lines: list[str] = []
    for name, category in sorted(categories.items()):
        price = (getattr(category, "filters", None) or {}).get("price")
        low = getattr(price, "min_value", None)
        high = getattr(price, "max_value", None)
        if low is not None:
            lows.append(float(low))
        if high is not None:
            highs.append(float(high))
        subcategories = sorted(getattr(category, "subcategories", None) or {})
        lines.append(
            f"- {name}: {getattr(category, 'product_count', '?')} products, "
            f"{_money(low)}-{_money(high)}, "
            f"subcategories: {', '.join(subcategories) or '(none)'}"
        )

    header = ["CATALOG_SHAPE (published, not a search result):"]
    total = getattr(capabilities, "product_count", None)
    if total is not None:
        header.append(f"- {total} products in total")
    if lows and highs:
        dearest = max(
            categories.items(),
            key=lambda item: float(
                getattr(
                    (getattr(item[1], "filters", None) or {}).get("price"),
                    "max_value",
                    0,
                )
                or 0
            ),
        )[0]
        cheapest = min(
            categories.items(),
            key=lambda item: float(
                getattr(
                    (getattr(item[1], "filters", None) or {}).get("price"),
                    "min_value",
                    10**9,
                )
                or 10**9
            ),
        )[0]
        header.append(
            f"- prices run {_money(min(lows))} to {_money(max(highs))}; the "
            f"dearest things are in {dearest}, the cheapest in {cheapest}"
        )
        header.append(
            "- Nothing exists outside that range. A budget below it is answered "
            "by saying so and naming where prices start -- never by searching "
            "and reporting an empty result."
        )
        header.append(
            "- To name the actual dearest or cheapest item, search that "
            "category at that price bound. A range is not an answer on its own; "
            "show the shopper the thing."
        )
    return "\n".join(header + lines)
