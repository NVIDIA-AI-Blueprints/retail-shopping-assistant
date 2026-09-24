# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How search results, product records and the catalog's shape are
written for the model.

These turn already-decided data into the exact strings the model reads and
decide nothing themselves. Several are asserted byte-for-byte by the
evidence tests, so a wording change is a visible diff.
"""

from __future__ import annotations

import json
from typing import Any

from shared.commerce_contracts import ProductSummary

from .tools.loop_control import SEARCH_BUDGET_EXHAUSTED_PREFIX

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


def _format_excluded_near_miss(near_miss: dict[str, Any]) -> str:
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

    Rendered from the same dictionary the evidence artifact carries, so the
    line the agent reads and the line the grounding editor reads cannot come
    to differ. Told only here, it reached the agent and never the editor,
    which is the component that writes what the shopper gets.
    """

    if not near_miss:
        return ""
    return (
        "EXCLUDED BY A FILTER ON THIS SEARCH: "
        + json.dumps(near_miss, sort_keys=True, default=str)
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


_SEARCH_RESULT_GROUNDING_NOTE = (
    "SEARCH_RESULT_GROUNDING_NOTE: Use search results for candidate names, prices, "
    "categories, image availability, confirmed filters listed in "
    "SEARCH_FILTER_EVIDENCE, advertised taxonomy listed in "
    "SEARCH_TAXONOMY_EVIDENCE, and modest styling fit only. Treat product names as "
    "display names, not attribute evidence. Do not infer or group-claim "
    "length, color, print, material, care, construction, fit, comfort, weather, "
    "grass, gravel, or best-in-category performance from names or search snippets. "
    "Do not override a confirmed filter based on words in a display name."
)


_SEARCH_NO_MATCH_GROUNDING_NOTE = (
    "SEARCH_NO_MATCH_GROUNDING_NOTE: No product matched all of these filters "
    "together. That says nothing about products outside this search.\n"
    "- Drop one filter, search again, and tell the shopper which one you "
    "dropped. Do not answer with a list of things you could search for.\n"
    "- Never drop a size: a garment in the wrong size is not an alternative. "
    "If nothing comes in that size, say so and name the nearest one.\n"
    "- If the shopper asked for only this, drop nothing: say there is none."
)


#: The last line of a zero-result reply. It names this search's own filters:
#: the general rule above sat before four blocks of evidence echoing the call,
#: and replayed, the model sent the same search back 3 of 3 times; with this
#: line after the evidence it dropped a filter 3 of 3.
_SEARCH_NO_MATCH_NEXT_STEP = (
    "NEXT STEP: search again without one of these filters: {droppable}. "
    "Do not send the same search again."
)


_SEARCH_SCOPE_COMPLETE_NOTE = (
    "SEARCH_SCOPE_COMPLETE: The shopper's current request can now be answered "
    "from this search and existing turn evidence. Answer now. Do not search an "
    "adjacent category or substitute merely because search budget remains. Use "
    "the direct antecedent from recent discussion as the styling anchor; an item "
    "does not need to be in the cart to receive styling advice."
)


_SEARCH_BUDGET_EXHAUSTED_NOTE = (
    f"{SEARCH_BUDGET_EXHAUSTED_PREFIX} No additional catalog searches are "
    "available this turn. Continue with any requested non-search action, or "
    "answer honestly from the grounded products already returned."
)
