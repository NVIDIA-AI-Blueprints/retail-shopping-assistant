# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Product evidence records, and matching a product by the name it was shown under."""

from __future__ import annotations

from typing import Any

from shared.commerce_contracts import (
    CatalogCapabilities,
    CommerceError,
    ProductDetail,
    ProductSummary,
)

from .agenttypes import State
from .catalog_format import _format_detail_value

_PRODUCT_NAME_STOPWORDS = frozenset({"a", "an", "and", "in", "of", "the", "to", "with"})


def _append_product_results(state: State, products: list[ProductSummary]) -> None:
    existing_ids = {
        str(product.get("product_id") or "")
        for product in state.product_results
        if isinstance(product, dict)
    }
    for product in products:
        payload = product.model_dump(mode="json")
        product_id = str(payload.get("product_id") or "")
        if product_id and product_id in existing_ids:
            continue
        state.product_results.append(payload)
        if product_id:
            existing_ids.add(product_id)


_NON_ATTRIBUTE_SEARCH_KEYS = frozenset({"catalog_text", "similarity", "taxonomy"})


def _search_attribute_facts(product: Any) -> dict[str, str]:
    """Structured attributes the catalog confirmed for one search hit.

    The catalog declares which fields are product detail and returns them with
    every search result. They were dropped here, so the model was told to spend
    one of its two product-detail reads to fetch what the response already
    carried -- and when that budget ran out it reported a confirmed attribute as
    unknown.
    """

    attributes = getattr(product, "attributes", None)
    if not isinstance(attributes, dict):
        return {}
    facts: dict[str, str] = {}
    for name, value in sorted(attributes.items()):
        if name in _NON_ATTRIBUTE_SEARCH_KEYS:
            continue
        text = _format_detail_value(value).strip()
        if text:
            facts[str(name)] = text
    return facts


def _search_product_record(product: Any) -> dict[str, Any]:
    """Project one search hit into the record both the model text and the
    composer summary are rendered from.

    Previously the model-visible text was the only rendering and the composer
    parsed it back into this same shape. Building the record once removes the
    round trip, and keeps the two renderings unable to disagree.
    """

    return {
        "product_ref": str(product.product_id),
        "name": str(product.display_name),
        "category": str(getattr(product, "category", "") or ""),
        "price": (
            f"${product.price.amount:.2f} {product.price.currency}"
            if product.price
            else ""
        ),
        "image_url": str(product.image_url or ""),
        "attributes": _search_attribute_facts(product),
    }


def _product_detail_record(product: ProductDetail) -> dict[str, Any]:
    """Project one product-detail read into the record the text renders from."""

    return {
        "product_ref": str(product.product_id),
        "name": str(product.display_name),
        "category": str(product.category or ""),
        "brand": str(product.brand or ""),
        "price": (
            f"${product.price.amount:.2f} {product.price.currency}"
            if product.price
            else ""
        ),
        "image_url": str(product.image_url or ""),
        "details": [
            f"{name.replace('_', ' ')}: {_format_detail_value(value)}"
            for name, value in sorted((product.attributes or {}).items())
        ],
    }


def _same_product_display_name(expected: str, actual: str) -> bool:
    return _normalize_product_name(expected) == _normalize_product_name(actual)


def _where_a_product_was_already_shown(
    historical_product_sets: list[Any] | None,
    product_id: str,
) -> dict[str, Any] | None:
    """Where an earlier turn put this product, if one did.

    Matched on the catalog's id rather than its name, so it answers whether
    this exact product was on the screen and not whether something like it
    was. Newest showing first: the place the shopper is most likely counting
    from is the last one they saw.
    """

    wanted = str(product_id or "").strip()
    if not wanted:
        return None
    for entry in reversed(list(historical_product_sets or [])):
        if not isinstance(entry, dict):
            continue
        for product in entry.get("products") or []:
            if not isinstance(product, dict):
                continue
            if str(product.get("ref") or "").strip() != wanted:
                continue
            return {
                # The index writes this as turn_seq, not turn_sequence.
                "turn_sequence": entry.get("turn_seq"),
                "position": product.get("position"),
                "group": product.get("group") or "",
            }
    return None


def _product_detail_failure_message(
    error: CommerceError | None,
    *,
    cart_validation: bool,
) -> str:
    if error is not None and error.code == "product_not_found":
        return (
            "The product is no longer present in the active catalog. "
            "Search again before adding it."
            if cart_validation
            else (
                "That product is no longer available in the active catalog. "
                "Search the catalog again before using its details."
            )
        )
    if error is not None and error.retryable:
        return (
            "The catalog is temporarily unavailable, so the cart was not changed. "
            "Please try again."
            if cart_validation
            else "Product details are temporarily unavailable. Please try again."
        )
    return (
        "The product could not be verified, so the cart was not changed. "
        "Search again before adding it."
        if cart_validation
        else "Product details could not be verified. Search the catalog again."
    )


def _normalize_product_name(value: str) -> str:
    chars = []
    for char in str(value or "").casefold():
        chars.append(char if char.isalnum() else " ")
    return " ".join("".join(chars).split())


def _product_name_tokens(value: str) -> list[str]:
    return [
        token
        for token in _normalize_product_name(value).split()
        if token not in _PRODUCT_NAME_STOPWORDS
    ]


def _product_name_tokens_match(
    query_tokens: set[str],
    product_tokens: list[str],
    *,
    required_overlap: int,
) -> bool:
    if len(product_tokens) < 2:
        return False
    overlap = query_tokens.intersection(product_tokens)
    required = min(required_overlap, len(set(product_tokens)))
    return len(overlap) >= required


def _detail_fields_already_held(
    product: Any,
    capabilities: CatalogCapabilities,
) -> bool:
    """Return whether evidence already holds every advertised detail field.

    A search returns the same attributes a detail read does -- measured across
    all five advertised categories and 20 products, the detail-only set was
    empty -- so re-reading spends a model round trip to learn what is already in
    hand.

    But "empty on this catalog" is not "empty on every catalog", and silently
    dropping a field is worse than a redundant call. So this asks the capability
    contract rather than assuming: only when evidence covers every field the
    product's own category advertises as a detail field is the read redundant.
    Any gap, any unknown category, and the fetch goes ahead.
    """

    held = getattr(product, "attributes", None)
    if not held:
        return False
    category = getattr(product, "category", None)
    taxonomy = capabilities.taxonomy
    advertised: set[str] = set()
    for name, entry in taxonomy.categories.items():
        subcategories = getattr(entry, "subcategories", {}) or {}
        if category not in (name, *subcategories):
            continue
        advertised |= {
            field
            for field, capability in entry.filters.items()
            if getattr(capability, "detail", False)
        }
    if not advertised:
        return False
    return advertised <= set(held)
