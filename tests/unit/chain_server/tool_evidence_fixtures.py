# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build search tool messages the way the tool actually builds them.

Tests used to hand-write the marker strings a search emits and assert against
whatever the parsers pulled back out. That made the tests agree with the
implementation by construction: a test could only fail if parsing changed, not
if the *evidence* changed.

These helpers build one ``SearchEvidence`` and derive both the text and the
artifact from it, mirroring the tool. A test that wants a search result now
states the facts, not the wire format.
"""

from __future__ import annotations

from typing import Any

from chain_server.src.tool_evidence import ProductDetailEvidence, SearchEvidence


def search_evidence(
    *,
    products: list[dict[str, Any]] | None = None,
    confirmed_filters: dict[str, Any] | None = None,
    taxonomy: dict[str, Any] | None = None,
    shopper_guidance: str = "",
    semantic_query: str = "",
    requested_product_type: str | None = None,
    advertised_category: str | None = None,
    outcome: str = "results",
    scope_outcome: dict[str, Any] | None = None,
) -> SearchEvidence:
    return SearchEvidence(
        outcome=outcome,
        taxonomy=taxonomy or {},
        confirmed_filters=confirmed_filters or {},
        semantic_query=semantic_query,
        shopper_guidance=shopper_guidance,
        requested_product_type=requested_product_type,
        advertised_category=advertised_category,
        products=products or [],
        scope_outcome=scope_outcome,
    )


def product(
    name: str,
    *,
    product_ref: str = "",
    category: str = "",
    price: str = "",
    image_url: str = "",
) -> dict[str, Any]:
    """One search-result record, in the shape the tool projects."""

    return {
        "product_ref": product_ref or f"prod_{name.lower().replace(' ', '_')}",
        "name": name,
        "category": category,
        "price": price,
        "image_url": image_url,
    }


def search_tool_message(
    evidence: SearchEvidence,
    *,
    content: str = "SEARCH_RESULT_GROUNDING_NOTE: grounded.",
    tool_call_id: str = "",
    name: str = "search_catalog_tool",
) -> dict[str, Any]:
    """A dict-shaped tool message carrying the typed payload."""

    message: dict[str, Any] = {
        "role": "tool",
        "name": name,
        "content": content,
        "artifact": evidence.as_artifact(),
    }
    if tool_call_id:
        message["tool_call_id"] = tool_call_id
    return message


def product_detail(
    name: str,
    *,
    product_ref: str = "",
    category: str = "",
    brand: str = "",
    price: str = "",
    image_url: str = "",
    details: list[str] | None = None,
) -> dict[str, Any]:
    """One product-detail record, in the shape the tool projects."""

    return {
        "product_ref": product_ref or f"prod_{name.lower().replace(' ', '_')}",
        "name": name,
        "category": category,
        "brand": brand,
        "price": price,
        "image_url": image_url,
        "details": details or [],
    }


def detail_artifact(*records: dict[str, Any]) -> dict[str, Any]:
    return ProductDetailEvidence(products=list(records)).as_artifact()
