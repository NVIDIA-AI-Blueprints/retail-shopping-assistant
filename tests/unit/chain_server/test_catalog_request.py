# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from chain_server.src.catalog_request import (
    CatalogSearchIntent,
    build_catalog_search_plan,
)
from shared.commerce_contracts import (
    CatalogCapabilities,
    CatalogFacetCapability,
    CatalogFilterCapability,
)


def _capabilities() -> CatalogCapabilities:
    return CatalogCapabilities(
        catalog_id="fashion",
        retrieval_modes=["text", "image", "hybrid"],
        image_search_enabled=True,
        filters={
            "category": CatalogFilterCapability(
                type="enum",
                operators=["in"],
                source_fields=["subcategory"],
                values=["bag", "dress", "shoes"],
            ),
            "price": CatalogFilterCapability(
                type="number",
                operators=["gte", "lte"],
                source_fields=["price"],
                min_value=39.9,
                max_value=269.99,
            ),
        },
        soft_facets={
            "style": CatalogFacetCapability(type="text"),
            "occasion": CatalogFacetCapability(type="text"),
        },
    )


def test_builds_hard_filters_from_structured_intent() -> None:
    plan = build_catalog_search_plan(
        CatalogSearchIntent(
            query="work bag",
            categories=["bag"],
            max_price=60,
            soft_preferences={"style": "practical", "unknown": "ignored"},
            strictness="hard",
        ),
        _capabilities(),
    )

    assert plan.should_search is True
    assert plan.queries == ["work bag"]
    assert plan.hard_filters == {"category": ["bag"], "max_price": 60}
    assert plan.soft_preferences == {"style": "practical"}
    assert plan.strictness == "hard"
    assert plan.search_mode == "text"


def test_drops_unsupported_categories_and_invalid_price_range() -> None:
    plan = build_catalog_search_plan(
        CatalogSearchIntent(
            query="watch",
            categories=["watch"],
            min_price=200,
            max_price=50,
        ),
        _capabilities(),
    )

    assert plan.hard_filters == {}


def test_defaults_to_hybrid_when_image_is_available() -> None:
    plan = build_catalog_search_plan(
        CatalogSearchIntent(query="similar shoes", categories=["shoes"]),
        _capabilities(),
        has_image=True,
    )

    assert plan.search_mode == "hybrid"
    assert plan.hard_filters == {"category": ["shoes"]}


def test_no_query_and_no_image_returns_no_search_plan() -> None:
    plan = build_catalog_search_plan(
        CatalogSearchIntent(soft_preferences={"style": "practical"}),
        _capabilities(),
    )

    assert plan.should_search is False
    assert plan.no_search_reason == "missing_query_or_image"
    assert plan.soft_preferences == {"style": "practical"}


def test_uses_explicit_queries_over_single_query() -> None:
    plan = build_catalog_search_plan(
        CatalogSearchIntent(query="ignored", queries=["bag", "work tote"]),
        _capabilities(),
    )

    assert plan.queries == ["bag", "work tote"]
