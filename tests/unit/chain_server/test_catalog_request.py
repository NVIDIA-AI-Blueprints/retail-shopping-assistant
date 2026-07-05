# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from chain_server.src.catalog_request import (
    CatalogSearchIntent,
    build_catalog_search_plan,
)
from shared.commerce_contracts import (
    CatalogCapabilities,
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
            "color": CatalogFilterCapability(
                type="enum",
                operators=["in"],
                source_fields=["color"],
                values=["black", "blue"],
            ),
        },
    )


def test_builds_hard_filters_from_structured_intent() -> None:
    plan = build_catalog_search_plan(
        CatalogSearchIntent(
            semantic_query="work bag",
            filters={
                "category": ["bag"],
                "price": {"max": 60},
                "color": ["blue"],
                "unknown": "ignored",
            },
            strictness="hard",
        ),
        _capabilities(),
    )

    assert plan.should_search is True
    assert plan.semantic_queries == ["work bag"]
    assert plan.hard_filters == {
        "category": ["bag"],
        "price": {"max": 60.0},
        "color": ["blue"],
    }
    assert plan.strictness == "hard"
    assert plan.search_mode == "text"


def test_drops_unsupported_filters_and_invalid_number_range() -> None:
    plan = build_catalog_search_plan(
        CatalogSearchIntent(
            semantic_query="watch",
            filters={
                "category": ["watch"],
                "color": ["purple"],
                "price": {"min": 200, "max": 50},
            },
        ),
        _capabilities(),
    )

    assert plan.hard_filters == {}


def test_defaults_to_hybrid_when_image_is_available() -> None:
    plan = build_catalog_search_plan(
        CatalogSearchIntent(
            semantic_query="similar shoes",
            filters={"category": ["shoes"]},
        ),
        _capabilities(),
        has_image=True,
    )

    assert plan.search_mode == "hybrid"
    assert plan.hard_filters == {"category": ["shoes"]}


def test_no_query_and_no_image_returns_no_search_plan() -> None:
    plan = build_catalog_search_plan(
        CatalogSearchIntent(filters={"category": ["bag"]}),
        _capabilities(),
    )

    assert plan.should_search is False
    assert plan.no_search_reason == "missing_query_or_image"
    assert plan.hard_filters == {"category": ["bag"]}


def test_uses_explicit_queries_over_single_query() -> None:
    plan = build_catalog_search_plan(
        CatalogSearchIntent(
            semantic_query="ignored",
            semantic_queries=["bag", "work tote"],
        ),
        _capabilities(),
    )

    assert plan.semantic_queries == ["bag", "work tote"]


def test_semantic_query_keeps_hard_filters_out_of_search_text() -> None:
    plan = build_catalog_search_plan(
        CatalogSearchIntent(
            semantic_query="floral dresses",
            filters={
                "category": ["dress"],
                "price": {"max": 100},
            },
            strictness="hard",
        ),
        _capabilities(),
    )

    assert plan.semantic_queries == ["floral dresses"]
    assert plan.hard_filters == {
        "category": ["dress"],
        "price": {"max": 100.0},
    }
