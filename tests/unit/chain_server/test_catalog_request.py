# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from chain_server.src.catalog_request import (
    CatalogSearchIntent,
    build_catalog_search_plan,
)
from shared.commerce_contracts import (
    CatalogCapabilities,
    CatalogCoverage,
    CatalogFieldCapability,
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
            required_constraints={
                "category": ["bag"],
                "price": {"max": 60},
                "color": ["blue"],
            },
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
    assert plan.search_mode == "text"
    assert plan.constraint_issues == []


def test_enum_values_are_canonicalized_without_alias_rules() -> None:
    plan = build_catalog_search_plan(
        CatalogSearchIntent(
            semantic_query="work item",
            required_constraints={"color": ["BLUE"]},
        ),
        _capabilities(),
    )

    assert plan.should_search is True
    assert plan.hard_filters == {"color": ["blue"]}


def test_unsupported_required_constraint_stops_instead_of_weakening_search() -> None:
    plan = build_catalog_search_plan(
        CatalogSearchIntent(
            semantic_query="work bag",
            required_constraints={"unsupported": ["value"]},
        ),
        _capabilities(),
    )

    assert plan.should_search is False
    assert plan.no_search_reason == "unsupported_required_constraint"
    assert plan.constraint_issues == [
        "'unsupported' is not an advertised hard filter"
    ]


def test_invalid_required_constraints_stop_instead_of_being_dropped() -> None:
    plan = build_catalog_search_plan(
        CatalogSearchIntent(
            semantic_query="watch",
            required_constraints={
                "category": ["watch"],
                "color": ["purple"],
                "price": {"min": 200, "max": 50},
            },
        ),
        _capabilities(),
    )

    assert plan.should_search is False
    assert plan.hard_filters == {}
    assert plan.constraint_issues == [
        "'category' contains an unsupported value or operator",
        "'color' contains an unsupported value or operator",
        "'price' contains an unsupported value or operator",
    ]


@pytest.mark.parametrize(
    "bound",
    [True, False, "NaN", "Infinity", "-Infinity", float("nan"), float("inf")],
)
def test_non_finite_or_boolean_required_numeric_bound_stops_search(bound) -> None:
    plan = build_catalog_search_plan(
        CatalogSearchIntent(
            semantic_query="work bag",
            required_constraints={"price": {"max": bound}},
        ),
        _capabilities(),
    )

    assert plan.should_search is False
    assert plan.hard_filters == {}
    assert plan.no_search_reason == "unsupported_required_constraint"
    assert plan.constraint_issues == [
        "'price' contains an unsupported value or operator"
    ]


@pytest.mark.parametrize(
    "bounds",
    [
        {"min": 0, "gte": 100},
        {"max": 200, "lte": 50},
    ],
)
def test_duplicate_numeric_bound_aliases_stop_search(bounds) -> None:
    plan = build_catalog_search_plan(
        CatalogSearchIntent(
            semantic_query="work bag",
            required_constraints={"price": bounds},
        ),
        _capabilities(),
    )

    assert plan.should_search is False
    assert plan.hard_filters == {}
    assert plan.no_search_reason == "unsupported_required_constraint"
    assert plan.constraint_issues == [
        "'price' contains an unsupported value or operator"
    ]


def test_soft_semantic_only_preference_remains_in_semantic_query() -> None:
    plan = build_catalog_search_plan(
        CatalogSearchIntent(semantic_query="maybe cotton dresses"),
        _capabilities(),
    )

    assert plan.should_search is True
    assert plan.semantic_queries == ["maybe cotton dresses"]
    assert plan.hard_filters == {}
    assert plan.constraint_issues == []


def test_required_semantic_only_constraint_stops_before_search() -> None:
    capabilities = _capabilities().model_copy(
        update={
            "fields": {
                "composition": CatalogFieldCapability(
                    type="text",
                    searchable=True,
                    detail=True,
                    coverage=CatalogCoverage(present=10, total=10),
                )
            }
        }
    )

    # Language understanding represents "only cotton dresses" without parsing
    # those words in deterministic catalog code.
    plan = build_catalog_search_plan(
        CatalogSearchIntent(
            semantic_query="dresses",
            required_constraints={"composition": "cotton"},
        ),
        capabilities,
    )

    assert plan.should_search is False
    assert plan.semantic_queries == ["dresses"]
    assert plan.hard_filters == {}
    assert plan.no_search_reason == "unsupported_required_constraint"
    assert plan.constraint_issues == [
        "'composition' is not an advertised hard filter"
    ]


def test_defaults_to_hybrid_when_image_is_available() -> None:
    plan = build_catalog_search_plan(
        CatalogSearchIntent(
            semantic_query="similar shoes",
            required_constraints={"category": ["shoes"]},
        ),
        _capabilities(),
        has_image=True,
    )

    assert plan.search_mode == "hybrid"
    assert plan.hard_filters == {"category": ["shoes"]}


def test_image_only_request_stops_when_catalog_has_no_image_mode() -> None:
    capabilities = _capabilities().model_copy(
        update={
            "retrieval_modes": ["text"],
            "image_search_enabled": False,
        }
    )

    plan = build_catalog_search_plan(
        CatalogSearchIntent(required_constraints={"category": ["shoes"]}),
        capabilities,
        has_image=True,
    )

    assert plan.should_search is False
    assert plan.no_search_reason == "image_search_unavailable"


@pytest.mark.parametrize("requested_mode", ["image", "hybrid"])
def test_explicit_image_mode_is_not_downgraded_to_text(requested_mode) -> None:
    capabilities = _capabilities().model_copy(
        update={"retrieval_modes": ["text"], "image_search_enabled": False}
    )

    plan = build_catalog_search_plan(
        CatalogSearchIntent(
            semantic_query="find something similar",
            search_mode=requested_mode,
        ),
        capabilities,
        has_image=True,
    )

    assert plan.should_search is False
    assert plan.search_mode == requested_mode
    assert plan.no_search_reason == "unsupported_search_mode"


@pytest.mark.parametrize("requested_mode", ["image", "hybrid"])
def test_explicit_image_mode_requires_an_attached_image(requested_mode) -> None:
    plan = build_catalog_search_plan(
        CatalogSearchIntent(
            semantic_query="find something similar",
            search_mode=requested_mode,
        ),
        _capabilities(),
        has_image=False,
    )

    assert plan.should_search is False
    assert plan.search_mode == requested_mode
    assert plan.no_search_reason == "missing_image_for_search_mode"


def test_no_query_and_no_image_returns_no_search_plan() -> None:
    plan = build_catalog_search_plan(
        CatalogSearchIntent(required_constraints={"category": ["bag"]}),
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
            required_constraints={
                "category": ["dress"],
                "price": {"max": 100},
            },
        ),
        _capabilities(),
    )

    assert plan.semantic_queries == ["floral dresses"]
    assert plan.hard_filters == {
        "category": ["dress"],
        "price": {"max": 100.0},
    }
