# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build catalog search plans from structured agent intent.

This module does not parse shopper language. The agent or another language
understanding layer supplies structured intent; this builder validates that
intent against catalog-owned capabilities.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from shared.commerce_contracts import CatalogCapabilities


SearchMode = Literal["text", "image", "hybrid"]
SearchStrictness = Literal["unspecified", "hard", "soft"]


class CatalogSearchIntent(BaseModel):
    query: str = ""
    queries: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    min_price: float | None = None
    max_price: float | None = None
    soft_preferences: dict[str, Any] = Field(default_factory=dict)
    strictness: SearchStrictness = "unspecified"
    search_mode: SearchMode | None = None

    @model_validator(mode="after")
    def normalize_query_fields(self) -> "CatalogSearchIntent":
        self.query = self.query.strip()
        self.queries = [query.strip() for query in self.queries if query.strip()]
        self.categories = [
            category.strip() for category in self.categories if category.strip()
        ]
        return self


class CatalogSearchPlan(BaseModel):
    should_search: bool
    queries: list[str] = Field(default_factory=list)
    hard_filters: dict[str, Any] = Field(default_factory=dict)
    soft_preferences: dict[str, Any] = Field(default_factory=dict)
    strictness: SearchStrictness = "unspecified"
    search_mode: SearchMode = "text"
    top_k: int = 4
    no_search_reason: str | None = None


def build_catalog_search_plan(
    intent: CatalogSearchIntent,
    capabilities: CatalogCapabilities,
    *,
    has_image: bool = False,
    top_k: int = 4,
) -> CatalogSearchPlan:
    queries = intent.queries or ([intent.query] if intent.query else [])
    mode = _search_mode(intent.search_mode, capabilities, has_image=has_image)
    hard_filters = _hard_filters(intent, capabilities)
    soft_preferences = _soft_preferences(intent, capabilities)

    if not queries and not has_image:
        return CatalogSearchPlan(
            should_search=False,
            search_mode=mode,
            top_k=top_k,
            strictness=intent.strictness,
            soft_preferences=soft_preferences,
            hard_filters=hard_filters,
            no_search_reason="missing_query_or_image",
        )

    return CatalogSearchPlan(
        should_search=True,
        queries=queries,
        hard_filters=hard_filters,
        soft_preferences=soft_preferences,
        strictness=intent.strictness,
        search_mode=mode,
        top_k=top_k,
    )


def _hard_filters(
    intent: CatalogSearchIntent,
    capabilities: CatalogCapabilities,
) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    category_values = _filter_values(capabilities, "category")
    if category_values is not None:
        categories = [
            category
            for category in intent.categories
            if category in category_values
        ]
        if categories:
            filters["category"] = categories

    price_filter = capabilities.filters.get("price")
    if price_filter is not None and price_filter.type == "number":
        if intent.min_price is not None and intent.min_price > 0:
            filters["min_price"] = intent.min_price
        if intent.max_price is not None and intent.max_price > 0:
            filters["max_price"] = intent.max_price

    if (
        "min_price" in filters
        and "max_price" in filters
        and filters["min_price"] > filters["max_price"]
    ):
        filters.pop("min_price")
        filters.pop("max_price")

    return filters


def _soft_preferences(
    intent: CatalogSearchIntent,
    capabilities: CatalogCapabilities,
) -> dict[str, Any]:
    return {
        key: value
        for key, value in intent.soft_preferences.items()
        if key in capabilities.soft_facets and value not in (None, "", [], {})
    }


def _search_mode(
    requested: SearchMode | None,
    capabilities: CatalogCapabilities,
    *,
    has_image: bool,
) -> SearchMode:
    supported = set(capabilities.retrieval_modes)
    if requested in supported:
        return requested
    if has_image and "hybrid" in supported:
        return "hybrid"
    if has_image and "image" in supported:
        return "image"
    return "text"


def _filter_values(
    capabilities: CatalogCapabilities,
    filter_name: str,
) -> set[str] | None:
    capability = capabilities.filters.get(filter_name)
    if capability is None or capability.type != "enum":
        return None
    return set(capability.values)
