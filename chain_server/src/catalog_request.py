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

from shared.commerce_contracts import CatalogCapabilities, CatalogFilterCapability


SearchMode = Literal["text", "image", "hybrid"]
SearchStrictness = Literal["unspecified", "hard"]


class CatalogSearchIntent(BaseModel):
    query: str = ""
    queries: list[str] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)
    strictness: SearchStrictness = "unspecified"
    search_mode: SearchMode | None = None

    @model_validator(mode="after")
    def normalize_query_fields(self) -> "CatalogSearchIntent":
        self.query = self.query.strip()
        self.queries = [query.strip() for query in self.queries if query.strip()]
        return self


class CatalogSearchPlan(BaseModel):
    should_search: bool
    queries: list[str] = Field(default_factory=list)
    hard_filters: dict[str, Any] = Field(default_factory=dict)
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

    if not queries and not has_image:
        return CatalogSearchPlan(
            should_search=False,
            search_mode=mode,
            top_k=top_k,
            strictness=intent.strictness,
            hard_filters=hard_filters,
            no_search_reason="missing_query_or_image",
        )

    return CatalogSearchPlan(
        should_search=True,
        queries=queries,
        hard_filters=hard_filters,
        strictness=intent.strictness,
        search_mode=mode,
        top_k=top_k,
    )


def _hard_filters(
    intent: CatalogSearchIntent,
    capabilities: CatalogCapabilities,
) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    for name, raw_value in intent.filters.items():
        capability = capabilities.filters.get(name)
        if capability is None:
            continue
        value = _validated_filter_value(raw_value, capability)
        if value not in (None, "", [], {}):
            filters[name] = value

    return filters


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


def _validated_filter_value(
    value: Any,
    capability: CatalogFilterCapability,
) -> Any:
    if capability.type == "number":
        return _number_filter(value)
    if capability.type == "enum":
        return _enum_filter(value, capability.values)
    if capability.type == "text":
        return _text_filter(value)
    return None


def _enum_filter(value: Any, allowed_values: list[str]) -> list[str]:
    if not allowed_values:
        return []
    candidates = _filter_values(value)
    allowed = set(allowed_values)
    return [candidate for candidate in candidates if candidate in allowed]


def _text_filter(value: Any) -> list[str]:
    return _filter_values(value)


def _filter_values(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _number_filter(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}

    lower = _first_number(value, ("min", "gte"))
    upper = _first_number(value, ("max", "lte"))
    if lower is not None and upper is not None and lower > upper:
        return {}

    number_filter: dict[str, float] = {}
    if lower is not None:
        number_filter["min"] = lower
    if upper is not None:
        number_filter["max"] = upper
    return number_filter


def _first_number(values: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in values:
            return _coerce_float(values[key])
    return None


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None
