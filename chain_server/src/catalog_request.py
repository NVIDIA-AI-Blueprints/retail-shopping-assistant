# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build catalog search plans from structured agent intent.

This module does not parse shopper language. The agent or another language
understanding layer supplies structured intent; this builder validates that
intent against catalog-owned capabilities.
"""

from __future__ import annotations

from math import isfinite
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from .catalog_capabilities import effective_filter_capabilities
from shared.commerce_contracts import CatalogCapabilities, CatalogFilterCapability


SearchMode = Literal["text", "image", "hybrid"]


class CatalogSearchIntent(BaseModel):
    semantic_query: str = ""
    semantic_queries: list[str] = Field(default_factory=list)
    required_constraints: dict[str, Any] = Field(default_factory=dict)
    search_mode: SearchMode | None = None

    @model_validator(mode="after")
    def normalize_query_fields(self) -> "CatalogSearchIntent":
        self.semantic_query = self.semantic_query.strip()
        self.semantic_queries = [
            query.strip() for query in self.semantic_queries if query.strip()
        ]
        return self


class CatalogSearchPlan(BaseModel):
    should_search: bool
    semantic_queries: list[str] = Field(default_factory=list)
    hard_filters: dict[str, Any] = Field(default_factory=dict)
    search_mode: SearchMode = "text"
    top_k: int = 4
    no_search_reason: str | None = None
    constraint_issues: list[str] = Field(default_factory=list)
    #: Enum requirements honoured in part: the advertised values filtered,
    #: the rest could not and are disclosed rather than silently dropped.
    partial_constraints: list[str] = Field(default_factory=list)


def build_catalog_search_plan(
    intent: CatalogSearchIntent,
    capabilities: CatalogCapabilities,
    *,
    has_image: bool = False,
    top_k: int = 4,
) -> CatalogSearchPlan:
    semantic_queries = intent.semantic_queries or (
        [intent.semantic_query] if intent.semantic_query else []
    )
    mode = _search_mode(intent.search_mode, capabilities, has_image=has_image)
    hard_filters, constraint_issues, partial_constraints = _hard_filters(
        intent, capabilities
    )

    if constraint_issues:
        return CatalogSearchPlan(
            should_search=False,
            semantic_queries=semantic_queries,
            search_mode=mode,
            top_k=top_k,
            hard_filters=hard_filters,
            no_search_reason="unsupported_required_constraint",
            constraint_issues=constraint_issues,
            partial_constraints=partial_constraints,
        )

    mode_issue = _search_mode_issue(
        intent.search_mode,
        capabilities,
        has_image=has_image,
    )
    if mode_issue:
        return CatalogSearchPlan(
            should_search=False,
            semantic_queries=semantic_queries,
            search_mode=mode,
            top_k=top_k,
            hard_filters=hard_filters,
            no_search_reason=mode_issue,
        )

    if not semantic_queries and has_image and mode == "text":
        return CatalogSearchPlan(
            should_search=False,
            search_mode=mode,
            top_k=top_k,
            hard_filters=hard_filters,
            no_search_reason="image_search_unavailable",
            constraint_issues=constraint_issues,
            partial_constraints=partial_constraints,
        )

    if not semantic_queries and not has_image:
        return CatalogSearchPlan(
            should_search=False,
            search_mode=mode,
            top_k=top_k,
            hard_filters=hard_filters,
            no_search_reason="missing_query_or_image",
            constraint_issues=constraint_issues,
            partial_constraints=partial_constraints,
        )

    return CatalogSearchPlan(
        should_search=True,
        semantic_queries=semantic_queries,
        hard_filters=hard_filters,
        search_mode=mode,
        top_k=top_k,
        constraint_issues=constraint_issues,
        partial_constraints=partial_constraints,
    )


def _hard_filters(
    intent: CatalogSearchIntent,
    capabilities: CatalogCapabilities,
) -> tuple[dict[str, Any], list[str], list[str]]:
    filters: dict[str, Any] = {}
    issues: list[str] = []
    partial: list[str] = []
    available = effective_filter_capabilities(capabilities)
    for name, raw_value in intent.required_constraints.items():
        capability = available.get(name)
        if capability is None:
            issues.append(f"'{name}' is not an advertised hard filter")
            continue
        value = _validated_filter_value(raw_value, capability)
        if value not in (None, "", [], {}):
            filters[name] = value
        if not _filter_value_is_fully_valid(raw_value, value, capability):
            # An enum that keeps at least one advertised value is honourable in
            # part: filter on what is advertised and disclose the rest. This is
            # not the weakening the other branches guard against -- nothing the
            # shopper can act on is dropped, and the unadvertised words were
            # never enforceable in the first place.
            #
            # Everything else still stops. A malformed numeric bound must never
            # become "no bound": "under $300" that quietly returns $400 items is
            # worse than refusing.
            if capability.type == "enum" and value:
                partial.append(
                    f"'{name}' also had values this catalog does not carry"
                )
            else:
                issues.append(
                    f"'{name}' contains an unsupported value or operator"
                )

    return filters, issues, partial


def _search_mode(
    requested: SearchMode | None,
    capabilities: CatalogCapabilities,
    *,
    has_image: bool,
) -> SearchMode:
    supported = set(capabilities.retrieval_modes)
    if requested is not None:
        return requested
    if has_image and "hybrid" in supported:
        return "hybrid"
    if has_image and "image" in supported:
        return "image"
    return "text"


def _search_mode_issue(
    requested: SearchMode | None,
    capabilities: CatalogCapabilities,
    *,
    has_image: bool,
) -> str | None:
    if requested is None:
        return None
    if requested not in set(capabilities.retrieval_modes):
        return "unsupported_search_mode"
    if requested in {"image", "hybrid"} and not has_image:
        return "missing_image_for_search_mode"
    return None


def _validated_filter_value(
    value: Any,
    capability: CatalogFilterCapability,
) -> Any:
    if capability.type == "number":
        return _number_filter(value)
    if capability.type in {"enum", "enum_list"}:
        return _enum_filter(value, capability.values)
    if capability.type == "text":
        return _text_filter(value)
    return None


def _filter_value_is_fully_valid(
    raw_value: Any,
    validated_value: Any,
    capability: CatalogFilterCapability,
) -> bool:
    if capability.type in {"enum", "enum_list"}:
        if "in" not in capability.operators:
            return False
        candidates = _filter_values(raw_value)
        return bool(candidates) and len(candidates) == len(validated_value or [])
    if capability.type == "number":
        if not isinstance(raw_value, dict) or not validated_value:
            return False
        if set(raw_value) - {"min", "max", "gte", "lte"}:
            return False
        if _has_duplicate_number_aliases(raw_value):
            return False
        lower = _first_number(raw_value, ("min", "gte"))
        upper = _first_number(raw_value, ("max", "lte"))
        if lower is not None and "gte" not in capability.operators:
            return False
        if upper is not None and "lte" not in capability.operators:
            return False
        if lower is None and any(key in raw_value for key in ("min", "gte")):
            return False
        if upper is None and any(key in raw_value for key in ("max", "lte")):
            return False
        return not (lower is not None and upper is not None and lower > upper)
    if capability.type == "text":
        return bool(validated_value)
    return False


def _enum_filter(value: Any, allowed_values: list[str]) -> list[str]:
    if not allowed_values:
        return []
    candidates = _filter_values(value)
    allowed = {item.casefold(): item for item in allowed_values}
    return [
        allowed[candidate.casefold()]
        for candidate in candidates
        if candidate.casefold() in allowed
    ]


def _text_filter(value: Any) -> list[str]:
    return _filter_values(value)


def _filter_values(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _number_filter(value: Any) -> dict[str, float]:
    if not isinstance(value, dict) or _has_duplicate_number_aliases(value):
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


def _has_duplicate_number_aliases(value: dict[str, Any]) -> bool:
    return ("min" in value and "gte" in value) or (
        "max" in value and "lte" in value
    )


def _first_number(values: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in values:
            return _coerce_float(values[key])
    return None


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    return number if isfinite(number) else None
