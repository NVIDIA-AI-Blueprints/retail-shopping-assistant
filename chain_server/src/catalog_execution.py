# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure catalog search execution for structured search plans."""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel

from .catalog_request import CatalogSearchPlan
from .commerce_tools import search_catalog
from shared.commerce_contracts import SearchCatalogInput, SearchCatalogResult


SearchCatalogFn = Callable[..., SearchCatalogResult]


class CatalogSearchExecution(BaseModel):
    result: SearchCatalogResult
    fallback_used: bool = False


def execute_catalog_search(
    plan: CatalogSearchPlan,
    catalog_retriever_url: str,
    *,
    image_base64: str = "",
    timeout_seconds: float | None = None,
    search_fn: SearchCatalogFn = search_catalog,
) -> CatalogSearchExecution:
    if not plan.should_search:
        return CatalogSearchExecution(
            result=SearchCatalogResult(ok=True, products=[])
        )

    request = _request_from_plan(plan, image_base64=image_base64)
    result = search_fn(
        request,
        catalog_retriever_url,
        timeout_seconds=timeout_seconds,
    )
    fallback_used = False

    if (
        plan.search_mode == "hybrid"
        and image_base64
        and plan.semantic_queries
        and result.ok
        and not result.products
    ):
        result = search_fn(
            _request_from_plan(plan, image_base64=""),
            catalog_retriever_url,
            timeout_seconds=timeout_seconds,
        )
        fallback_used = bool(result.products)

    return CatalogSearchExecution(result=result, fallback_used=fallback_used)


def _request_from_plan(
    plan: CatalogSearchPlan,
    *,
    image_base64: str,
) -> SearchCatalogInput:
    hard_filters = dict(plan.hard_filters)
    categories = hard_filters.pop("category", [])
    if not isinstance(categories, list):
        categories = []

    effective_image = image_base64 if plan.search_mode in {"image", "hybrid"} else ""
    query = " ".join(plan.semantic_queries)
    return SearchCatalogInput(
        query=query,
        queries=plan.semantic_queries,
        image_base64=effective_image,
        categories=categories,
        filters=hard_filters,
        top_k=plan.top_k,
    )
