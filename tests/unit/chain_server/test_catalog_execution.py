# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from chain_server.src.catalog_execution import execute_catalog_search
from chain_server.src.catalog_request import CatalogSearchPlan
from shared.commerce_contracts import ProductSummary, SearchCatalogResult


def test_execute_catalog_search_maps_structured_filters_to_legacy_payload() -> None:
    captured = {}

    def fake_search(request, catalog_retriever_url, timeout_seconds=None):
        captured["request"] = request
        captured["url"] = catalog_retriever_url
        captured["timeout"] = timeout_seconds
        return SearchCatalogResult(
            ok=True,
            products=[ProductSummary(product_id="prod_1", display_name="Work Bag")],
        )

    execution = execute_catalog_search(
        CatalogSearchPlan(
            should_search=True,
            queries=["work bag"],
            hard_filters={"category": ["bag"], "max_price": 60},
            search_mode="text",
            top_k=4,
        ),
        "http://catalog",
        timeout_seconds=5,
        search_fn=fake_search,
    )

    assert execution.result.products[0].display_name == "Work Bag"
    assert captured["url"] == "http://catalog"
    assert captured["timeout"] == 5
    assert captured["request"].queries == ["work bag"]
    assert captured["request"].categories == ["bag"]
    assert captured["request"].filters == {"max_price": 60}
    assert captured["request"].image_base64 == ""


def test_hybrid_search_retries_text_without_image_when_image_has_no_products() -> None:
    requests = []

    def fake_search(request, catalog_retriever_url, timeout_seconds=None):
        requests.append(request)
        if request.image_base64:
            return SearchCatalogResult(ok=True, products=[])
        return SearchCatalogResult(
            ok=True,
            products=[ProductSummary(product_id="prod_2", display_name="Text Bag")],
        )

    execution = execute_catalog_search(
        CatalogSearchPlan(
            should_search=True,
            queries=["work bag"],
            hard_filters={"category": ["bag"]},
            search_mode="hybrid",
            top_k=4,
        ),
        "http://catalog",
        image_base64="data:image/jpeg;base64,abc",
        search_fn=fake_search,
    )

    assert execution.fallback_used is True
    assert len(requests) == 2
    assert requests[0].image_base64 == "data:image/jpeg;base64,abc"
    assert requests[1].image_base64 == ""


def test_no_search_plan_does_not_call_catalog() -> None:
    def fail_search(*args, **kwargs):
        raise AssertionError("catalog should not be called")

    execution = execute_catalog_search(
        CatalogSearchPlan(
            should_search=False,
            no_search_reason="missing_query_or_image",
        ),
        "http://catalog",
        search_fn=fail_search,
    )

    assert execution.result.ok is True
    assert execution.result.products == []
