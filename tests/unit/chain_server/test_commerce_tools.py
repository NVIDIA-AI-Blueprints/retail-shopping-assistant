# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import requests

from chain_server.src.commerce_tools import search_catalog
from shared.commerce_contracts import SearchCatalogInput


class FakeSession:
    def __init__(self, response=None, exc: Exception | None = None) -> None:
        self.response = response
        self.exc = exc
        self.calls = []

    def post(self, url, json, timeout):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        if self.exc:
            raise self.exc
        return self.response


class FakeResponse:
    def __init__(self, payload, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def json(self):
        return self.payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def test_search_catalog_posts_stateless_text_payload() -> None:
    session = FakeSession(
        FakeResponse(
            {
                "texts": [
                    "Classic Black Patent Leather Purse | Formal bag | bag,purse\nPRICE: 89.99"
                ],
                "ids": ["prod_123"],
                "similarities": [0.91],
                "names": ["Classic Black Patent Leather Purse"],
                "images": ["/images/purse.jpg"],
            }
        )
    )

    result = search_catalog(
        SearchCatalogInput(
            query="black purse",
            categories=["bag"],
            filters={"max_price": 100},
            top_k=3,
        ),
        "http://catalog-retriever:8010",
        session=session,
    )

    assert result.ok is True
    assert session.calls[0]["url"] == "http://catalog-retriever:8010/query/text"
    assert session.calls[0]["json"] == {
        "text": ["black purse"],
        "categories": ["bag"],
        "filters": {"max_price": 100},
        "k": 3,
    }
    assert "user_id" not in session.calls[0]["json"]
    assert "cart" not in session.calls[0]["json"]
    assert "context" not in session.calls[0]["json"]
    assert result.products[0].product_id == "prod_123"
    assert result.products[0].price.amount == 89.99
    assert result.products[0].category == "bag"


def test_search_catalog_posts_image_payload_when_image_is_present() -> None:
    session = FakeSession(
        FakeResponse(
            {
                "texts": ["Red sunglasses | Oval frames | sunglasses,accessory\nPRICE: 49"],
                "ids": ["prod_456"],
                "similarities": [0.88],
                "names": ["Red Oval Sunglasses"],
                "images": ["/images/sunglasses.jpg"],
            }
        )
    )

    result = search_catalog(
        SearchCatalogInput(
            query="similar under $60",
            image_base64="data:image/jpeg;base64,abc",
            categories=["sunglasses"],
            filters={"max_price": 60},
        ),
        "http://catalog-retriever:8010/",
        session=session,
    )

    assert result.ok is True
    assert session.calls[0]["url"] == "http://catalog-retriever:8010/query/image"
    assert session.calls[0]["json"]["image_base64"] == "data:image/jpeg;base64,abc"
    assert result.products[0].product_id == "prod_456"


def test_search_catalog_rejects_empty_query_without_image() -> None:
    session = FakeSession(FakeResponse({}))

    result = search_catalog(
        SearchCatalogInput(query=""),
        "http://catalog-retriever:8010",
        session=session,
    )

    assert result.ok is False
    assert result.error.code == "invalid_search_request"
    assert session.calls == []


def test_search_catalog_returns_structured_request_error() -> None:
    result = search_catalog(
        SearchCatalogInput(query="black purse"),
        "http://catalog-retriever:8010",
        session=FakeSession(exc=requests.Timeout("timed out")),
    )

    assert result.ok is False
    assert result.error.code == "catalog_request_failed"
    assert result.error.retryable is True


def test_commerce_tools_do_not_import_current_agents() -> None:
    source = Path("chain_server/src/commerce_tools.py").read_text()

    forbidden_references = [
        "PlannerAgent",
        "RetrieverAgent",
        "CartAgent",
        "ChatterAgent",
        "SummaryAgent",
        "create_graph",
        ".planner",
        ".retriever",
        ".cart",
        ".chatter",
        ".summarizer",
        ".graph",
    ]

    for reference in forbidden_references:
        assert reference not in source
