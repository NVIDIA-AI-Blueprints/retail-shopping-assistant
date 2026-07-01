# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import requests

from chain_server.src.commerce_tools import (
    add_cart_item,
    get_cart,
    remove_cart_item,
    search_catalog,
)
from shared.commerce_contracts import (
    AddCartItemInput,
    GetCartInput,
    Money,
    RemoveCartItemInput,
    SearchCatalogInput,
)


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

    def get(self, url, timeout):
        self.calls.append({"url": url, "timeout": timeout})
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


def test_search_catalog_preserves_multiple_query_terms() -> None:
    session = FakeSession(
        FakeResponse(
            {
                "texts": [],
                "ids": [],
                "similarities": [],
                "names": [],
                "images": [],
            }
        )
    )

    search_catalog(
        SearchCatalogInput(queries=["earrings", "necklace"]),
        "http://catalog-retriever:8010",
        session=session,
    )

    assert session.calls[0]["json"]["text"] == ["earrings", "necklace"]


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


def test_get_cart_maps_memory_rows_to_contract_cart() -> None:
    session = FakeSession(
        FakeResponse(
            {
                "user_id": 42,
                "cart": [
                    {"item": "Silk Dress", "amount": 2, "price": 49.99},
                    {"item": "Leather Bag", "amount": 1, "price": 199.0},
                ],
            }
        )
    )

    result = get_cart(
        GetCartInput(user_id="42"),
        "http://memory-retriever:8011",
        session=session,
    )

    assert result.ok is True
    assert session.calls[0]["url"] == "http://memory-retriever:8011/user/42/cart"
    assert result.cart.user_id == "42"
    assert result.cart.lines[0].display_name == "Silk Dress"
    assert result.cart.lines[0].unit_price.amount == 49.99
    assert result.cart.subtotal.amount == 298.98


def test_add_cart_item_posts_memory_payload_and_returns_message() -> None:
    session = FakeSession(FakeResponse({"message": "added 2 Silk Dress"}))

    result = add_cart_item(
        AddCartItemInput(
            user_id="42",
            product_id="prod_123",
            display_name="Silk Dress",
            quantity=2,
            unit_price=Money(amount=49.99),
            idempotency_key="cart-add-1",
        ),
        "http://memory-retriever:8011",
        session=session,
    )

    assert result.ok is True
    assert session.calls[0]["url"] == "http://memory-retriever:8011/user/42/cart/add"
    assert session.calls[0]["json"] == {
        "item": "Silk Dress",
        "amount": 2,
        "price": 49.99,
    }
    assert result.changed_line.product_id == "prod_123"
    assert result.message == "added 2 Silk Dress"


def test_remove_cart_item_posts_memory_payload_and_returns_message() -> None:
    session = FakeSession(FakeResponse({"message": "removed 1 Silk Dress"}))

    result = remove_cart_item(
        RemoveCartItemInput(
            user_id="42",
            cart_line_id="Silk Dress",
            product_id="prod_123",
            display_name="Silk Dress",
            quantity=1,
            idempotency_key="cart-remove-1",
        ),
        "http://memory-retriever:8011",
        session=session,
    )

    assert result.ok is True
    assert session.calls[0]["url"] == "http://memory-retriever:8011/user/42/cart/remove"
    assert session.calls[0]["json"] == {"item": "Silk Dress", "amount": 1}
    assert result.changed_line.product_id == "prod_123"
    assert result.message == "removed 1 Silk Dress"


def test_commerce_tools_do_not_import_current_agents() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    source = (repo_root / "chain_server/src/commerce_tools.py").read_text()

    forbidden_references = [
        "PlannerAgent",
        "RetrieverAgent",
        "CartAgent",
        "ChatterAgent",
        "SummaryAgent",
        "create_graph",
        "from .planner",
        "from .retriever",
        "from .cart",
        "from .chatter",
        "from .summarizer",
        "from .graph",
    ]

    for reference in forbidden_references:
        assert reference not in source
