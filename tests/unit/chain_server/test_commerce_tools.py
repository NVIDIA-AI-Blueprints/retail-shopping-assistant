# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
import requests

from chain_server.src import commerce_tools as commerce_tools_mod
from chain_server.src.commerce_tools import (
    add_cart_item,
    check_product_availability,
    get_cart,
    get_product_details,
    get_store_policy,
    remove_cart_item,
    search_catalog,
    update_cart_item,
)
from shared.commerce_contracts import (
    AddCartItemInput,
    Cart,
    CartLine,
    CartMutationResult,
    CheckProductAvailabilityInput,
    GetCartInput,
    GetCartResult,
    GetProductDetailsInput,
    GetStorePolicyInput,
    Money,
    RemoveCartItemInput,
    SearchCatalogInput,
    UpdateCartItemInput,
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
    assert session.calls[0]["timeout"] is None
    assert result.products[0].product_id == "prod_123"
    assert result.products[0].price.amount == 89.99
    assert result.products[0].category == "bag"


def test_search_catalog_prefers_structured_products_when_returned() -> None:
    session = FakeSession(
        FakeResponse(
            {
                "products": [
                    {
                        "product_id": "prod_1",
                        "display_name": "Work Bag",
                        "description": "structured tote",
                        "category": "bag",
                        "price": {"amount": 59.0, "currency": "USD"},
                        "image_url": "/images/work_bag.jpg",
                        "attributes": {"similarity": 0.91},
                    }
                ],
                "diagnostics": {"returned_count": 1},
                "no_result_reason": None,
            }
        )
    )

    result = search_catalog(
        SearchCatalogInput(query="work bag"),
        "http://catalog-retriever:8010",
        session=session,
    )

    assert result.ok is True
    assert result.products[0].display_name == "Work Bag"
    assert result.products[0].price.amount == 59.0
    assert result.diagnostics == {"returned_count": 1}


def test_search_catalog_uses_configured_timeout_when_provided() -> None:
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
        SearchCatalogInput(query="black purse"),
        "http://catalog-retriever:8010",
        timeout_seconds=120,
        session=session,
    )

    assert session.calls[0]["timeout"] == 120


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


def test_search_catalog_preserves_non_retryable_filter_rejection() -> None:
    result = search_catalog(
        SearchCatalogInput(
            query="blue item",
            filters={"unknown": ["value"]},
        ),
        "http://catalog-retriever:8010",
        session=FakeSession(
            FakeResponse(
                {"detail": "Unsupported catalog filter(s): unknown"},
                status_code=422,
            )
        ),
    )

    assert result.ok is False
    assert result.error.code == "catalog_filter_rejected"
    assert result.error.retryable is False
    assert result.error.message == "Unsupported catalog filter(s): unknown"


def test_get_product_details_reads_url_encoded_product_id() -> None:
    session = FakeSession(
        FakeResponse(
            {
                "product_id": "generated:abc123",
                "display_name": "Travel Pant",
                "description": "A soft pant.",
                "category": "pants",
                "price": {"amount": 49.9, "currency": "USD"},
                "image_url": "/images/pant.jpg",
                "attributes": {
                    "care": "Machine wash cold.",
                    "composition": "cotton",
                },
                "variants": [],
            }
        )
    )

    result = get_product_details(
        GetProductDetailsInput(product_id="generated:abc123"),
        "http://catalog-retriever:8010/",
        timeout_seconds=3,
        session=session,
    )

    assert result.ok is True
    assert result.product.attributes["care"] == "Machine wash cold."
    assert session.calls == [
        {
            "url": "http://catalog-retriever:8010/products/generated%3Aabc123",
            "timeout": 3,
        }
    ]


def test_get_product_details_maps_missing_product_to_non_retryable_error() -> None:
    result = get_product_details(
        GetProductDetailsInput(product_id="missing"),
        "http://catalog",
        session=FakeSession(FakeResponse({"detail": "not found"}, status_code=404)),
    )

    assert result.ok is False
    assert result.error.code == "product_not_found"
    assert result.error.retryable is False


def test_get_product_details_rejects_mismatched_response_id() -> None:
    result = get_product_details(
        GetProductDetailsInput(product_id="requested"),
        "http://catalog",
        session=FakeSession(
            FakeResponse(
                {
                    "product_id": "different",
                    "display_name": "Different",
                    "attributes": {},
                    "variants": [],
                }
            )
        ),
    )

    assert result.ok is False
    assert result.error.code == "catalog_response_invalid"


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
    assert result.meta.idempotency_key == "cart-add-1"


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
    assert result.meta.idempotency_key == "cart-remove-1"


def test_update_cart_item_replaces_existing_quantity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    cart = Cart(
        user_id="42",
        lines=[
            CartLine(
                cart_line_id="Silk Dress",
                product_id="prod_123",
                display_name="Silk Dress",
                quantity=2,
                unit_price=Money(amount=49.99),
                image_url="/images/silk-dress.jpg",
            )
        ],
    )

    def fake_get_cart(request, memory_retriever_url, **kwargs):
        calls.append(("get", request, memory_retriever_url, kwargs))
        return GetCartResult(ok=True, cart=cart)

    def fake_remove_cart_item(request, memory_retriever_url, **kwargs):
        calls.append(("remove", request, memory_retriever_url, kwargs))
        return CartMutationResult(ok=True)

    def fake_add_cart_item(request, memory_retriever_url, **kwargs):
        calls.append(("add", request, memory_retriever_url, kwargs))
        return CartMutationResult(ok=True, changed_line=cart.lines[0])

    monkeypatch.setattr(commerce_tools_mod, "get_cart", fake_get_cart)
    monkeypatch.setattr(
        commerce_tools_mod,
        "remove_cart_item",
        fake_remove_cart_item,
    )
    monkeypatch.setattr(commerce_tools_mod, "add_cart_item", fake_add_cart_item)

    session = object()
    result = update_cart_item(
        UpdateCartItemInput(
            user_id="42",
            cart_line_id="Silk Dress",
            quantity=4,
            idempotency_key="cart-update-1",
        ),
        "http://memory-retriever:8011",
        timeout_seconds=7,
        session=session,
    )

    assert result.ok is True
    assert [call[0] for call in calls] == ["get", "remove", "add"]
    remove_request = calls[1][1]
    assert remove_request.quantity == 2
    assert remove_request.product_id == "prod_123"
    assert remove_request.display_name == "Silk Dress"
    assert remove_request.idempotency_key == "cart-update-1-remove"
    add_request = calls[2][1]
    assert add_request.quantity == 4
    assert add_request.product_id == "prod_123"
    assert add_request.display_name == "Silk Dress"
    assert add_request.unit_price.amount == 49.99
    assert add_request.image_url == "/images/silk-dress.jpg"
    assert add_request.idempotency_key == "cart-update-1-add"
    assert all(call[3] == {"timeout_seconds": 7, "session": session} for call in calls)


def test_update_cart_item_quantity_zero_removes_complete_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    removed = []
    cart = Cart(
        user_id="42",
        lines=[
            CartLine(
                cart_line_id="Silk Dress",
                product_id="prod_123",
                display_name="Silk Dress",
                quantity=3,
            )
        ],
    )

    monkeypatch.setattr(
        commerce_tools_mod,
        "get_cart",
        lambda *args, **kwargs: GetCartResult(ok=True, cart=cart),
    )

    def fake_remove_cart_item(request, *args, **kwargs):
        removed.append(request)
        return CartMutationResult(ok=True)

    monkeypatch.setattr(
        commerce_tools_mod,
        "remove_cart_item",
        fake_remove_cart_item,
    )
    monkeypatch.setattr(
        commerce_tools_mod,
        "add_cart_item",
        lambda *args, **kwargs: pytest.fail("quantity zero must not re-add the line"),
    )

    result = update_cart_item(
        UpdateCartItemInput(
            user_id="42",
            cart_line_id="Silk Dress",
            quantity=0,
            idempotency_key="cart-update-remove",
        ),
        "http://memory-retriever:8011",
    )

    assert result.ok is True
    assert result.changed_line is None
    assert len(removed) == 1
    assert removed[0].quantity == 3
    assert removed[0].idempotency_key == "cart-update-remove"


def test_update_cart_item_returns_structured_error_for_unknown_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cart = Cart(
        user_id="42",
        lines=[
            CartLine(
                cart_line_id="Silk Dress",
                product_id="prod_123",
                display_name="Silk Dress",
                quantity=1,
            )
        ],
    )
    monkeypatch.setattr(
        commerce_tools_mod,
        "get_cart",
        lambda *args, **kwargs: GetCartResult(ok=True, cart=cart),
    )
    monkeypatch.setattr(
        commerce_tools_mod,
        "remove_cart_item",
        lambda *args, **kwargs: pytest.fail("unknown lines must not be removed"),
    )
    monkeypatch.setattr(
        commerce_tools_mod,
        "add_cart_item",
        lambda *args, **kwargs: pytest.fail("unknown lines must not be added"),
    )

    result = update_cart_item(
        UpdateCartItemInput(
            user_id="42",
            cart_line_id="Unknown Line",
            quantity=2,
            idempotency_key="cart-update-missing",
        ),
        "http://memory-retriever:8011",
    )

    assert result.ok is False
    assert result.error.code == "cart_line_not_found"
    assert "Call get_cart_tool" in result.error.message
    assert result.meta.idempotency_key == "cart-update-missing"


def test_get_store_policy_loads_all_configured_topics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policies_path = (
        Path(__file__).resolve().parents[3]
        / "chain_server"
        / "skills"
        / "shopper"
        / "store-policy"
        / "policies.yaml"
    )
    monkeypatch.setattr(commerce_tools_mod, "_POLICY_CACHE", None)

    for topic in (
        "returns",
        "shipping",
        "sizing",
        "payment",
        "price_match",
        "gift_cards",
    ):
        result = get_store_policy(GetStorePolicyInput(topic=topic), policies_path)

        assert result.ok is True
        assert result.policy.topic == topic
        assert result.policy.title
        assert result.policy.body
        assert "last_updated" not in type(result.policy).model_fields


def test_get_store_policy_returns_error_for_unknown_topic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policies_path = (
        Path(__file__).resolve().parents[3]
        / "chain_server"
        / "skills"
        / "shopper"
        / "store-policy"
        / "policies.yaml"
    )
    monkeypatch.setattr(commerce_tools_mod, "_POLICY_CACHE", None)

    result = get_store_policy(
        GetStorePolicyInput(topic="international_shipping"),
        policies_path,
    )

    assert result.ok is False
    assert result.error.code == "policy_topic_not_found"
    assert "retailer's help center" in result.error.message


def test_get_store_policy_returns_error_when_file_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(commerce_tools_mod, "_POLICY_CACHE", None)

    result = get_store_policy(
        GetStorePolicyInput(topic="returns"),
        tmp_path / "missing-policies.yaml",
    )

    assert result.ok is False
    assert result.error.code == "policy_load_failed"
    assert result.error.message == "Policy file could not be loaded."


def test_check_product_availability_returns_consistent_unknown_result() -> None:
    result = check_product_availability(
        CheckProductAvailabilityInput(
            product_ref="prod_123",
            variant_hint="blue, size medium",
        )
    )

    assert result.ok is True
    assert result.product_ref == "prod_123"
    assert result.availability == "unknown"
    assert result.message == (
        "Real-time inventory is not available through the assistant. "
        "Availability and size can be confirmed on the product page or at checkout."
    )


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
