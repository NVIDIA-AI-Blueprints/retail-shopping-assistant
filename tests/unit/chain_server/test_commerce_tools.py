# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import requests

from chain_server.src import commerce_tools as commerce_tools_mod
from chain_server.src.commerce_tools import (
    add_cart_item,
    check_active_promotions,
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
    CheckActivePromotionsResult,
    CheckProductAvailabilityInput,
    GetCartInput,
    GetProductDetailsInput,
    GetStorePolicyInput,
    Money,
    ProductSummary,
    RemoveCartItemInput,
    SearchCatalogInput,
    UpdateCartItemInput,
)


class FakeSession:
    def __init__(self, response=None, exc: Exception | None = None) -> None:
        self.response = response
        self.exc = exc
        self.calls = []
        self.put_calls = []

    def post(self, url, json, timeout):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        if self.exc:
            raise self.exc
        return self.response

    def put(self, url, json, timeout):
        call = {"url": url, "json": json, "timeout": timeout}
        self.calls.append(call)
        self.put_calls.append(call)
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
                    {
                        "cart_line_id": "17",
                        "item": "Silk Dress",
                        "amount": 2,
                        "price": 49.99,
                    },
                    {
                        "cart_line_id": "23",
                        "item": "Leather Bag",
                        "amount": 1,
                        "price": 199.0,
                    },
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
    assert result.cart.lines[0].cart_line_id == "17"
    assert result.cart.lines[0].display_name == "Silk Dress"
    assert result.cart.lines[0].unit_price.amount == 49.99
    assert result.cart.subtotal.amount == 298.98


def test_add_cart_item_posts_memory_payload_and_returns_message() -> None:
    session = FakeSession(
        FakeResponse(
            {
                "cart_line": {
                    "cart_line_id": "line-17",
                    "product_id": "prod_123",
                    "item": "Silk Dress",
                    "amount": 2,
                    "price": 49.99,
                },
                "message": "added 2 Silk Dress",
            }
        )
    )

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
        "product_id": "prod_123",
        "item": "Silk Dress",
        "amount": 2,
        "price": 49.99,
        "idempotency_key": "cart-add-1",
    }
    assert result.changed_line.cart_line_id == "line-17"
    assert result.changed_line.product_id == "prod_123"
    assert result.message == "added 2 Silk Dress"
    assert result.meta.idempotency_key == "cart-add-1"


def test_remove_cart_item_posts_memory_payload_and_returns_message() -> None:
    session = FakeSession(
        FakeResponse(
            {
                "cart_line": {
                    "cart_line_id": "line-17",
                    "product_id": "prod_123",
                    "item": "Silk Dress",
                    "amount": 1,
                    "price": 49.99,
                },
                "message": "removed 1 Silk Dress",
            }
        )
    )

    result = remove_cart_item(
        RemoveCartItemInput(
            user_id="42",
            cart_line_id="line-17",
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
    assert session.calls[0]["json"] == {
        "cart_line_id": "line-17",
        "amount": 1,
        "idempotency_key": "cart-remove-1",
    }
    assert result.changed_line.cart_line_id == "line-17"
    assert result.changed_line.product_id == "prod_123"
    assert result.message == "removed 1 Silk Dress"
    assert result.meta.idempotency_key == "cart-remove-1"


def test_update_cart_item_sets_quantity_in_one_request() -> None:
    session = FakeSession(
        FakeResponse(
            {
                "cart_line": {
                    "cart_line_id": "17",
                    "item": "Silk Dress",
                    "amount": 4,
                    "price": 49.99,
                },
                "message": "Updated 'Silk Dress' to quantity 4.",
            }
        )
    )

    result = update_cart_item(
        UpdateCartItemInput(
            user_id="42",
            cart_line_id="17",
            quantity=4,
            idempotency_key="cart-update-1",
        ),
        "http://memory-retriever:8011",
        timeout_seconds=7,
        session=session,
    )

    assert result.ok is True
    assert session.calls == [
        {
            "url": "http://memory-retriever:8011/user/42/cart/17/quantity",
            "json": {
                "quantity": 4,
                "idempotency_key": "cart-update-1",
            },
            "timeout": 7,
        }
    ]
    assert session.put_calls == session.calls
    assert result.changed_line.cart_line_id == "17"
    assert result.changed_line.display_name == "Silk Dress"
    assert result.changed_line.quantity == 4
    assert result.changed_line.unit_price.amount == 49.99
    assert result.message == "Updated 'Silk Dress' to quantity 4."
    assert result.meta.idempotency_key == "cart-update-1"


def test_update_cart_item_quantity_zero_uses_same_endpoint() -> None:
    session = FakeSession(
        FakeResponse(
            {
                "cart_line": {
                    "cart_line_id": "17",
                    "item": "Silk Dress",
                    "amount": 0,
                    "price": 49.99,
                },
                "message": "Updated 'Silk Dress' to quantity 0.",
            }
        )
    )

    result = update_cart_item(
        UpdateCartItemInput(
            user_id="42",
            cart_line_id="17",
            quantity=0,
            idempotency_key="cart-update-remove",
        ),
        "http://memory-retriever:8011",
        session=session,
    )

    assert result.ok is True
    assert result.changed_line is None
    assert session.calls == [
        {
            "url": "http://memory-retriever:8011/user/42/cart/17/quantity",
            "json": {
                "quantity": 0,
                "idempotency_key": "cart-update-remove",
            },
            "timeout": 10,
        }
    ]
    assert session.put_calls == session.calls
    assert result.meta.idempotency_key == "cart-update-remove"


def test_update_cart_item_rejects_mismatched_zero_quantity_response() -> None:
    result = update_cart_item(
        UpdateCartItemInput(
            user_id="42",
            cart_line_id="17",
            quantity=0,
            idempotency_key="cart-update-remove",
        ),
        "http://memory-retriever:8011",
        session=FakeSession(
            FakeResponse(
                {
                    "cart_line": {
                        "cart_line_id": "17",
                        "item": "Silk Dress",
                        "amount": 2,
                        "price": 49.99,
                    }
                }
            )
        ),
    )

    assert result.ok is False
    assert result.error.code == "cart_response_invalid"


def test_update_cart_item_returns_structured_error_for_unknown_line() -> None:
    result = update_cart_item(
        UpdateCartItemInput(
            user_id="42",
            cart_line_id="999",
            quantity=2,
            idempotency_key="cart-update-missing",
        ),
        "http://memory-retriever:8011",
        session=FakeSession(FakeResponse({}, status_code=404)),
    )

    assert result.ok is False
    assert result.error.code == "cart_line_not_found"
    assert "Call get_cart_tool" in result.error.message
    assert result.meta.idempotency_key == "cart-update-missing"


def test_update_cart_item_returns_retryable_error_on_request_failure() -> None:
    result = update_cart_item(
        UpdateCartItemInput(
            user_id="42",
            cart_line_id="17",
            quantity=4,
            idempotency_key="cart-update-failed",
        ),
        "http://memory-retriever:8011",
        session=FakeSession(exc=requests.Timeout("timed out")),
    )

    assert result.ok is False
    assert result.error.code == "cart_update_failed"
    assert result.error.retryable is True
    assert result.meta.idempotency_key == "cart-update-failed"


def test_update_cart_item_returns_nonretryable_error_for_client_failure() -> None:
    result = update_cart_item(
        UpdateCartItemInput(
            user_id="42",
            cart_line_id="17",
            quantity=4,
            idempotency_key="cart-update-rejected",
        ),
        "http://memory-retriever:8011",
        session=FakeSession(FakeResponse({}, status_code=422)),
    )

    assert result.ok is False
    assert result.error.code == "cart_update_failed"
    assert result.error.message == "Cart quantity update was rejected."
    assert result.error.retryable is False
    assert result.error.details == {"status_code": 422}


def test_update_cart_item_returns_retryable_error_for_server_failure() -> None:
    result = update_cart_item(
        UpdateCartItemInput(
            user_id="42",
            cart_line_id="17",
            quantity=4,
            idempotency_key="cart-update-server-failed",
        ),
        "http://memory-retriever:8011",
        session=FakeSession(FakeResponse({}, status_code=503)),
    )

    assert result.ok is False
    assert result.error.code == "cart_update_failed"
    assert result.error.retryable is True


def _write_configured_policies(tmp_path: Path) -> Path:
    topics = (
        "returns",
        "shipping",
        "sizing",
        "payment",
        "price_match",
        "gift_cards",
    )
    lines = ["configured: true", "policies:"]
    for topic in topics:
        lines.extend(
            (
                f"  {topic}:",
                f'    title: "Configured {topic} policy"',
                f'    body: "Configured content for {topic}."',
            )
        )
    policies_path = tmp_path / "policies.yaml"
    policies_path.write_text("\n".join(lines), encoding="utf-8")
    return policies_path


def test_get_store_policy_loads_all_configured_topics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policies_path = _write_configured_policies(tmp_path)
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


def test_get_store_policy_fails_closed_for_bundled_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policies_path = (
        Path(__file__).resolve().parents[3]
        / "shared"
        / "configs"
        / "chain_server"
        / "store_policies.yaml"
    )
    monkeypatch.setattr(commerce_tools_mod, "_POLICY_CACHE", None)

    result = get_store_policy(GetStorePolicyInput(topic="returns"), policies_path)

    assert result.ok is False
    assert result.policy is None
    assert result.error.code == "policy_not_configured"
    assert "retailer's help center" in result.error.message


def test_get_store_policy_rejects_enabled_operator_placeholder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policies_path = tmp_path / "store_policies.yaml"
    policies_path.write_text(
        "\n".join(
            (
                "configured: true",
                "policies:",
                "  returns:",
                '    title: "Return Policy"',
                '    body: "[Operator placeholder] Replace this policy."',
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(commerce_tools_mod, "_POLICY_CACHE", None)

    result = get_store_policy(GetStorePolicyInput(topic="returns"), policies_path)

    assert result.ok is False
    assert result.policy is None
    assert result.error.code == "policy_load_failed"
    assert "operator placeholders" in result.error.details["error"]


def test_get_store_policy_returns_error_for_unknown_topic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policies_path = _write_configured_policies(tmp_path)
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


def test_get_store_policy_cache_does_not_reload_at_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policies_path = _write_configured_policies(tmp_path)
    monkeypatch.setattr(commerce_tools_mod, "_POLICY_CACHE", None)

    first = get_store_policy(GetStorePolicyInput(topic="returns"), policies_path)
    policies_path.write_text("configured: false\npolicies: {}\n", encoding="utf-8")
    second = get_store_policy(GetStorePolicyInput(topic="returns"), policies_path)

    assert first.ok is True
    assert second == first


def test_check_product_availability_reports_general_availability() -> None:
    result = check_product_availability(
        CheckProductAvailabilityInput(product_ref="prod_123"),
        ProductSummary(product_id="prod_123", display_name="Everyday Dress"),
    )

    assert result.ok is True
    assert result.product_ref == "prod_123"
    assert result.availability == "in_stock"
    assert result.message == "Yes, Everyday Dress is available."


def test_check_active_promotions_reports_no_sales_without_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_io(*_args, **_kwargs):
        raise AssertionError("promotion stub must not perform external I/O")

    monkeypatch.setattr(commerce_tools_mod.requests, "get", fail_io)
    monkeypatch.setattr(commerce_tools_mod.requests, "post", fail_io)
    monkeypatch.setattr(commerce_tools_mod.requests, "Session", fail_io)
    monkeypatch.setattr(commerce_tools_mod, "_catalog_session", fail_io)

    result = check_active_promotions()

    assert isinstance(result, CheckActivePromotionsResult)
    assert result.ok is True
    assert result.active is False
    assert result.message == (
        "No active sale or promotion is available through the assistant right now."
    )


@pytest.mark.parametrize(
    ("category", "taxonomy", "display_name", "variant_hint"),
    [
        (
            "dresses",
            {"category": "apparel"},
            "Everyday Dress",
            "size medium",
        ),
        ("footwear", {}, "Everyday Flat", "size 8"),
    ],
)
def test_check_product_availability_reports_sized_category_availability(
    category: str,
    taxonomy: dict[str, str],
    display_name: str,
    variant_hint: str,
) -> None:
    result = check_product_availability(
        CheckProductAvailabilityInput(
            product_ref="prod_123",
            variant_hint=variant_hint,
        ),
        ProductSummary(
            product_id="prod_123",
            display_name=display_name,
            category=category,
            attributes={"taxonomy": taxonomy},
        ),
    )

    assert result.ok is True
    assert result.availability == "in_stock"
    assert result.message == f"Yes, {display_name} is available in {variant_hint}."


def test_check_product_availability_reports_one_size_for_other_categories() -> None:
    result = check_product_availability(
        CheckProductAvailabilityInput(
            product_ref="prod_watch",
            variant_hint="size small",
        ),
        ProductSummary(
            product_id="prod_watch",
            display_name="Classic Watch",
            category="watches",
            attributes={"taxonomy": {"category": "accessories"}},
        ),
    )

    assert result.ok is True
    assert result.availability == "in_stock"
    assert result.message == (
        "Classic Watch is one-size-fits-all and is available."
    )


LEGACY_AGENT_MODULES = frozenset(
    {
        "graph",
        "planner",
        "retriever",
        "cart",
        "chatter",
        "summarizer",
        "functions",
    }
)

LEGACY_AGENT_SYMBOLS = frozenset(
    {
        "PlannerAgent",
        "RetrieverAgent",
        "CartAgent",
        "ChatterAgent",
        "SummaryAgent",
        "create_graph",
    }
)


def _chain_server_src() -> Path:
    return Path(__file__).resolve().parents[3] / "chain_server/src"


def _chain_server_imports(tree: ast.AST) -> set[str]:
    """Return chain-server module names imported by one parsed source file.

    Only sibling-relative imports and absolute ``chain_server.src.<module>``
    imports are considered, so third-party packages cannot collide with a
    legacy module name.
    """

    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level >= 1:
                if node.module:
                    # from .cart import X
                    modules.add(node.module.split(".")[0])
                else:
                    # from . import cart
                    modules.update(alias.name for alias in node.names)
            elif node.module:
                parts = node.module.split(".")
                if parts[:2] == ["chain_server", "src"]:
                    if len(parts) > 2:
                        # from chain_server.src.cart import X
                        modules.add(parts[2])
                    else:
                        # from chain_server.src import cart
                        modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[:2] == ["chain_server", "src"] and len(parts) > 2:
                    modules.add(parts[2])
    return modules


def _referenced_symbols(tree: ast.AST) -> set[str]:
    """Return every identifier named by one parsed source file."""

    symbols: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            symbols.add(node.id)
        elif isinstance(node, ast.Attribute):
            symbols.add(node.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                symbols.add(alias.asname or alias.name)
    return symbols


def test_legacy_agent_modules_are_absent() -> None:
    """The pre-Deep-Agents pipeline is deleted and must not come back."""

    for module in sorted(LEGACY_AGENT_MODULES):
        assert not (_chain_server_src() / f"{module}.py").exists()


def test_chain_server_sources_do_not_reference_legacy_agents() -> None:
    """No serving-path module may import or name the deleted legacy stack.

    Module and symbol names are compared exactly. Substring matching would
    reject legitimate future modules such as ``cart_effects``.
    """

    for source_path in sorted(_chain_server_src().glob("*.py")):
        tree = ast.parse(source_path.read_text())

        imported = _chain_server_imports(tree) & LEGACY_AGENT_MODULES
        assert not imported, f"{source_path.name} imports {sorted(imported)}"

        named = _referenced_symbols(tree) & LEGACY_AGENT_SYMBOLS
        assert not named, f"{source_path.name} names {sorted(named)}"
