# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from pydantic import ValidationError

from shared.commerce_contracts import (
    AddCartItemInput,
    Cart,
    CartLine,
    Money,
    ProductSummary,
    SearchCatalogResult,
)


def test_product_summary_requires_stable_product_id() -> None:
    product = ProductSummary(
        product_id="prod_123",
        display_name="Classic Black Patent Leather Purse",
        price=Money(amount=89.99),
        availability="in_stock",
    )

    assert product.product_id == "prod_123"
    assert product.model_dump()["price"]["currency"] == "USD"


def test_cart_line_quantity_must_be_positive() -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        CartLine(
            cart_line_id="line_1",
            product_id="prod_123",
            display_name="Classic Black Patent Leather Purse",
            quantity=0,
        )


def test_mutating_cart_input_requires_idempotency_key() -> None:
    with pytest.raises(ValidationError, match="idempotency_key"):
        AddCartItemInput(
            user_id="user_1",
            product_id="prod_123",
            quantity=1,
        )


def test_contracts_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ProductSummary(
            product_id="prod_123",
            display_name="Classic Black Patent Leather Purse",
            unsupported_field=True,
        )


def test_search_result_serializes_structured_products() -> None:
    cart = Cart(
        user_id="user_1",
        lines=[
            CartLine(
                cart_line_id="line_1",
                product_id="prod_123",
                display_name="Classic Black Patent Leather Purse",
                quantity=1,
                unit_price=Money(amount=89.99),
            )
        ],
        subtotal=Money(amount=89.99),
    )
    result = SearchCatalogResult(
        ok=True,
        products=[
            ProductSummary(
                product_id=cart.lines[0].product_id,
                display_name=cart.lines[0].display_name,
            )
        ],
    )

    dumped = result.model_dump()
    assert dumped["products"][0]["product_id"] == "prod_123"
    assert dumped["products"][0]["display_name"] == cart.lines[0].display_name
