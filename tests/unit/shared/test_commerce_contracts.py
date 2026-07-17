# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from pydantic import ValidationError

from shared.commerce_contracts import (
    AddCartItemInput,
    CatalogCapabilities,
    CatalogCoverage,
    CatalogFieldCapability,
    CatalogTaxonomyCapabilities,
    CatalogTaxonomyCategory,
    CatalogValueCapability,
    Cart,
    CartLine,
    Money,
    ProductSummary,
    SearchCatalogInput,
    SearchCatalogResult,
)


def test_product_summary_requires_product_id() -> None:
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


def test_catalog_capabilities_support_nested_taxonomy_and_enum_lists() -> None:
    tags = CatalogFieldCapability(
        type="enum_list",
        filterable=True,
        searchable=True,
        detail=True,
        operators=["in"],
        source_fields=["tags"],
        coverage=CatalogCoverage(present=2, total=3),
        values=[CatalogValueCapability(value="travel", count=2)],
    )
    capabilities = CatalogCapabilities(
        catalog_id="dynamic",
        product_count=3,
        fields={"tags": tags},
        taxonomy=CatalogTaxonomyCapabilities(
            category_field="category",
            categories={
                "new_category": CatalogTaxonomyCategory(
                    product_count=3,
                    filters={"tags": tags},
                    semantic_fields={"tags": tags},
                )
            },
        ),
    )

    assert capabilities.fields["tags"].type == "enum_list"
    assert capabilities.taxonomy.categories["new_category"].product_count == 3


def test_search_input_rejects_client_supplied_embedding_vectors() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SearchCatalogInput.model_validate(
            {"query": "travel pants", "embedding": [0.1, 0.2]}
        )
