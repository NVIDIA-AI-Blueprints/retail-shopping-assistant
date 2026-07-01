# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal commerce contracts for agent-facing tools.

These models define the app's stable product, cart, and tool result shapes.
They intentionally avoid protocol-specific ACP/UCP fields so those protocols
can be added later as adapter layers instead of driving the core design.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


Availability = Literal["in_stock", "out_of_stock", "preorder", "backorder", "unknown"]


class CommerceModel(BaseModel):
    """Base model for commerce contracts."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Money(CommerceModel):
    amount: float = Field(..., ge=0, description="Decimal currency amount.")
    currency: str = Field(default="USD", min_length=3, max_length=3)


class ProductVariant(CommerceModel):
    variant_id: str = Field(..., min_length=1)
    product_id: str = Field(..., min_length=1)
    display_name: str = Field(..., min_length=1)
    options: dict[str, str] = Field(default_factory=dict)
    availability: Availability = "unknown"
    price: Money | None = None


class ProductSummary(CommerceModel):
    product_id: str = Field(..., min_length=1)
    display_name: str = Field(..., min_length=1)
    description: str = ""
    category: str | None = None
    brand: str | None = None
    price: Money | None = None
    image_url: str | None = None
    availability: Availability = "unknown"
    attributes: dict[str, Any] = Field(default_factory=dict)


class ProductDetail(ProductSummary):
    variants: list[ProductVariant] = Field(default_factory=list)
    source_uri: str | None = None


class StorePolicy(CommerceModel):
    policy_id: str = Field(..., min_length=1)
    topic: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    body: str = Field(..., min_length=1)
    source_uri: str | None = None


class CartLine(CommerceModel):
    cart_line_id: str = Field(..., min_length=1)
    product_id: str = Field(..., min_length=1)
    display_name: str = Field(..., min_length=1)
    quantity: int = Field(..., ge=1)
    variant_id: str | None = None
    unit_price: Money | None = None
    image_url: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class Cart(CommerceModel):
    user_id: str = Field(..., min_length=1)
    lines: list[CartLine] = Field(default_factory=list)
    subtotal: Money | None = None


class CommerceError(CommerceModel):
    code: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class ToolMeta(CommerceModel):
    trace_id: str | None = None
    idempotency_key: str | None = None


class SearchCatalogInput(CommerceModel):
    """Stateless catalog search input.

    This contract intentionally excludes user, cart, and conversation context.
    Agents may use context to decide what query to send, but the catalog search
    tool itself should behave as a pure read for the supplied request fields.
    """

    query: str = ""
    queries: list[str] = Field(default_factory=list)
    image_base64: str = ""
    categories: list[str] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)
    top_k: int = Field(default=4, ge=1, le=50)


class SearchCatalogResult(CommerceModel):
    ok: bool
    products: list[ProductSummary] = Field(default_factory=list)
    error: CommerceError | None = None
    meta: ToolMeta = Field(default_factory=ToolMeta)


class GetProductDetailsInput(CommerceModel):
    product_id: str = Field(..., min_length=1)


class GetProductDetailsResult(CommerceModel):
    ok: bool
    product: ProductDetail | None = None
    error: CommerceError | None = None
    meta: ToolMeta = Field(default_factory=ToolMeta)


class GetCartInput(CommerceModel):
    user_id: str = Field(..., min_length=1)


class GetCartResult(CommerceModel):
    ok: bool
    cart: Cart | None = None
    error: CommerceError | None = None
    meta: ToolMeta = Field(default_factory=ToolMeta)


class GetStorePolicyInput(CommerceModel):
    topic: str = Field(..., min_length=1)


class GetStorePolicyResult(CommerceModel):
    ok: bool
    policy: StorePolicy | None = None
    error: CommerceError | None = None
    meta: ToolMeta = Field(default_factory=ToolMeta)


class AddCartItemInput(CommerceModel):
    user_id: str = Field(..., min_length=1)
    product_id: str = Field(..., min_length=1)
    quantity: int = Field(..., ge=1)
    idempotency_key: str = Field(..., min_length=1)
    variant_id: str | None = None
    display_name: str | None = None
    unit_price: Money | None = None
    image_url: str | None = None


class UpdateCartItemInput(CommerceModel):
    user_id: str = Field(..., min_length=1)
    cart_line_id: str = Field(..., min_length=1)
    quantity: int = Field(..., ge=0)
    idempotency_key: str = Field(..., min_length=1)


class RemoveCartItemInput(CommerceModel):
    user_id: str = Field(..., min_length=1)
    cart_line_id: str = Field(..., min_length=1)
    idempotency_key: str = Field(..., min_length=1)
    quantity: int = Field(default=1, ge=1)
    product_id: str | None = None
    display_name: str | None = None


class CartMutationResult(CommerceModel):
    ok: bool
    cart: Cart | None = None
    changed_line: CartLine | None = None
    error: CommerceError | None = None
    message: str = ""
    meta: ToolMeta = Field(default_factory=ToolMeta)
