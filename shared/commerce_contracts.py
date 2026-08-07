# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal commerce contracts for agent-facing tools.

These models define the app's product, cart, and tool result shapes. The
contracts are stable, while product-ID durability across catalog replacements
still depends on the source feed. They intentionally avoid protocol-specific
ACP/UCP fields so those protocols can be added later as adapter layers instead
of driving the core design.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


Availability = Literal["in_stock", "out_of_stock", "preorder", "backorder", "unknown"]
CatalogFilterType = Literal["enum", "enum_list", "number", "text"]
CatalogFieldType = Literal["enum", "enum_list", "number", "text", "unclassified"]


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
    product_id: str = Field(
        ...,
        min_length=1,
        description=(
            "Source catalog product identifier. Its durability across catalog "
            "replacements depends on the feed."
        ),
    )
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
    #: The size on this line, or None for one-size goods.
    size: str | None = Field(default=None, max_length=32)
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
    candidate_k: int | None = Field(default=None, ge=1, le=200)


class SearchCatalogResult(CommerceModel):
    ok: bool
    products: list[ProductSummary] = Field(default_factory=list)
    error: CommerceError | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    no_result_reason: str | None = None
    meta: ToolMeta = Field(default_factory=ToolMeta)


class CatalogFilterCapability(CommerceModel):
    type: CatalogFilterType
    operators: list[str] = Field(default_factory=list)
    source_fields: list[str] = Field(default_factory=list)
    values: list[str] = Field(default_factory=list)
    min_value: float | None = None
    max_value: float | None = None
    request_aliases: dict[str, str] = Field(default_factory=dict)


class CatalogValueCapability(CommerceModel):
    value: str
    count: int = Field(..., ge=1)


class CatalogCoverage(CommerceModel):
    present: int = Field(..., ge=0)
    total: int = Field(..., ge=0)


class CatalogFieldCapability(CommerceModel):
    type: CatalogFieldType
    observed_type: str | None = None
    filterable: bool = False
    searchable: bool = False
    detail: bool = False
    taxonomy: bool = False
    operators: list[str] = Field(default_factory=list)
    source_fields: list[str] = Field(default_factory=list)
    coverage: CatalogCoverage
    values: list[CatalogValueCapability] = Field(default_factory=list)
    min_value: float | None = None
    max_value: float | None = None


class CatalogTaxonomySubcategory(CommerceModel):
    product_count: int = Field(..., ge=0)
    filters: dict[str, CatalogFieldCapability] = Field(default_factory=dict)
    semantic_fields: dict[str, CatalogFieldCapability] = Field(default_factory=dict)


class CatalogTaxonomyCategory(CommerceModel):
    product_count: int = Field(..., ge=0)
    filters: dict[str, CatalogFieldCapability] = Field(default_factory=dict)
    semantic_fields: dict[str, CatalogFieldCapability] = Field(default_factory=dict)
    subcategories: dict[str, CatalogTaxonomySubcategory] = Field(default_factory=dict)


class CatalogTaxonomyCapabilities(CommerceModel):
    category_field: str | None = None
    subcategory_field: str | None = None
    categories: dict[str, CatalogTaxonomyCategory] = Field(default_factory=dict)


class CatalogCapabilities(CommerceModel):
    catalog_id: str = "default"
    product_count: int = Field(default=0, ge=0)
    retrieval_modes: list[str] = Field(default_factory=list)
    image_search_enabled: bool = False
    filters: dict[str, CatalogFilterCapability] = Field(default_factory=dict)
    fields: dict[str, CatalogFieldCapability] = Field(default_factory=dict)
    taxonomy: CatalogTaxonomyCapabilities = Field(
        default_factory=CatalogTaxonomyCapabilities
    )


class GetProductDetailsInput(CommerceModel):
    product_id: str = Field(..., min_length=1)


class GetProductDetailsResult(CommerceModel):
    ok: bool
    product: ProductDetail | None = None
    error: CommerceError | None = None
    meta: ToolMeta = Field(default_factory=ToolMeta)


class CheckProductAvailabilityInput(CommerceModel):
    product_ref: str = Field(..., min_length=1)
    # Field name retained for compatibility; the current stub accepts size wording.
    variant_hint: str | None = None


class CheckProductAvailabilityResult(CommerceModel):
    ok: bool
    product_ref: str
    availability: Availability
    message: str
    meta: ToolMeta = Field(default_factory=ToolMeta)


class CheckActivePromotionsResult(CommerceModel):
    ok: bool
    active: bool
    message: str
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
    #: The size the shopper chose, or None for one-size goods. Distinct from
    #: `variant_id`, which would be a catalog variant: the catalog lists which
    #: sizes a product comes in, this records which one was picked.
    size: str | None = Field(default=None, max_length=32)
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
