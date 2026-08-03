# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed boundary for products presented earlier in a conversation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Literal
from urllib.parse import quote

import requests
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from shared.commerce_contracts import ProductSummary


_DEFAULT_TIMEOUT_SECONDS = 10.0
_DEFAULT_INDEX_MAX_CHARS = 12_000


class _ConversationProductModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ProductReferenceDescriptor(_ConversationProductModel):
    """Structured selectors for one historical product reference."""

    reference_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Short label used to identify this result.",
    )
    product_ref: str | None = Field(
        default=None,
        min_length=1,
        max_length=512,
        description="Exact PRODUCT_REF from the historical product index.",
    )
    display_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=512,
        description="Exact product name from the historical product index.",
    )
    category: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
        description="Exact category from the historical product index.",
    )
    turn_sequence: int | None = Field(
        default=None,
        ge=1,
        description="Exact turn number from the historical product index.",
    )
    candidate_set_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        description="Exact set ID from the historical product index.",
    )
    ordinal: int | None = Field(
        default=None,
        ge=1,
        description="One-based product position within the selected turn or set.",
    )

    @model_validator(mode="after")
    def _selectors_are_valid(self):
        selectors = (
            self.product_ref,
            self.display_name,
            self.category,
            self.turn_sequence,
            self.candidate_set_id,
        )
        if not any(value is not None for value in selectors):
            raise ValueError("at least one product reference selector is required")
        if self.ordinal is not None and (
            self.turn_sequence is None and self.candidate_set_id is None
        ):
            raise ValueError("ordinal requires turn_sequence or candidate_set_id")
        return self


class ResolveConversationProductsRequest(_ConversationProductModel):
    """One batched historical-product resolution request."""

    references: list[ProductReferenceDescriptor] = Field(
        ...,
        min_length=1,
        max_length=20,
    )


class ConversationProductMatch(_ConversationProductModel):
    """One presented product and its durable presentation coordinates."""

    product: ProductSummary
    candidate_set_id: str = Field(..., min_length=1, max_length=64)
    turn_sequence: int = Field(..., ge=1)
    position: int = Field(..., ge=1)
    catalog_revision: str | None = Field(default=None, max_length=512)


class ProductReferenceResolution(_ConversationProductModel):
    """Deterministic outcome for one reference descriptor."""

    reference_id: str = Field(..., min_length=1, max_length=128)
    status: Literal["resolved", "ambiguous", "not_found"]
    matches: list[ConversationProductMatch] = Field(default_factory=list)
    match_count: int = Field(..., ge=0)

    @model_validator(mode="after")
    def _status_matches_result(self):
        if self.status == "resolved" and (
            self.match_count != 1 or len(self.matches) != 1
        ):
            raise ValueError("resolved references require exactly one match")
        if self.status == "ambiguous" and self.match_count < 2:
            raise ValueError("ambiguous references require multiple matches")
        if self.status == "not_found" and (self.match_count or self.matches):
            raise ValueError("not_found references cannot contain matches")
        return self


class ResolveConversationProductsResult(_ConversationProductModel):
    """Batch response from conversation memory."""

    results: list[ProductReferenceResolution] = Field(..., min_length=1)


class ConversationProductsError(RuntimeError):
    """Stable failure at the historical-product service boundary."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.retryable = retryable


class ConversationProductsClient:
    """Resolve products from durable conversation events in one request."""

    def __init__(
        self,
        memory_retriever_url: str,
        *,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        session: Any | None = None,
    ) -> None:
        self.memory_retriever_url = memory_retriever_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.session = session or requests

    def resolve(
        self,
        conversation_id: str,
        references: Sequence[ProductReferenceDescriptor],
    ) -> ResolveConversationProductsResult:
        """Resolve a nonempty descriptor batch with one memory-service call."""

        request = ResolveConversationProductsRequest(references=list(references))
        try:
            response = self.session.post(
                (
                    f"{self.memory_retriever_url}/conversations/"
                    f"{quote(conversation_id, safe='')}/products/resolve"
                ),
                json=request.model_dump(mode="json", exclude_none=True),
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise ConversationProductsError(
                "conversation_products_request_failed",
                "Historical product resolution request failed.",
                retryable=True,
            ) from exc

        status_code = int(getattr(response, "status_code", 500))
        if status_code >= 400:
            raise ConversationProductsError(
                "conversation_products_unavailable"
                if status_code >= 500
                else "conversation_products_request_rejected",
                "Historical product resolution was not available.",
                status_code=status_code,
                retryable=status_code >= 500,
            )
        try:
            payload = response.json()
            return ResolveConversationProductsResult.model_validate(payload)
        except (TypeError, ValueError, ValidationError) as exc:
            raise ConversationProductsError(
                "conversation_products_response_invalid",
                "Historical product resolution returned an invalid response.",
                status_code=status_code,
            ) from exc


class ProductEvidence:
    """Products authorized for deterministic tools in the current turn."""

    def __init__(self, products: Iterable[ProductSummary] = ()) -> None:
        self._products = {product.product_id: product for product in products}

    def add(self, products: Iterable[ProductSummary]) -> None:
        for product in products:
            self._products[product.product_id] = product

    def add_resolutions(
        self,
        resolutions: Iterable[ProductReferenceResolution],
    ) -> None:
        for resolution in resolutions:
            if resolution.status == "resolved" and len(resolution.matches) == 1:
                self.add([resolution.matches[0].product])

    def get(self, product_ref: str) -> ProductSummary | None:
        return self._products.get((product_ref or "").strip())

    def values(self) -> tuple[ProductSummary, ...]:
        return tuple(self._products.values())


def format_product_resolution(result: ResolveConversationProductsResult) -> str:
    """Render resolution outcomes without authorizing a guessed product."""

    lines: list[str] = []
    for resolution in result.results:
        if resolution.status == "resolved":
            match = resolution.matches[0]
            product = match.product
            # The durable presentation event stores the whole ProductSummary, so
            # these facts are already resolved and in hand. Rendering only the
            # ref and the name discarded them, and the assistant then had no way
            # to answer "which of those was cheapest" about products it had
            # itself shown -- it re-searched, and still came back without the
            # prices it had displayed a few turns earlier.
            lines.extend(
                (
                    f"REFERENCE {resolution.reference_id}: RESOLVED",
                    f"PRODUCT_REF: {product.product_id}",
                    f"NAME: {product.display_name}",
                )
            )
            if product.category:
                lines.append(f"CATEGORY: {product.category}")
            if product.price:
                # Stated as what was shown, with the turn, because a stored
                # presentation is evidence of what the shopper saw and not of
                # today's price.
                lines.append(
                    f"PRICE_WHEN_SHOWN: ${product.price.amount:.2f} "
                    f"{product.price.currency} (turn {match.turn_sequence})"
                )
            if product.image_url:
                lines.append("IMAGE_AVAILABLE: yes")
            lines.append(
                "These are the facts presented earlier. Attributes such as "
                "material, care, and dimensions require "
                "get_product_details_tool. Confirm price with a fresh read "
                "before a cart action or a budget claim."
            )
            continue
        if resolution.status == "ambiguous":
            names = ", ".join(
                match.product.display_name for match in resolution.matches
            )
            lines.append(
                f"REFERENCE {resolution.reference_id}: CLARIFICATION REQUIRED "
                f"({names}). Do not guess."
            )
            continue
        lines.append(
            f"REFERENCE {resolution.reference_id}: NOT FOUND. "
            "Ask which earlier product the shopper means; do not guess."
        )
    return "\n".join(lines)


def format_historical_product_index(
    reference_sets: Sequence[Any],
    *,
    max_chars: int = _DEFAULT_INDEX_MAX_CHARS,
) -> str:
    """Render the compact projection as bounded read-only model context."""

    if max_chars < 256:
        raise ValueError("max_chars must be at least 256")
    heading = "HISTORICAL PRODUCT INDEX (read-only):"
    formatted_sets = []
    for raw_set in reference_sets:
        line = _format_reference_set(raw_set)
        if line:
            formatted_sets.append(line)
    if not formatted_sets:
        return ""

    remaining = max_chars - len(heading) - 1
    selected_newest_first: list[str] = []
    for line in reversed(formatted_sets):
        separator = 1 if selected_newest_first else 0
        if len(line) + separator > remaining:
            break
        selected_newest_first.append(line)
        remaining -= len(line) + separator

    omitted = len(selected_newest_first) < len(formatted_sets)
    marker = "(earlier historical products omitted)"
    if omitted:
        while selected_newest_first and len(marker) + 1 > remaining:
            removed = selected_newest_first.pop()
            remaining += len(removed) + 1
    lines = [heading]
    if omitted:
        lines.append(marker)
    lines.extend(reversed(selected_newest_first))
    return "\n".join(lines)


def _format_reference_set(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    set_id = _one_line(value.get("candidate_set_id"))
    turn = value.get("turn_seq")
    products = value.get("products")
    if not set_id or not isinstance(turn, int) or not isinstance(products, list):
        return ""
    rendered = []
    for product in products:
        if not isinstance(product, dict):
            continue
        ref = _one_line(product.get("ref"))
        name = _one_line(product.get("name"))
        position = product.get("position")
        if not ref or not name or not isinstance(position, int):
            continue
        category = _one_line(product.get("category"))
        rendered.append(
            f"{position}:{name} [{category}] <{ref}>"
            if category
            else f"{position}:{name} <{ref}>"
        )
    return f"- set={set_id} turn={turn}: " + "; ".join(rendered) if rendered else ""


def _one_line(value: Any) -> str:
    return " ".join(str(value or "").split())[:256]
