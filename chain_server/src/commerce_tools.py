# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal commerce tool wrappers used by agent runtimes.

These functions are deliberately small adapters around existing services. They
return shared commerce contracts so current LangGraph agents, future Deep
Agents tools, and later protocol adapters can share the same typed boundary.
"""

from __future__ import annotations

import re
from typing import Any

import requests

from shared.commerce_contracts import (
    CommerceError,
    Money,
    ProductSummary,
    SearchCatalogInput,
    SearchCatalogResult,
)


_PRICE_RE = re.compile(r"\bPRICE:\s*\$?([0-9][0-9,]*(?:\.[0-9]+)?)", re.IGNORECASE)


def search_catalog(
    request: SearchCatalogInput,
    catalog_retriever_url: str,
    *,
    timeout_seconds: float = 10,
    session: requests.Session | None = None,
) -> SearchCatalogResult:
    """Search the product catalog without using shopper session state.

    The caller may use conversation context to build ``request``, but this tool
    only reads the catalog for the supplied query/image/categories/filters. It
    does not accept ``user_id``, cart state, or memory context.
    """

    query = request.query.strip()
    image = request.image_base64.strip()
    if not query and not image:
        return SearchCatalogResult(
            ok=False,
            error=CommerceError(
                code="invalid_search_request",
                message="Catalog search requires a query or image.",
            ),
        )

    endpoint = "query/image" if image else "query/text"
    payload: dict[str, Any] = {
        "text": [query] if query else [],
        "categories": request.categories,
        "filters": request.filters,
        "k": request.top_k,
    }
    if image:
        payload["image_base64"] = image

    http = session or requests.Session()
    try:
        response = http.post(
            f"{catalog_retriever_url.rstrip('/')}/{endpoint}",
            json=payload,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        return SearchCatalogResult(
            ok=False,
            error=CommerceError(
                code="catalog_request_failed",
                message="Catalog search request failed.",
                retryable=True,
                details={"error": str(exc)},
            ),
        )
    except ValueError as exc:
        return SearchCatalogResult(
            ok=False,
            error=CommerceError(
                code="catalog_response_invalid",
                message="Catalog search returned an invalid response.",
                details={"error": str(exc)},
            ),
        )

    return SearchCatalogResult(ok=True, products=_products_from_catalog_response(data))


def _products_from_catalog_response(data: dict[str, Any]) -> list[ProductSummary]:
    texts = data.get("texts") or []
    ids = data.get("ids") or []
    names = data.get("names") or []
    images = data.get("images") or []
    similarities = data.get("similarities") or []

    products: list[ProductSummary] = []
    for text, product_id, name, image_url, similarity in zip(
        texts, ids, names, images, similarities
    ):
        if not product_id or not name:
            continue
        products.append(
            ProductSummary(
                product_id=str(product_id),
                display_name=str(name),
                description=_strip_price(str(text or "")),
                category=_category_from_text(str(text or "")),
                price=_price_from_text(str(text or "")),
                image_url=str(image_url) if image_url else None,
                attributes={"similarity": float(similarity)},
            )
        )
    return products


def _price_from_text(text: str) -> Money | None:
    match = _PRICE_RE.search(text)
    if not match:
        return None
    try:
        return Money(amount=float(match.group(1).replace(",", "")))
    except ValueError:
        return None


def _strip_price(text: str) -> str:
    return _PRICE_RE.sub("", text).strip()


def _category_from_text(text: str) -> str | None:
    before_price = text.split("PRICE:", 1)[0]
    parts = [part.strip() for part in before_price.split("|")]
    if len(parts) < 3:
        return None
    category = parts[-1].split(",", 1)[0].strip()
    return category or None
