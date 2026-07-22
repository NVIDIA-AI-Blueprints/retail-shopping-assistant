# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal commerce tool wrappers used by agent runtimes.

These functions are deliberately small adapters around existing services. They
return shared commerce contracts so current LangGraph agents, future Deep
Agents tools, and later protocol adapters can share the same typed boundary.
"""

# Cart reads expose the memory service's opaque CartItem.cart_line_id as
# CART_LINE_ID. Add requests use catalog product identity; remove and absolute
# quantity requests use the stable cart-line ID.

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from shared.commerce_contracts import (
    AddCartItemInput,
    Cart,
    CartLine,
    CartMutationResult,
    CheckActivePromotionsResult,
    CheckProductAvailabilityInput,
    CheckProductAvailabilityResult,
    CommerceError,
    GetCartInput,
    GetCartResult,
    GetProductDetailsInput,
    GetProductDetailsResult,
    GetStorePolicyInput,
    GetStorePolicyResult,
    Money,
    ProductDetail,
    ProductSummary,
    RemoveCartItemInput,
    SearchCatalogInput,
    SearchCatalogResult,
    StorePolicy,
    ToolMeta,
    UpdateCartItemInput,
)


_POLICY_CACHE: tuple[bool, dict[str, StorePolicy]] | None = None
_POLICY_CACHE_LOCK = Lock()
_POLICY_PLACEHOLDER_MARKER = "[operator placeholder]"
_SIZED_AVAILABILITY_CATEGORIES = frozenset({"apparel", "footwear"})


def search_catalog(
    request: SearchCatalogInput,
    catalog_retriever_url: str,
    *,
    timeout_seconds: float | None = None,
    session: requests.Session | None = None,
) -> SearchCatalogResult:
    """Search the product catalog without using shopper session state.

    The caller may use conversation context to build ``request``, but this tool
    only reads the catalog for the supplied query/image/categories/filters. It
    does not accept ``user_id``, cart state, or memory context.
    """

    query_terms = _query_terms(request)
    image = request.image_base64.strip()
    if not query_terms and not image:
        return SearchCatalogResult(
            ok=False,
            error=CommerceError(
                code="invalid_search_request",
                message="Catalog search requires a query or image.",
            ),
        )

    endpoint = "query/image" if image else "query/text"
    payload: dict[str, Any] = {
        "text": query_terms,
        "categories": request.categories,
        "filters": request.filters,
        "k": request.top_k,
    }
    if request.candidate_k is not None:
        payload["candidate_k"] = request.candidate_k
    if image:
        payload["image_base64"] = image

    http = session or _catalog_session()
    try:
        response = http.post(
            f"{catalog_retriever_url.rstrip('/')}/{endpoint}",
            json=payload,
            timeout=timeout_seconds,
        )
        if getattr(response, "status_code", None) in {400, 422}:
            status_code = int(response.status_code)
            try:
                error_payload = response.json()
            except (AttributeError, TypeError, ValueError):
                error_payload = {}
            detail = (
                error_payload.get("detail")
                if isinstance(error_payload, dict)
                else None
            )
            message = detail if isinstance(detail, str) and detail.strip() else (
                "Catalog rejected the search request."
            )
            return SearchCatalogResult(
                ok=False,
                error=CommerceError(
                    code=(
                        "catalog_filter_rejected"
                        if status_code == 422
                        else "catalog_request_rejected"
                    ),
                    message=message,
                    details={"status_code": status_code},
                ),
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

    return SearchCatalogResult(
        ok=True,
        products=_products_from_catalog_response(data),
        diagnostics=data.get("diagnostics") or {},
        no_result_reason=data.get("no_result_reason"),
    )


def get_product_details(
    request: GetProductDetailsInput,
    catalog_retriever_url: str,
    *,
    timeout_seconds: float | None = None,
    session: Any | None = None,
) -> GetProductDetailsResult:
    """Read one product from the active deterministic catalog snapshot."""

    http = session or requests
    product_id = quote(request.product_id, safe="")
    try:
        response = http.get(
            f"{catalog_retriever_url.rstrip('/')}/products/{product_id}",
            timeout=timeout_seconds,
        )
        if getattr(response, "status_code", None) == 404:
            return GetProductDetailsResult(
                ok=False,
                error=CommerceError(
                    code="product_not_found",
                    message="Product is not present in the active catalog. Search again.",
                ),
            )
        response.raise_for_status()
        product = ProductDetail.model_validate(response.json())
    except requests.RequestException as exc:
        return GetProductDetailsResult(
            ok=False,
            error=CommerceError(
                code="catalog_request_failed",
                message="Product detail request failed.",
                retryable=True,
                details={"error": str(exc)},
            ),
        )
    except ValueError as exc:
        return GetProductDetailsResult(
            ok=False,
            error=CommerceError(
                code="catalog_response_invalid",
                message="Product detail response was invalid.",
                details={"error": str(exc)},
            ),
        )

    if product.product_id != request.product_id:
        return GetProductDetailsResult(
            ok=False,
            error=CommerceError(
                code="catalog_response_invalid",
                message="Product detail response did not match the requested product.",
            ),
        )
    return GetProductDetailsResult(ok=True, product=product)


def get_cart(
    request: GetCartInput,
    memory_retriever_url: str,
    *,
    timeout_seconds: float = 10,
    session: Any | None = None,
) -> GetCartResult:
    """Read the authoritative cart for a shopper from the memory service."""

    http = session or requests
    try:
        response = http.get(
            f"{memory_retriever_url.rstrip('/')}/user/{request.user_id}/cart",
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        return GetCartResult(
            ok=False,
            error=CommerceError(
                code="cart_read_failed",
                message="Cart read request failed.",
                retryable=True,
                details={"error": str(exc)},
            ),
        )
    except ValueError as exc:
        return GetCartResult(
            ok=False,
            error=CommerceError(
                code="cart_response_invalid",
                message="Cart read returned an invalid response.",
                details={"error": str(exc)},
            ),
        )

    return GetCartResult(
        ok=True,
        cart=_cart_from_memory_items(str(request.user_id), data.get("cart") or []),
    )


def add_cart_item(
    request: AddCartItemInput,
    memory_retriever_url: str,
    *,
    timeout_seconds: float = 10,
    session: Any | None = None,
) -> CartMutationResult:
    """Add one item to the shopper cart through the memory service adapter."""

    display_name = request.display_name or request.product_id
    payload: dict[str, Any] = {
        "product_id": request.product_id,
        "item": display_name,
        "amount": request.quantity,
        "idempotency_key": request.idempotency_key,
    }
    if request.unit_price is not None:
        payload["price"] = request.unit_price.amount

    http = session or requests
    try:
        response = http.post(
            f"{memory_retriever_url.rstrip('/')}/user/{request.user_id}/cart/add",
            json=payload,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        return CartMutationResult(
            ok=False,
            error=CommerceError(
                code="cart_add_failed",
                message=f"Failed to add {request.quantity} {display_name} to cart.",
                details={"error": str(exc)},
            ),
            meta=ToolMeta(idempotency_key=request.idempotency_key),
        )
    except ValueError as exc:
        return CartMutationResult(
            ok=False,
            error=CommerceError(
                code="cart_response_invalid",
                message="Cart add returned an invalid response.",
                details={"error": str(exc)},
            ),
            meta=ToolMeta(idempotency_key=request.idempotency_key),
        )

    changed_line = _cart_line_from_memory_item(data.get("cart_line") or {})
    if changed_line is None or changed_line.product_id != request.product_id:
        return CartMutationResult(
            ok=False,
            error=CommerceError(
                code="cart_response_invalid",
                message="Cart add returned an invalid response.",
            ),
            meta=ToolMeta(idempotency_key=request.idempotency_key),
        )
    return CartMutationResult(
        ok=True,
        changed_line=changed_line,
        message=str(data.get("message") or ""),
        meta=ToolMeta(idempotency_key=request.idempotency_key),
    )


def remove_cart_item(
    request: RemoveCartItemInput,
    memory_retriever_url: str,
    *,
    timeout_seconds: float = 10,
    session: Any | None = None,
) -> CartMutationResult:
    """Remove quantity from a cart line through the memory service adapter."""

    display_name = request.display_name or request.cart_line_id
    http = session or requests
    try:
        response = http.post(
            f"{memory_retriever_url.rstrip('/')}/user/{request.user_id}/cart/remove",
            json={
                "cart_line_id": request.cart_line_id,
                "amount": request.quantity,
                "idempotency_key": request.idempotency_key,
            },
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        return CartMutationResult(
            ok=False,
            error=CommerceError(
                code="cart_remove_failed",
                message=f"Failed to remove {request.quantity} {display_name} from cart.",
                details={"error": str(exc)},
            ),
            meta=ToolMeta(idempotency_key=request.idempotency_key),
        )
    except ValueError as exc:
        return CartMutationResult(
            ok=False,
            error=CommerceError(
                code="cart_response_invalid",
                message="Cart remove returned an invalid response.",
                details={"error": str(exc)},
            ),
            meta=ToolMeta(idempotency_key=request.idempotency_key),
        )

    cart_line_data = data.get("cart_line") if isinstance(data, dict) else None
    changed_line = None
    if isinstance(cart_line_data, dict):
        response_line_id = str(cart_line_data.get("cart_line_id") or "")
        if response_line_id != request.cart_line_id:
            return CartMutationResult(
                ok=False,
                error=CommerceError(
                    code="cart_response_invalid",
                    message="Cart remove returned an invalid response.",
                ),
                meta=ToolMeta(idempotency_key=request.idempotency_key),
            )
        changed_line = _cart_line_from_memory_item(cart_line_data)
    else:
        return CartMutationResult(
            ok=False,
            error=CommerceError(
                code="cart_response_invalid",
                message="Cart remove returned an invalid response.",
            ),
            meta=ToolMeta(idempotency_key=request.idempotency_key),
        )
    return CartMutationResult(
        ok=True,
        changed_line=changed_line,
        message=str(data.get("message") or ""),
        meta=ToolMeta(idempotency_key=request.idempotency_key),
    )


def update_cart_item(
    request: UpdateCartItemInput,
    memory_retriever_url: str,
    *,
    timeout_seconds: float = 10,
    session: Any | None = None,
) -> CartMutationResult:
    """Set an existing cart line's absolute quantity in one service request."""

    http = session or requests
    cart_line_id = quote(request.cart_line_id, safe="")
    try:
        response = http.put(
            (
                f"{memory_retriever_url.rstrip('/')}/user/{request.user_id}"
                f"/cart/{cart_line_id}/quantity"
            ),
            json={
                "quantity": request.quantity,
                "idempotency_key": request.idempotency_key,
            },
            timeout=timeout_seconds,
        )
        status_code = getattr(response, "status_code", None)
        if status_code == 404:
            return _cart_update_failure(
                request,
                "cart_line_not_found",
                f"CART_LINE_ID '{request.cart_line_id}' not found. "
                "Call get_cart_tool to get current line IDs.",
            )
        if isinstance(status_code, int) and 400 <= status_code < 500:
            return _cart_update_failure(
                request,
                "cart_update_failed",
                "Cart quantity update was rejected.",
                details={"status_code": status_code},
            )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        return _cart_update_failure(
            request,
            "cart_update_failed",
            "Cart quantity update failed.",
            retryable=True,
            exception=exc,
        )
    except ValueError as exc:
        return _cart_update_failure(
            request,
            "cart_response_invalid",
            "Cart update returned an invalid response.",
            exception=exc,
        )
    return _cart_update_result(request, data)


def _cart_update_result(
    request: UpdateCartItemInput,
    data: Any,
) -> CartMutationResult:
    cart_line_data = data.get("cart_line") if isinstance(data, dict) else None
    if not isinstance(cart_line_data, dict) or str(
        cart_line_data.get("cart_line_id") or ""
    ) != request.cart_line_id:
        return _cart_update_failure(
            request,
            "cart_response_invalid",
            "Cart update returned an invalid response.",
        )
    if _int_or_default(cart_line_data.get("amount"), -1) != request.quantity:
        return _cart_update_failure(
            request,
            "cart_response_invalid",
            "Cart update returned an invalid response.",
        )

    changed_line = None
    if request.quantity > 0:
        changed_line = _cart_line_from_memory_item(cart_line_data)
        if changed_line is None:
            return _cart_update_failure(
                request,
                "cart_response_invalid",
                "Cart update returned an invalid response.",
            )

    return CartMutationResult(
        ok=True,
        changed_line=changed_line,
        message=str(data.get("message") or ""),
        meta=ToolMeta(idempotency_key=request.idempotency_key),
    )


def _cart_update_failure(
    request: UpdateCartItemInput,
    code: str,
    message: str,
    *,
    retryable: bool = False,
    exception: Exception | None = None,
    details: dict[str, Any] | None = None,
) -> CartMutationResult:
    error_details = dict(details or {})
    if exception is not None:
        error_details["error"] = str(exception)
    return CartMutationResult(
        ok=False,
        error=CommerceError(
            code=code,
            message=message,
            retryable=retryable,
            details=error_details,
        ),
        meta=ToolMeta(idempotency_key=request.idempotency_key),
    )


def _load_policies(
    policies_path: str | Path,
) -> tuple[bool, dict[str, StorePolicy]]:
    global _POLICY_CACHE
    with _POLICY_CACHE_LOCK:
        if _POLICY_CACHE is None:
            import yaml

            with open(policies_path, "r") as policy_file:
                data = yaml.safe_load(policy_file) or {}
            configured = data.get("configured") is True
            policy_rows = data.get("policies") or {}
            policies: dict[str, StorePolicy] = {}
            if configured:
                if any(
                    _POLICY_PLACEHOLDER_MARKER
                    in str(policy.get(field) or "").casefold()
                    for policy in policy_rows.values()
                    for field in ("title", "body")
                ):
                    raise ValueError(
                        "Configured store policies still contain operator placeholders."
                    )
                policies = {
                    topic: StorePolicy(
                        policy_id=topic,
                        topic=topic,
                        title=policy["title"],
                        body=policy["body"],
                        source_uri=policy.get("source_uri"),
                    )
                    for topic, policy in policy_rows.items()
                }
            _POLICY_CACHE = (configured, policies)
    return _POLICY_CACHE


def get_store_policy(
    request: GetStorePolicyInput,
    policies_path: str | Path,
) -> GetStorePolicyResult:
    """Read controlled store policy content from a static YAML file.

    Never reads from model knowledge. Returns a structured failure when the
    topic is absent so the agent can relay the honest answer to the shopper.
    """

    try:
        configured, policies = _load_policies(policies_path)
    except Exception as exc:
        return GetStorePolicyResult(
            ok=False,
            error=CommerceError(
                code="policy_load_failed",
                message="Policy file could not be loaded.",
                details={"error": str(exc)},
            ),
        )

    if not configured:
        return GetStorePolicyResult(
            ok=False,
            error=CommerceError(
                code="policy_not_configured",
                message=(
                    "Store policies are not configured for this deployment. "
                    "Direct the shopper to the retailer's help center."
                ),
            ),
        )

    policy = policies.get(request.topic)
    if policy is None:
        return GetStorePolicyResult(
            ok=False,
            error=CommerceError(
                code="policy_topic_not_found",
                message=(
                    f"Policy topic '{request.topic}' is not available through "
                    "the assistant. Direct the shopper to the retailer's help center."
                ),
            ),
        )
    return GetStorePolicyResult(ok=True, policy=policy)


def check_product_availability(
    request: CheckProductAvailabilityInput,
    product: ProductSummary,
) -> CheckProductAvailabilityResult:
    """Return the configured availability-stub result for a known product.

    Apparel and footwear accept a requested size. Other catalog categories are
    treated as one-size products. No inventory call is made.
    """

    variant_hint = (request.variant_hint or "").strip()
    if not variant_hint:
        message = f"Yes, {product.display_name} is available."
    elif _availability_category(product) in _SIZED_AVAILABILITY_CATEGORIES:
        message = f"Yes, {product.display_name} is available in {variant_hint}."
    else:
        message = f"{product.display_name} is one-size-fits-all and is available."

    return CheckProductAvailabilityResult(
        ok=True,
        product_ref=request.product_ref,
        availability="in_stock",
        message=message,
    )


def check_active_promotions() -> CheckActivePromotionsResult:
    """Return the fixed promotion-status boundary without making an external call."""

    return CheckActivePromotionsResult(
        ok=True,
        active=False,
        message=(
            "No active sale or promotion is available through the assistant "
            "right now."
        ),
    )


def _availability_category(product: ProductSummary) -> str:
    taxonomy = product.attributes.get("taxonomy")
    if isinstance(taxonomy, dict):
        category = str(taxonomy.get("category") or "").strip().lower()
        if category:
            return category
    return str(product.category or "").strip().lower()


def _query_terms(request: SearchCatalogInput) -> list[str]:
    if request.queries:
        return [query.strip() for query in request.queries if query.strip()]
    query = request.query.strip()
    return [query] if query else []


def _catalog_session() -> requests.Session:
    retry_strategy = Retry(
        total=3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"],
        backoff_factor=1,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _products_from_catalog_response(data: dict[str, Any]) -> list[ProductSummary]:
    structured_products = data.get("products") or []
    if structured_products:
        products: list[ProductSummary] = []
        for product in structured_products:
            if not isinstance(product, dict):
                continue
            try:
                products.append(ProductSummary.model_validate(product))
            except ValueError:
                continue
        return products

    texts = data.get("texts") or []
    ids = data.get("ids") or []
    names = data.get("names") or []
    images = data.get("images") or []
    similarities = data.get("similarities") or []

    products: list[ProductSummary] = []
    item_count = max(len(texts), len(ids), len(names), len(images), len(similarities))
    for idx in range(item_count):
        text = _list_get(texts, idx, "")
        name = _list_get(names, idx, "")
        product_id = _list_get(ids, idx, name)
        image_url = _list_get(images, idx, None)
        similarity = _float_or_default(_list_get(similarities, idx, 0.0), 0.0)
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
                attributes={
                    "catalog_text": str(text or ""),
                    "similarity": similarity,
                },
            )
        )
    return products


def _cart_from_memory_items(user_id: str, items: list[dict[str, Any]]) -> Cart:
    lines = [_cart_line_from_memory_item(item) for item in items]
    lines = [line for line in lines if line is not None]

    subtotal: Money | None = None
    if lines and all(line.unit_price is not None for line in lines):
        subtotal = Money(
            amount=sum(
                line.unit_price.amount * line.quantity
                for line in lines
                if line.unit_price is not None
            )
        )

    return Cart(user_id=user_id, lines=lines, subtotal=subtotal)


def _cart_line_from_memory_item(item: dict[str, Any]) -> CartLine | None:
    display_name = str(item.get("item") or "").strip()
    if not display_name:
        return None
    quantity = _int_or_default(item.get("amount"), 0)
    if quantity <= 0:
        return None

    price = _float_or_default(item.get("price"), None)
    return CartLine(
        cart_line_id=str(item.get("cart_line_id") or display_name),
        product_id=str(item.get("product_id") or display_name),
        display_name=display_name,
        quantity=quantity,
        unit_price=Money(amount=price) if price is not None else None,
    )


def _list_get(values: list[Any], idx: int, default: Any) -> Any:
    return values[idx] if idx < len(values) else default


def _int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_or_default(value: Any, default: float | None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _price_from_text(text: str) -> Money | None:
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip().upper() == "PRICE":
            try:
                return Money(amount=float(value.strip().replace("$", "").replace(",", "")))
            except ValueError:
                return None
    return None


def _strip_price(text: str) -> str:
    lines = []
    for line in text.splitlines():
        key, separator, _value = line.partition(":")
        if separator and key.strip().upper() == "PRICE":
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _category_from_text(text: str) -> str | None:
    before_price = text.split("PRICE:", 1)[0]
    parts = [part.strip() for part in before_price.split("|")]
    if len(parts) < 3:
        return None
    category = parts[-1].split(",", 1)[0].strip()
    return category or None
