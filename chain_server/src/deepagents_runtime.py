# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deep Agents SDK runtime for the shopping assistant."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import os
import time
from typing import Any, AsyncIterator
import uuid

from langgraph.checkpoint.memory import MemorySaver
import requests

from .agenttypes import Cart, State
from .catalog_capabilities import (
    CatalogCapabilitiesClient,
    format_catalog_capabilities_for_prompt,
)
from .catalog_execution import execute_catalog_search
from .catalog_request import CatalogSearchIntent, build_catalog_search_plan
from .commerce_tools import (
    add_cart_item,
    get_cart,
    remove_cart_item,
)
from .media_perception import MediaPerceptionClient
from shared.commerce_contracts import (
    AddCartItemInput,
    CatalogCapabilities,
    GetCartInput,
    ProductSummary,
    RemoveCartItemInput,
)


logger = logging.getLogger(__name__)

_EXCLUDED_DEEP_AGENT_TOOLS = frozenset(
    {"write_todos", "ls", "read_file", "write_file", "edit_file", "glob", "grep", "execute"}
)


@dataclass(frozen=True)
class RequestIdentity:
    """Server-owned identity used to scope one assistant turn."""

    session_id: str
    conversation_id: str
    cart_id: str
    context_user_id: int
    cart_user_id: int
    request_id: str

    @property
    def legacy_user_id(self) -> int:
        return self.context_user_id


class DeepAgentsRuntime:
    """Small adapter around the Deep Agents SDK.

    This class intentionally keeps commerce truth in existing services. Deep
    Agents gets scoped context and deterministic tools; it does not own carts,
    profiles, prices, inventory, or session identity.
    """

    def __init__(self, config: Any) -> None:
        self.config = config
        self._checkpointer = MemorySaver()
        self._profile_registered = False
        self._product_refs: dict[str, dict[str, ProductSummary]] = {}
        self._media_perception = MediaPerceptionClient(config)
        self._catalog_capabilities = CatalogCapabilitiesClient(
            config.retriever_port,
            timeout_seconds=config.catalog_search_timeout_seconds,
        )

    def catalog_capabilities(self, *, force_refresh: bool = False) -> CatalogCapabilities:
        """Return catalog-owned capability metadata for API/UI consumers."""

        return self._catalog_capabilities.get(force_refresh=force_refresh)

    async def astream(
        self, state: State, identity: RequestIdentity
    ) -> AsyncIterator[str]:
        output = await self._run_turn(state, identity)
        images = output.retrieved or {}
        yield json.dumps({"type": "images", "payload": images, "timestamp": time.time()})
        if output.response:
            yield json.dumps(
                {"type": "content", "payload": output.response, "timestamp": time.time()}
            )

    async def ainvoke(self, state: State, identity: RequestIdentity) -> dict[str, Any]:
        output = await self._run_turn(state, identity)
        return {
            "response": output.response,
            "images": output.retrieved or {},
            "timings": output.timings,
        }

    async def _run_turn(self, state: State, identity: RequestIdentity) -> State:
        start = time.monotonic()
        state.user_id = identity.context_user_id
        self._load_memory(state, identity)

        if state.guardrails and not self._check_safety("input", identity.context_user_id, state.query):
            state.response = self.config.unsafe_message
            state.timings["deepagents"] = time.monotonic() - start
            return state

        media_start = time.monotonic()
        state.media_analysis = await self._media_perception.analyze(state)
        if state.media:
            state.timings["media_perception"] = time.monotonic() - media_start

        agent = self._create_agent(state, identity)
        input_message = self._build_user_message(state, identity)
        try:
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": input_message}]},
                config={
                    "configurable": {"thread_id": identity.conversation_id},
                    "recursion_limit": 12,
                },
            )
            state.response = _extract_final_text(result)
        except Exception as exc:  # noqa: BLE001 - keep endpoint resilient.
            logger.exception("DeepAgentsRuntime failed")
            state.response = (
                "I encountered an error while helping with your shopping request. "
                "Please try again."
            )
            state.timings["deepagents_error"] = time.monotonic() - start
            return state

        if state.guardrails and not self._check_safety("output", identity.context_user_id, state.response):
            state.response = self.config.unsafe_message

        state.context = self._updated_context(
            state.context,
            state.query,
            state.response,
            media_analysis=state.media_analysis,
        )
        self._persist_context(state, identity)
        state.timings["deepagents"] = time.monotonic() - start
        return state

    def _create_agent(self, state: State, identity: RequestIdentity):
        from deepagents import (
            GeneralPurposeSubagentProfile,
            HarnessProfile,
            create_deep_agent,
            register_harness_profile,
        )
        from langchain_core.tools import tool
        from langchain_openai import ChatOpenAI

        if not self._profile_registered:
            register_harness_profile(
                f"openai:{self.config.llm_name}",
                HarnessProfile(
                    excluded_tools=_EXCLUDED_DEEP_AGENT_TOOLS,
                    general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
                ),
            )
            self._profile_registered = True

        retrieved: dict[str, str] = {}
        state.retrieved = retrieved

        @tool(return_direct=False)
        def search_catalog_tool(
            query: str,
            filters: dict[str, Any] | None = None,
            strictness: str = "unspecified",
            search_mode: str | None = None,
        ) -> str:
            """Execute product discovery with catalog-declared hard filters."""

            capabilities = self._catalog_capabilities.get()
            if capabilities.catalog_id == "unavailable" and not capabilities.filters:
                return "Catalog search is unavailable. Please try again."

            intent = CatalogSearchIntent(
                query=query,
                filters=filters if isinstance(filters, dict) else {},
                strictness=_tool_strictness(strictness),
                search_mode=_tool_search_mode(search_mode),
            )
            plan = build_catalog_search_plan(
                intent,
                capabilities,
                has_image=bool(state.image),
                top_k=self.config.top_k_retrieve,
            )
            if not plan.should_search:
                return "Catalog search requires a query or image."

            execution = execute_catalog_search(
                plan,
                self.config.retriever_port,
                image_base64=state.image,
                timeout_seconds=self.config.catalog_search_timeout_seconds,
            )
            result = execution.result
            if not result.ok:
                return result.error.message if result.error else "Catalog search failed."
            if not result.products:
                return "No matching catalog products were found."

            self._remember_products(identity, result.products)
            lines = []
            for product in result.products:
                if product.image_url:
                    retrieved[product.display_name] = product.image_url
                lines.append(_format_product(product))
            prefix = (
                "Image similarity returned no matches; text fallback results:\n\n"
                if execution.fallback_used
                else ""
            )
            return prefix + "\n\n".join(lines)

        @tool(return_direct=False)
        def get_cart_tool() -> str:
            """Read the current cart contents."""

            cart = self._read_cart(identity.cart_user_id)
            state.cart = cart
            return _format_cart(cart)

        @tool(return_direct=False)
        def get_product_details_tool(product_ref: str) -> str:
            """Read details for a PRODUCT_REF returned by search_catalog_tool."""

            product = self._product_from_ref(identity, product_ref)
            if product is None:
                return (
                    f"No product with PRODUCT_REF '{product_ref}' is available. "
                    "Search the catalog first and use the PRODUCT_REF from the result."
                )
            return _format_product_details(product)

        @tool(return_direct=True)
        def add_cart_item_tool(product_ref: str, quantity: int = 1) -> str:
            """Add a catalog item by PRODUCT_REF from a prior search_catalog_tool result."""

            quantity = max(1, int(quantity or 1))
            product = self._product_from_ref(identity, product_ref)
            if product is None:
                return (
                    f"No product with PRODUCT_REF '{product_ref}' is available. "
                    "Search the catalog first and use the PRODUCT_REF from the result."
                )
            result = add_cart_item(
                AddCartItemInput(
                    user_id=str(identity.cart_user_id),
                    product_id=product.product_id,
                    display_name=product.display_name,
                    quantity=quantity,
                    unit_price=product.price,
                    image_url=product.image_url,
                    idempotency_key=f"{identity.request_id}:add:{product.product_id}:{quantity}",
                ),
                self.config.memory_port,
            )
            state.cart = self._read_cart(identity.cart_user_id)
            if result.ok:
                return result.message or f"Added {quantity} {product.display_name} to cart."
            return result.error.message if result.error else "Cart add failed."

        @tool(return_direct=True)
        def remove_cart_item_tool(cart_line_id: str, quantity: int = 1) -> str:
            """Remove a cart item by CART_LINE_ID from get_cart_tool/current-cart output."""

            quantity = max(1, int(quantity or 1))
            cart = self._read_cart(identity.cart_user_id)
            line = _cart_line_by_id(cart_line_id, cart)
            if line is None:
                return f"No cart line with CART_LINE_ID '{cart_line_id}' could be found."
            result = remove_cart_item(
                RemoveCartItemInput(
                    user_id=str(identity.cart_user_id),
                    cart_line_id=line["cart_line_id"],
                    product_id=line.get("product_id"),
                    display_name=line["item"],
                    quantity=quantity,
                    idempotency_key=f"{identity.request_id}:remove:{line['cart_line_id']}:{quantity}",
                ),
                self.config.memory_port,
            )
            state.cart = self._read_cart(identity.cart_user_id)
            if result.ok:
                return result.message or f"Removed {quantity} {line['item']} from cart."
            return result.error.message if result.error else "Cart remove failed."

        @tool(return_direct=True)
        def view_cart_total_tool() -> str:
            """Compute the current cart total from cached cart line prices."""

            cart = self._read_cart(identity.cart_user_id)
            state.cart = cart
            return _format_cart_total(cart)

        api_key_env = getattr(self.config, "llm_api_key_env", None)
        api_key = os.environ.get(api_key_env, "") if api_key_env else "not-needed"
        model = ChatOpenAI(
            model=self.config.llm_name,
            base_url=self.config.llm_port,
            api_key=api_key or "not-needed",
            temperature=0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return create_deep_agent(
            model=model,
            tools=[
                search_catalog_tool,
                get_product_details_tool,
                get_cart_tool,
                add_cart_item_tool,
                remove_cart_item_tool,
                view_cart_total_tool,
            ],
            system_prompt=self._system_prompt(),
            checkpointer=self._checkpointer,
        )

    def _system_prompt(self) -> str:
        catalog_context = format_catalog_capabilities_for_prompt(
            self._catalog_capabilities.get()
        )
        return f"""You are a retail shopping assistant for clothing and accessories.

Use tools for catalog facts and cart actions. Do not invent product names,
prices, availability, materials, care instructions, or cart changes.

Catalog capabilities:
{catalog_context}

Rules:
- Product discovery, product recommendations, budget filters, and image-similar
  shopping require search_catalog_tool.
- Use the search_catalog_tool `filters` object only for hard filters listed in
  Catalog capabilities. Enum filter values must exactly match the listed values.
  Numeric filters use an object with `min` and/or `max`.
- If the shopper says "only", "must be", "under", "over", or otherwise gives a
  strict constraint that is listed as a catalog hard filter, include that
  constraint in `filters`. Do not place unsupported constraints in `filters`.
- Media-only or descriptive media requests such as "what's in this look",
  "describe this outfit", "what am I wearing", or "what colors are here" must
  be answered from MEDIA ANALYSIS. Do not call search_catalog_tool and do not
  show catalog products unless the shopper explicitly asks to find, shop,
  recommend, compare, price-check, check availability, or add an item.
- Use at most one catalog search per user turn. Search with the complete user
  request rather than running many keyword probes. If one search is not enough,
  answer from the results you have or ask one concise clarifying question.
- A tool result is enough to produce a final answer. Do not keep searching for
  alternatives unless the shopper explicitly rejects the current result.
- Product-detail or research questions about a product already returned by
  search_catalog_tool should use get_product_details_tool with that
  PRODUCT_REF. Do not run another broad catalog search for known-product facts.
- When the shopper asks to add an item that has not already been searched in
  this conversation, call search_catalog_tool first, then call
  add_cart_item_tool with the selected PRODUCT_REF.
- If an image is attached, the current image is already available to
  search_catalog_tool. Use that tool for "this", "similar", and image-price
  refinement requests.
- If MEDIA ANALYSIS is present, use it as the visual/video understanding of
  the attached media. It can guide search_catalog_tool queries and follow-up
  pronoun resolution, but catalog tool results remain the source of truth for
  product names, prices, and availability.
- If video understanding is unavailable, say so plainly or ask the shopper for
  a text description. Do not invent visual details.
- Cart reads require get_cart_tool. Cart totals require view_cart_total_tool.
- Cart mutations require explicit shopper intent and must use add_cart_item_tool
  or remove_cart_item_tool. Never claim a cart mutation unless the tool reports
  success.
- Use PRODUCT_REF from search_catalog_tool when adding an item. Do not pass
  display names to add_cart_item_tool.
- Use PRODUCT_REF from search_catalog_tool when requesting product details. Do
  not pass display names to get_product_details_tool.
- Use CART_LINE_ID from CURRENT CART or get_cart_tool when removing an item. Do
  not guess cart line IDs from product names.
- If the shopper asks for anything under a budget without a product type,
  category, occasion, style, outfit goal, or image, ask one concise clarifying
  question instead of guessing.
- Persona or preference context is guidance only. The current shopper request
  wins when it conflicts with previous preferences.
- Keep final answers concise and grounded in tool results.
"""

    def _build_user_message(self, state: State, identity: RequestIdentity) -> str:
        return (
            f"REQUEST ID: {identity.request_id}\n"
            f"SESSION ID: {identity.session_id}\n"
            f"CONVERSATION ID: {identity.conversation_id}\n"
            f"CART ID: {identity.cart_id}\n\n"
            f"USER QUERY: {state.query}\n"
            f"IMAGE ATTACHED: {'yes' if state.image else 'no'}\n\n"
            f"MEDIA ATTACHED:\n{_format_media_summary(state.media)}\n\n"
            f"MEDIA ANALYSIS:\n{state.media_analysis or '(none)'}\n\n"
            f"CURRENT CART:\n{_format_cart(state.cart)}\n\n"
            f"RECENT DISCUSSION:\n{state.context or '(none)'}"
        )

    def _remember_products(
        self, identity: RequestIdentity, products: list[ProductSummary]
    ) -> None:
        refs = self._product_refs.setdefault(identity.conversation_id, {})
        for product in products:
            refs[product.product_id] = product
        while len(refs) > 50:
            refs.pop(next(iter(refs)))

    def _product_from_ref(
        self, identity: RequestIdentity, product_ref: str
    ) -> ProductSummary | None:
        return self._product_refs.get(identity.conversation_id, {}).get(
            (product_ref or "").strip()
        )

    def _read_cart(self, user_id: int) -> Cart:
        result = get_cart(GetCartInput(user_id=str(user_id)), self.config.memory_port)
        if not result.ok or result.cart is None:
            return Cart()
        return Cart(
            contents=[
                {
                    "cart_line_id": line.cart_line_id,
                    "product_id": line.product_id,
                    "item": line.display_name,
                    "amount": line.quantity,
                    "price": line.unit_price.amount if line.unit_price else None,
                }
                for line in result.cart.lines
            ]
        )

    def _load_memory(self, state: State, identity: RequestIdentity) -> None:
        start = time.monotonic()
        try:
            memory_response = requests.get(
                f"{self.config.memory_port}/user/{identity.context_user_id}/context",
                timeout=10,
            )
            memory_response.raise_for_status()
            cart = self._read_cart(identity.cart_user_id)
            state.context = memory_response.json().get("context") or ""
            state.cart = cart
        except requests.RequestException as exc:
            logger.error("Failed to load memory for Deep Agents turn: %s", exc)
            state.context = ""
            state.cart = Cart()
        state.timings["memory"] = time.monotonic() - start

    def _persist_context(self, state: State, identity: RequestIdentity) -> None:
        try:
            requests.post(
                f"{self.config.memory_port}/user/{identity.context_user_id}/context/replace",
                json={"new_context": state.context},
                timeout=10,
            )
        except requests.RequestException as exc:
            logger.error("Failed to persist Deep Agents context: %s", exc)

    def _check_safety(self, mode: str, user_id: int, text: str) -> bool:
        endpoint = "input" if mode == "input" else "output"
        try:
            response = requests.post(
                f"{self.config.rails_port}/rail/{endpoint}/check",
                json={"user_id": user_id, "query": text},
                timeout=10,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            logger.error("Guardrails %s check failed: %s", mode, exc)
            return True

        responses = payload.get("response") or []
        if not responses:
            return True
        return responses[0].get("content") == text

    def _updated_context(
        self,
        old_context: str,
        query: str,
        response: str,
        *,
        media_analysis: str = "",
    ) -> str:
        media_line = f"\nMedia analysis: {media_analysis}" if media_analysis else ""
        new_context = (
            f"{old_context}\nUser: {query}{media_line}\nAssistant: {response}"
        ).strip()
        max_chars = max(1000, int(self.config.memory_length))
        return new_context[-max_chars:]


def create_request_identity(
    *,
    legacy_user_id: int,
    session_id: str | None = None,
    conversation_id: str | None = None,
    cart_id: str | None = None,
) -> RequestIdentity:
    """Create scoped request identity while preserving legacy user_id behavior."""

    session = session_id or f"legacy-session-{legacy_user_id}"
    conversation = conversation_id or f"legacy-conversation-{legacy_user_id}"
    cart = cart_id or f"legacy-cart-{legacy_user_id}"
    return RequestIdentity(
        session_id=session,
        conversation_id=conversation,
        cart_id=cart,
        context_user_id=(
            _stable_numeric_id("conversation", conversation_id)
            if conversation_id
            else legacy_user_id
        ),
        cart_user_id=_stable_numeric_id("cart", cart_id) if cart_id else legacy_user_id,
        request_id=str(uuid.uuid4()),
    )


def _stable_numeric_id(namespace: str, value: str) -> int:
    digest = hashlib.sha256(f"{namespace}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:15], 16)


def _extract_final_text(result: Any) -> str:
    if isinstance(result, dict):
        messages = result.get("messages")
        if isinstance(messages, list):
            for message in reversed(messages):
                content = getattr(message, "content", None)
                if content is None and isinstance(message, dict):
                    content = message.get("content")
                text = _content_to_text(content)
                if text:
                    return text
        return _content_to_text(result.get("response")) or ""
    return _content_to_text(getattr(result, "content", result)) or ""


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts).strip()
    return ""


def _tool_strictness(value: str) -> str:
    return value if value in {"unspecified", "hard"} else "unspecified"


def _tool_search_mode(value: str | None) -> str | None:
    return value if value in {"text", "image", "hybrid"} else None


def _format_product(product: Any) -> str:
    text = product.attributes.get("catalog_text") if product.attributes else None
    if isinstance(text, str) and text.strip():
        return f"PRODUCT_REF: {product.product_id}\n{text.strip()}"
    price = f"\nPRICE: ${product.price.amount:.2f}" if product.price else ""
    return (
        f"PRODUCT_REF: {product.product_id}\n"
        f"{product.display_name} | {product.description}{price}"
    ).strip()


def _format_product_details(product: ProductSummary) -> str:
    lines = [
        f"PRODUCT_REF: {product.product_id}",
        f"NAME: {product.display_name}",
    ]
    if product.category:
        lines.append(f"CATEGORY: {product.category}")
    if product.brand:
        lines.append(f"BRAND: {product.brand}")
    if product.price:
        lines.append(f"PRICE: ${product.price.amount:.2f} {product.price.currency}")
    if product.description:
        lines.append(f"DESCRIPTION: {product.description}")
    catalog_text = product.attributes.get("catalog_text") if product.attributes else None
    if isinstance(catalog_text, str) and catalog_text.strip():
        lines.append("CATALOG FACTS:")
        lines.append(catalog_text.strip())
    return "\n".join(lines)


def _format_cart(cart: Cart) -> str:
    if not cart.contents:
        return "(empty)"
    lines = []
    for item in cart.contents:
        price = item.get("price")
        suffix = ""
        if price is not None:
            try:
                suffix = f" @ ${float(price):.2f}"
            except (TypeError, ValueError):
                suffix = ""
        cart_line_id = item.get("cart_line_id") or item.get("item", "")
        lines.append(
            f"- CART_LINE_ID: {cart_line_id} | "
            f"{item.get('amount', 1)} x {item.get('item', '')}{suffix}"
        )
    return "\n".join(lines)


def _format_cart_total(cart: Cart) -> str:
    if not cart.contents:
        return "Your cart is empty, so the total is $0.00."
    subtotal = 0.0
    missing = []
    lines = []
    for item in cart.contents:
        name = item.get("item", "")
        amount = int(item.get("amount") or 0)
        price = item.get("price")
        if price is None:
            missing.append(name)
            lines.append(f"- {amount} x {name}: price unavailable")
            continue
        line_total = float(price) * amount
        subtotal += line_total
        lines.append(f"- {amount} x {name} @ ${float(price):.2f} = ${line_total:.2f}")
    total = f"Cart total: ${subtotal:.2f}"
    if missing:
        total += f" excluding items without cached prices: {', '.join(missing)}"
    return "\n".join(lines + [total])


def _format_media_summary(media: list[dict[str, Any]]) -> str:
    if not media:
        return "(none)"
    counts: dict[str, int] = {}
    for item in media:
        media_type = str(item.get("type") or "unknown")
        counts[media_type] = counts.get(media_type, 0) + 1
    return ", ".join(f"{count} {media_type}(s)" for media_type, count in sorted(counts.items()))


def _cart_line_by_id(cart_line_id: str, cart: Cart) -> dict[str, Any] | None:
    if not cart.contents:
        return None
    target = (cart_line_id or "").strip()
    if not target:
        return None
    for item in cart.contents:
        if str(item.get("cart_line_id") or "").strip() == target:
            return item
    return None
