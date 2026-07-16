# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deep Agents SDK runtime for the shopping assistant."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, AsyncIterator
import uuid

from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, ConfigDict, Field, ValidationError
import requests

from .agenttypes import Cart, State
from .catalog_capabilities import (
    CatalogCapabilitiesClient,
    format_catalog_capabilities_for_prompt,
)
from .catalog_execution import execute_catalog_search
from .catalog_request import (
    CatalogSearchIntent,
    CatalogSearchPlan,
    build_catalog_search_plan,
)
from .commerce_tools import (
    add_cart_item,
    get_cart,
    get_product_details,
    remove_cart_item,
)
from .media_perception import MediaPerceptionClient
from shared.commerce_contracts import (
    AddCartItemInput,
    CatalogCapabilities,
    CommerceError,
    GetCartInput,
    GetProductDetailsInput,
    ProductDetail,
    ProductSummary,
    RemoveCartItemInput,
)


logger = logging.getLogger(__name__)

try:
    from deepagents.backends import FilesystemBackend as _FilesystemBackend
except Exception:  # pragma: no cover - dependency import is validated at runtime.
    _FilesystemBackend = None

_SHOPPER_SKILLS_ENV = "SHOPPER_SKILLS_ROOT"
_SHOPPER_SKILLS_SOURCE = "/shopper"
_SEARCH_RESULT_GROUNDING_NOTE = (
    "SEARCH_RESULT_GROUNDING_NOTE: Use search results for candidate names, prices, "
    "categories, image availability, and modest styling fit only. Treat product "
    "names as display names, not attribute evidence. Do not infer or group-claim "
    "length, color, print, material, care, construction, fit, comfort, weather, "
    "grass, gravel, or best-in-category performance from names or search snippets."
)
_PRODUCT_DETAIL_GROUNDING_NOTE = (
    "PRODUCT_DETAIL_GROUNDING_NOTE: This detail result exposes only "
    "the fields shown below. Material, care, dimensions, closures, fit, "
    "sizing, colorways, and outdoor performance are unavailable unless explicitly "
    "listed. Do not infer them from product names or prior marketing text."
)
_GROUNDING_EDITOR_SYSTEM_PROMPT = """You are a final response editor for a retail shopping assistant.

Rewrite the draft response only as needed so it is grounded in TOOL EVIDENCE
and CURRENT CART. Keep the shopper's requested task and any successful cart
action intact.

Rules:
- Return only the final shopper-facing response text.
- Do not add products, prices, cart actions, or product facts absent from TOOL
  EVIDENCE or CURRENT CART.
- Remove PRODUCT_REF, CART_LINE_ID, tool names, and internal IDs.
- Remove internal skill, mode, evaluator, judge, cache, backend, tool-evidence,
  structured-field, and data-layer language. Use shopper-safe phrasing such as
  "I don't have fabric or care details available for that item."
- If the draft says "product detail tool", "catalog detail tool", "the tool
  requires", or similar internal mechanics, rewrite it into shopper-safe
  language without the word "tool".
- If a product appears only in search results, you may state only its name,
  price, category/role, image availability, and a modest styling reason.
- Treat product names as display names, not proof of length, color, print,
  material, construction, fit, care, or vibe. Do not say a product is solid,
  floral, gingham, maxi, knee-length, woven, structured, neutral, lightweight,
  polished, or dressier unless that attribute appears in product-detail evidence.
- Material, care, dimensions, pockets, closures, fit, comfort, and outdoor
  practicality claims require matching product-detail evidence and a direct
  shopper need for that fact.
- Group claims such as "all are maxi length", "both are cotton", "the lightest",
  "most polished", or "best for heat" require product-detail evidence for every
  item included in that claim. Remove the claim if any item lacks that support.
- Do not say an item is stable on grass or gravel, water-resistant, bug-safe,
  all-day comfortable, maximally breathable, or best-in-category unless the
  evidence explicitly says that exact claim.
- Do not convert indirect evidence into outdoor surface performance. If the
  evidence says flat sole, ankle strap, linen, cotton, or elastic waistband,
  state only that fact when needed; do not add grass, gravel, outdoor-surface,
  heat, or all-evening performance claims. Avoid phrases such as "works well
  for outdoor surfaces"; use "a flat shoe option" or "fits the practical
  direction" instead.
- The shopper and Judge see only the final answer, not hidden tool output. Avoid
  long product-spec dumps; keep catalog facts item-specific and visibly modest.
- If image evidence is available, do not say images are unavailable or that you
  cannot show them. Say the product image should appear with the result, or
  simply answer the comparison.
- Preserve exact cart totals and cart contents when they are present in CURRENT
  CART or tool evidence.
- If the draft is already compliant, return it unchanged.
"""
_EXCLUDED_DEEP_AGENT_TOOLS = frozenset(
    {"write_todos", "ls", "write_file", "edit_file", "glob", "grep", "execute"}
)
_PRODUCT_NAME_STOPWORDS = frozenset({"a", "an", "and", "in", "of", "the", "to", "with"})
_INTERNAL_SHOPPER_REPLACEMENTS = (
    ("The product detail tool doesn't return", "I don't have"),
    ("the product detail tool doesn't return", "I don't have"),
    ("The catalog detail tool doesn't return", "I don't have"),
    ("the catalog detail tool doesn't return", "I don't have"),
    ("The product detail tool does not return", "I don't have"),
    ("the product detail tool does not return", "I don't have"),
    ("The catalog detail tool does not return", "I don't have"),
    ("the catalog detail tool does not return", "I don't have"),
    ("the product detail tool", "the product details I can access"),
    ("the catalog detail tool", "the product details I can access"),
    ("because the tool requires", "because I need"),
    ("The tool requires", "I need"),
    ("the tool requires", "I need"),
)


class SearchCatalogToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    semantic_query: str = Field(
        default="",
        description=(
            "Soft or descriptive product search text. Include product type, style, "
            "occasion, material, visual descriptors, and other preferences that "
            "may be ranked semantically. Do not rely on this field for must-have "
            "requirements; put every must-have in required_constraints."
        ),
    )
    required_constraints: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Every shopper must-have as a structured field and value, including "
            "requirements that Catalog capabilities mark semantic/detail-only or "
            "do not advertise. Advertised numeric constraints use objects like "
            "{'max': 100}; advertised enum constraints use exact listed values. "
            "Unsupported requirements are preserved so validation can refuse the "
            "search instead of silently weakening it."
        ),
    )
    search_mode: str | None = Field(
        default=None,
        description="Optional search mode from Catalog capabilities.",
    )


class AddCartItemsToolItemInput(BaseModel):
    product_ref: str = Field(
        ...,
        min_length=1,
        description="PRODUCT_REF returned by search_catalog_tool in this conversation.",
    )
    quantity: int = Field(
        default=1,
        ge=1,
        description="Quantity of this product to add.",
    )
    expected_display_name: str | None = Field(
        default=None,
        description=(
            "Shopper-facing product name the agent intends to add. When the "
            "shopper explicitly names the product, copy that exact product name."
        ),
    )


class AddCartItemsToolInput(BaseModel):
    items: list[AddCartItemsToolItemInput] = Field(
        ...,
        min_length=1,
        description=(
            "One or more catalog products to add. Each product must use a "
            "PRODUCT_REF returned by search_catalog_tool."
        ),
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

    def catalog_capabilities(self) -> CatalogCapabilities:
        """Return the process-lifecycle catalog capability contract."""

        return self._catalog_capabilities.get()

    async def astream(
        self, state: State, identity: RequestIdentity
    ) -> AsyncIterator[str]:
        output = await self._run_turn(state, identity)
        products = output.product_results or []
        if products:
            yield json.dumps(
                {"type": "products", "payload": products, "timestamp": time.time()}
            )
        images = output.retrieved or {}
        yield json.dumps({"type": "images", "payload": images, "timestamp": time.time()})
        if output.response:
            yield json.dumps(
                {"type": "content", "payload": output.response, "timestamp": time.time()}
            )
        yield json.dumps(
            {
                "type": "metrics",
                "payload": {
                    "timings": output.timings,
                    "total_seconds": sum(output.timings.values()),
                    "token_usage": _normalized_token_usage(output.token_usage),
                    "model_usage": output.model_usage,
                },
                "timestamp": time.time(),
            }
        )

    async def ainvoke(self, state: State, identity: RequestIdentity) -> dict[str, Any]:
        output = await self._run_turn(state, identity)
        return {
            "response": output.response,
            "images": output.retrieved or {},
            "cart": output.cart.model_dump(mode="json"),
            "timings": output.timings,
            "token_usage": _normalized_token_usage(output.token_usage),
            "model_usage": output.model_usage,
        }

    async def _run_turn(self, state: State, identity: RequestIdentity) -> State:
        start = time.monotonic()
        state.user_id = identity.context_user_id
        self._load_memory(state, identity)

        if state.guardrails:
            safety_start = time.monotonic()
            input_safe, input_check_ok = self._check_safety(
                "input",
                identity.context_user_id,
                state.query,
            )
            state.timings["safety_input"] = time.monotonic() - safety_start
            _record_safety_model_usage(state, "input", ok=input_check_ok)
            if not input_safe:
                state.response = self.config.unsafe_message
                state.timings["deepagents"] = time.monotonic() - start
                return state

        media_start = time.monotonic()
        state.media_analysis = await self._media_perception.analyze(state)
        if state.media:
            state.timings["media_perception"] = time.monotonic() - media_start
            _record_media_model_usage(state, self.config)
        if _should_short_circuit_media_failure(state):
            state.response = _media_failure_response(state.media_analysis)
            state.timings["deepagents"] = time.monotonic() - start
            return state

        turn_capabilities = await asyncio.to_thread(self._catalog_capabilities.get)
        agent = self._create_agent(state, identity, turn_capabilities)
        input_message = self._build_user_message(state, identity)
        try:
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": input_message}]},
                config={
                    "configurable": {"thread_id": identity.conversation_id},
                    "recursion_limit": self.config.deepagents_recursion_limit,
                },
            )
            draft_response = _extract_final_text(result)
            state.token_usage = _collect_token_usage(result)
            state.response = self._rewrite_response_for_grounding(
                state,
                result,
                draft_response,
            )
        except Exception:  # noqa: BLE001 - keep endpoint resilient.
            logger.exception("DeepAgentsRuntime failed")
            self._reset_agent_thread(identity)
            fallback_response = _partial_product_results_response(state)
            state.response = fallback_response or (
                "I encountered an error while helping with your shopping request. "
                "Please try again."
            )
            _record_language_model_failure(state)
            state.timings["deepagents_error"] = time.monotonic() - start
            if fallback_response:
                state.context = self._updated_context(
                    state.context,
                    state.query,
                    state.response,
                    media_analysis=state.media_analysis,
                )
                self._persist_context(state, identity)
            return state

        if state.guardrails:
            safety_start = time.monotonic()
            output_safe, output_check_ok = self._check_safety(
                "output",
                identity.context_user_id,
                state.response,
            )
            state.timings["safety_output"] = time.monotonic() - safety_start
            _record_safety_model_usage(state, "output", ok=output_check_ok)
            if not output_safe:
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

    def _create_agent(
        self,
        state: State,
        identity: RequestIdentity,
        turn_capabilities: CatalogCapabilities | None = None,
    ):
        from deepagents import (
            GeneralPurposeSubagentProfile,
            HarnessProfile,
            create_deep_agent,
            register_harness_profile,
        )
        from langchain_core.tools import tool

        # One cached lifecycle contract is authoritative for prompt construction
        # and deterministic validation. Catalog requests are revalidated by the
        # active catalog service before execution.
        if turn_capabilities is None:
            turn_capabilities = self._catalog_capabilities.get()

        if not self._profile_registered:
            register_harness_profile(
                f"openai:{self.config.llm_name}",
                HarnessProfile(
                    excluded_tools=_EXCLUDED_DEEP_AGENT_TOOLS,
                    general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
                ),
            )
            self._profile_registered = True

        skills_backend = self._create_skills_backend()
        retrieved: dict[str, str] = {}
        state.retrieved = retrieved
        self._append_cached_cart_images(retrieved, state.cart, identity)
        catalog_searches_this_turn = 0
        product_detail_reads_this_turn = 0

        @tool(args_schema=SearchCatalogToolInput, return_direct=False)
        def search_catalog_tool(
            semantic_query: str,
            required_constraints: dict[str, Any] | None = None,
            search_mode: str | None = None,
        ) -> str:
            """Validate must-haves, then execute grounded product discovery."""

            nonlocal catalog_searches_this_turn
            if catalog_searches_this_turn >= self.config.max_catalog_searches_per_turn:
                return (
                    "STOP_TOOL_USE: Catalog search limit reached for this turn. "
                    "Do not call more tools this turn. Use the products already "
                    "returned in this turn to answer concisely, or ask one concise "
                    "clarifying question if the available products are not enough."
                )
            catalog_searches_this_turn += 1

            capabilities = turn_capabilities
            if capabilities.catalog_id == "unavailable" and not capabilities.filters:
                return "Catalog search is unavailable. Please try again."

            intent = CatalogSearchIntent(
                semantic_query=semantic_query,
                required_constraints=(
                    required_constraints
                    if isinstance(required_constraints, dict)
                    else {}
                ),
                search_mode=_tool_search_mode(search_mode),
            )
            plan = build_catalog_search_plan(
                intent,
                capabilities,
                has_image=bool(state.image),
                top_k=self.config.top_k_retrieve,
            )
            if not plan.should_search:
                if plan.constraint_issues:
                    return (
                        "The requested catalog requirement cannot be enforced: "
                        + "; ".join(plan.constraint_issues)
                        + ". Ask the shopper to relax it or use an advertised filter."
                    )
                if plan.no_search_reason == "image_search_unavailable":
                    return (
                        "Image search is not available for the active catalog. "
                        "Ask the shopper to describe what they want to find."
                    )
                if plan.no_search_reason == "unsupported_search_mode":
                    return (
                        "The requested search mode is not available for the active "
                        "catalog. Ask the shopper to use an advertised mode."
                    )
                if plan.no_search_reason == "missing_image_for_search_mode":
                    return (
                        "That search mode requires an attached image. Ask the shopper "
                        "to attach one or use text search."
                    )
                return "Catalog search requires a query or image."

            search_start = time.monotonic()
            execution = execute_catalog_search(
                plan,
                self.config.retriever_port,
                image_base64=state.image,
                timeout_seconds=self.config.catalog_search_timeout_seconds,
            )
            state.timings["catalog_search"] = time.monotonic() - search_start
            result = execution.result
            _record_catalog_model_usage(state, plan, result.ok)
            if not result.ok:
                return result.error.message if result.error else "Catalog search failed."
            if not result.products:
                return "No matching catalog products were found."

            self._remember_products(identity, result.products)
            _append_product_results(state, result.products)
            lines = [_SEARCH_RESULT_GROUNDING_NOTE]
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
            self._append_cached_cart_images(retrieved, cart, identity)
            return _format_cart(cart)

        @tool(return_direct=False)
        def get_product_details_tool(product_ref: str) -> str:
            """Read details for a PRODUCT_REF returned by search_catalog_tool."""

            nonlocal product_detail_reads_this_turn
            if (
                product_detail_reads_this_turn
                >= self.config.max_product_detail_reads_per_turn
            ):
                return (
                    "STOP_TOOL_USE: Product-detail read limit reached for this "
                    "turn. Do not call more tools this turn. Answer now from the "
                    "details already read and keep any other products to names, "
                    "prices, categories, image availability, and styling role."
                )
            product_detail_reads_this_turn += 1

            cached_product = self._product_from_ref(identity, product_ref)
            if cached_product is None:
                return (
                    f"No product with PRODUCT_REF '{product_ref}' is available. "
                    "Search the catalog first and use the PRODUCT_REF from the result."
                )
            detail_result = get_product_details(
                GetProductDetailsInput(product_id=cached_product.product_id),
                self.config.retriever_port,
                timeout_seconds=self.config.catalog_search_timeout_seconds,
            )
            if not detail_result.ok or detail_result.product is None:
                return _product_detail_failure_message(
                    detail_result.error,
                    cart_validation=False,
                )
            product = detail_result.product
            if not _same_product_display_name(
                product.display_name,
                cached_product.display_name,
            ):
                return (
                    "That product reference now resolves to a different item. "
                    "Search the catalog again before using its details."
                )
            if product.image_url:
                retrieved[product.display_name] = product.image_url
            return _format_product_details(product)

        @tool(args_schema=AddCartItemsToolInput, return_direct=False)
        def add_cart_items_tool(items: list[AddCartItemsToolItemInput]) -> str:
            """Add one or more catalog items by PRODUCT_REF from prior search results."""

            try:
                requested_items = _normalize_cart_add_tool_items(items)
            except ValueError as exc:
                return f"Cart add failed: {exc}"
            if not requested_items:
                return "Cart add failed: provide at least one PRODUCT_REF to add."

            refs = self._product_refs.get(identity.conversation_id, {})
            resolved: list[tuple[str, ProductSummary, int]] = []
            failed: list[str] = []
            blocked: list[str] = []
            for product_ref, request in requested_items.items():
                product = self._product_from_ref(identity, product_ref)
                if product is None:
                    failed.append(
                        f"- PRODUCT_REF '{product_ref}': Search the catalog first "
                        "and use the PRODUCT_REF from the result."
                    )
                    continue
                expected_name = request.get("expected_display_name") or ""
                if expected_name and not _same_product_display_name(
                    expected_name,
                    product.display_name,
                ):
                    blocked.append(
                        f"- PRODUCT_REF '{product_ref}': expected "
                        f"'{expected_name}', but that ref resolves to "
                        f"'{product.display_name}'. Use the matching PRODUCT_REF "
                        "for the intended product before adding."
                    )
                    continue
                active_detail = get_product_details(
                    GetProductDetailsInput(product_id=product.product_id),
                    self.config.retriever_port,
                    timeout_seconds=self.config.catalog_search_timeout_seconds,
                )
                if not active_detail.ok or active_detail.product is None:
                    failed.append(
                        f"- PRODUCT_REF '{product_ref}': "
                        + _product_detail_failure_message(
                            active_detail.error,
                            cart_validation=True,
                        )
                    )
                    continue
                if not _same_product_display_name(
                    active_detail.product.display_name,
                    product.display_name,
                ):
                    blocked.append(
                        f"- PRODUCT_REF '{product_ref}': That reference now "
                        "resolves to a different product. Search again and use "
                        "the new PRODUCT_REF before adding it."
                    )
                    continue
                resolved.append(
                    (product_ref, active_detail.product, int(request["quantity"]))
                )

            blocked.extend(
                _cart_add_scope_failures(
                    state.query,
                    [(product_ref, product) for product_ref, product, _ in resolved],
                    refs.values(),
                )
            )
            if blocked:
                state.cart = self._read_cart(identity.cart_user_id)
                self._append_cached_cart_images(retrieved, state.cart, identity)
                return _format_cart_add_result([], failed + blocked, state.cart)

            added: list[str] = []
            for product_ref, product, quantity in resolved:
                result = add_cart_item(
                    AddCartItemInput(
                        user_id=str(identity.cart_user_id),
                        product_id=product.product_id,
                        display_name=product.display_name,
                        quantity=quantity,
                        unit_price=product.price,
                        image_url=product.image_url,
                        idempotency_key=(
                            f"{identity.request_id}:add:{product.product_id}:{quantity}"
                        ),
                    ),
                    self.config.memory_port,
                )
                if result.ok:
                    added.append(
                        f"- {quantity} x {product.display_name} "
                        f"(PRODUCT_REF: {product.product_id})"
                    )
                else:
                    message = result.error.message if result.error else "Cart add failed."
                    failed.append(f"- PRODUCT_REF '{product_ref}': {message}")

            state.cart = self._read_cart(identity.cart_user_id)
            self._append_cached_cart_images(retrieved, state.cart, identity)
            return _format_cart_add_result(added, failed, state.cart)

        @tool(return_direct=False)
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
            self._append_cached_cart_images(retrieved, state.cart, identity)
            if result.ok:
                return result.message or f"Removed {quantity} {line['item']} from cart."
            return result.error.message if result.error else "Cart remove failed."

        @tool(return_direct=False)
        def view_cart_total_tool() -> str:
            """Compute the current cart total from cached cart line prices."""

            cart = self._read_cart(identity.cart_user_id)
            state.cart = cart
            self._append_cached_cart_images(retrieved, cart, identity)
            return _format_cart_total(cart)

        return create_deep_agent(
            model=self._create_chat_model(),
            tools=[
                search_catalog_tool,
                get_product_details_tool,
                get_cart_tool,
                add_cart_items_tool,
                remove_cart_item_tool,
                view_cart_total_tool,
            ],
            system_prompt=self._system_prompt(turn_capabilities),
            skills=[_SHOPPER_SKILLS_SOURCE] if skills_backend is not None else None,
            backend=skills_backend,
            checkpointer=self._checkpointer,
        )

    def _reset_agent_thread(self, identity: RequestIdentity) -> None:
        try:
            self._checkpointer.delete_thread(identity.conversation_id)
        except Exception as exc:  # pragma: no cover - cleanup is best effort.
            logger.warning("Could not reset Deep Agents thread after failure: %s", exc)

    def _create_chat_model(self):
        from langchain_openai import ChatOpenAI

        api_key_env = getattr(self.config, "llm_api_key_env", None)
        api_key = os.environ.get(api_key_env, "") if api_key_env else "not-needed"
        return ChatOpenAI(
            model=self.config.llm_name,
            base_url=self.config.llm_port,
            api_key=api_key or "not-needed",
            temperature=0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

    def _rewrite_response_for_grounding(
        self,
        state: State,
        result: Any,
        draft_response: str,
    ) -> str:
        if not draft_response:
            return draft_response
        if not getattr(self.config, "grounding_rewrite_enabled", True):
            return _scrub_internal_shopper_language(draft_response)

        evidence = _collect_tool_grounding_evidence(
            result,
            max_chars=getattr(self.config, "grounding_rewrite_max_evidence_chars", 12000),
        )
        if not evidence:
            return _scrub_internal_shopper_language(draft_response)

        start = time.monotonic()
        prompt = (
            f"USER QUERY:\n{state.query}\n\n"
            f"CURRENT CART:\n{_format_cart(state.cart)}\n\n"
            f"AVAILABLE IMAGES:\n{_format_retrieved_images(state.retrieved)}\n\n"
            f"TOOL EVIDENCE:\n{evidence}\n\n"
            f"DRAFT RESPONSE:\n{draft_response}"
        )
        try:
            rewrite_result = self._create_chat_model().invoke(
                [
                    {"role": "system", "content": _GROUNDING_EDITOR_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]
            )
        except Exception:  # noqa: BLE001 - response editor must fail open.
            logger.exception("Grounding response editor failed")
            state.timings["grounding_rewrite"] = time.monotonic() - start
            _add_model_usage(
                state,
                "app_llm_grounding_editor",
                status="failed",
                calls=1,
                detail="Final response grounding rewrite failed open",
            )
            return draft_response

        state.timings["grounding_rewrite"] = time.monotonic() - start
        _add_model_usage(
            state,
            "app_llm_grounding_editor",
            status="used",
            calls=1,
            detail="Final response grounding rewrite",
        )
        state.token_usage = _merge_token_usage(
            state.token_usage,
            _collect_token_usage(rewrite_result),
        )

        rewritten = _content_to_text(_value(rewrite_result, "content"))
        if not rewritten:
            rewritten = _content_to_text(rewrite_result)
        return _scrub_internal_shopper_language(rewritten or draft_response)

    def _create_skills_backend(self):
        if _FilesystemBackend is None:
            logger.warning("Deep Agents filesystem backend unavailable; shopper skills disabled.")
            return None
        skills_root = self._shopper_skills_root()
        if skills_root is None:
            logger.warning("Shopper skills root not found; Deep Agents skills disabled.")
            return None
        return _FilesystemBackend(root_dir=skills_root, virtual_mode=True)

    def _shopper_skills_root(self) -> Path | None:
        configured_root = os.environ.get(_SHOPPER_SKILLS_ENV)
        candidates = [Path(configured_root)] if configured_root else []
        candidates.append(Path(__file__).resolve().parents[1] / "skills")

        for candidate in candidates:
            if (candidate / "shopper").is_dir():
                return candidate
        return None

    def _system_prompt(self, capabilities: CatalogCapabilities) -> str:
        catalog_context = format_catalog_capabilities_for_prompt(capabilities)
        return f"""You are a retail shopping assistant for the products advertised by the active catalog.

Use tools for catalog facts and cart actions. Do not invent product names,
prices, availability, materials, care instructions, tax, shipping, stock
status, delivery dates, or cart changes.
PRODUCT_REF and CART_LINE_ID are internal identifiers for tool calls. Do not
show them to shoppers in normal responses.
Do not expose skill names, tool names, entry-mode names, evaluator/judge names,
cache/backend details, structured-field labels, or internal data-layer language
to shoppers. If a detail is unavailable, say it plainly, for example: "I don't
have fabric or care details available for that item."
Do not upgrade shopper assumptions, preference language, or earlier styling
inferences into catalog facts. Separate confirmed product facts from styling
judgment, and keep outfit-wide material or comfort claims item-specific unless
every included piece is supported by tool evidence.
Do not group leather, rubber, metal, or generic canvas under "natural fibers";
attribute materials item by item.
Outdoor-practicality claims require exact support: do not say products are
stable on grass or gravel, water-resistant, all-day comfortable, weather-safe,
secure for a full event, or good for outdoor surfaces unless the catalog says
so. With indirect evidence, state only the confirmed construction detail and
keep the styling judgment separate.
Do not convert sole or strap facts into surface guarantees. Rubber sole means
rubber sole; ankle strap means ankle strap. Do not add grass, gravel, weather,
outdoor-surface performance, all-day comfort, maximum breathability, or
best-in-category performance claims unless those claims are directly supported
by product details.

Catalog capabilities:
{catalog_context}

Rules:
- Product discovery, product recommendations, budget filters, and image-similar
  shopping require search_catalog_tool.
- Put product meaning and soft or descriptive preferences in
  search_catalog_tool.semantic_query. Semantic relevance ranks candidates but
  cannot guarantee a must-have requirement.
- Put every shopper must-have in `required_constraints`, including requirements
  whose fields are semantic/detail-only or absent from Catalog capabilities. Do
  not omit an unsupported must-have or rely on semantic search to enforce it;
  deterministic validation must refuse that search instead of weakening it.
- For required constraints advertised as hard filters, enum values must exactly
  match listed values and numeric values use an object with `min` and/or `max`.
- Media-only or descriptive media requests such as "what's in this look",
  "describe this outfit", "what am I wearing", or "what colors are here" must
  be answered from MEDIA ANALYSIS. Do not call search_catalog_tool and do not
  show catalog products unless the shopper explicitly asks to find, shop,
  recommend, compare, price-check, check availability, or add an item.
- Use at most {self.config.max_catalog_searches_per_turn} catalog searches per
  user turn. For outfit requests with multiple required item types, run one
  focused search per required item type, then stop and synthesize from those
  results.
- Use at most {self.config.max_product_detail_reads_per_turn} product-detail
  reads per user turn. Product details are for direct product fact questions,
  cart or comparison follow-ups, or already-shortlisted items; they are not
  required for an initial no-anchor outfit recommendation.
- If a tool returns STOP_TOOL_USE, stop tool calling immediately and produce the
  best concise shopper-facing answer from the evidence already available.
- A tool result is enough to produce a final answer. Once you have at least one
  plausible product for each required item type, answer from those results. Do
  not keep searching for alternatives unless the shopper explicitly rejects the
  current result.
- Product-detail or research questions about a product already returned by
  search_catalog_tool should use get_product_details_tool with that
  PRODUCT_REF. Do not run another broad catalog search for known-product facts.
- Initial recommendations should use product name, price, category or role,
  and one styling reason. Do not enumerate materials, dimensions, pockets,
  closures, care, or construction details unless the shopper asks for those
  details and you have called get_product_details_tool.
- Search-only product names are display names, not confirmed attributes. Do not
  parse length, color, print, material, construction, fit, care, or formality
  from descriptive names unless product details confirm the attribute. You may
  say "candidate" or "could be worth checking" and offer to pull details.
- Do not make group-level claims such as "all are maxi length", "both are
  cotton", "the lightest", "most polished", or "best for heat" unless every
  item in the group has product-detail evidence supporting that exact claim.
- For no-anchor outfit building, do not call product details just to make the
  outfit sound richer. Search by the needed item roles, choose a coherent set,
  and keep the rationale to color, proportion, formality, silhouette, and
  shopper goal.
- If the shopper mentions outdoor practicality in a broad outfit request,
  prefer searched categories that naturally fit the situation, such as flat
  shoes or a light layer, but do not state product material, breathability,
  ground stability, outdoor-surface performance, heat performance, or
  all-evening comfort unless the shopper asks a direct product-specific
  question and details support it.
- Product comparison tables, material claims, dimensions, pocket/closure
  details, care/washability answers, comfort claims, and outdoor-practicality
  claims require get_product_details_tool for each relevant PRODUCT_REF before
  finalizing the answer. If you have only search results, keep the answer to
  names, prices, and brief candidate fit.
- Even after product details, compare only confirmed construction facts for
  surface or weather concerns: lower heel versus higher heel, strap versus no
  strap, rubber sole versus unspecified sole, zip closure versus open top. Do
  not state the resulting performance on grass, gravel, rain, bugs, spills, or
  outdoor ground unless product details explicitly state it.
- Shopper wording is not product evidence. If the shopper mentions an
  unverified attribute such as heel shape, material, colorway, fit, or care,
  verify it with tools or refer to it as the shopper's preference, not as a
  catalog fact.
- When the shopper asks to add an item that has not already been searched in
  this conversation, call search_catalog_tool first, then call
  add_cart_items_tool with a one-item list containing the selected PRODUCT_REF.
- If an image is attached, the current image is already available to
  search_catalog_tool. Use that tool for "this", "similar", and image-price
  refinement requests.
- If MEDIA ANALYSIS is present, use it as the visual/video understanding of
  the attached media. It can guide search_catalog_tool queries and follow-up
  pronoun resolution, but catalog results remain the source of truth for
  product names, prices, and availability.
- If MEDIA ANALYSIS says media analysis failed, VLM authentication failed, the
  VLM is unavailable, or video understanding is not configured, say so plainly.
  Do not infer video-similar products from the media; ask the shopper for a
  text description or search only from explicit text in the shopper request.
  If an image is attached, image embedding search through search_catalog_tool is
  still available even when MEDIA ANALYSIS is unavailable.
- Cart reads require get_cart_tool. Cart totals require view_cart_total_tool.
- Cart mutations require explicit shopper intent and must use add_cart_items_tool
  or remove_cart_item_tool. Never claim a cart mutation unless the tool reports
  success.
- Cart mutation scope must match the shopper's explicit add or remove request.
  Selection, approval, or styling preference is not cart intent by itself.
  If the shopper asks to "add those", add only the items named in that add
  request or its direct antecedent. Do not add earlier anchor, core outfit, or
  optional pieces unless the shopper explicitly includes them in the cart
  request.
- For an explicit cart swap, finish the whole swap before the final response:
  remove the rejected cart line, add the selected replacement when a valid
  PRODUCT_REF is already available, then summarize the updated cart. If the
  replacement has not been searched in this conversation, search first.
- If cart mutation scope is ambiguous, ask one concise clarification before
  calling any cart mutation tool. Example: "Do you want me to add just the bag,
  layer, and earrings, or the full outfit including the dress and sandals?"
- For cart styling requests, inspect CURRENT CART or call get_cart_tool first.
  Do not search for products already named as cart contents just to verify them.
  If the cart is empty but the shopper names items, say you do not see those
  items in the cart yet, then give provisional styling advice from the named
  items without claiming cart truth. Search at most once for a missing piece
  only after identifying the gap.
- Use PRODUCT_REF from search_catalog_tool when adding items. Do not pass
  display names as product_ref values to add_cart_items_tool. Include
  expected_display_name for each item so the tool can verify that the selected
  PRODUCT_REF resolves to the shopper-facing product name you intend to add.
- When the shopper asks to add multiple selected products, call
  add_cart_items_tool once with an item list. The tool may report partial
  success; the final answer must clearly distinguish added items from failures.
- Use PRODUCT_REF from search_catalog_tool when requesting product details. Do
  not pass display names to get_product_details_tool.
- Use internal identifiers only in tool calls. Do not expose PRODUCT_REF or
  CART_LINE_ID in customer-facing responses.
- Use CART_LINE_ID from CURRENT CART or get_cart_tool when removing an item. Do
  not guess cart line IDs from product names.
- If the shopper asks for anything under a budget without a product type,
  category, occasion, style, outfit goal, or image, ask one concise clarifying
  question instead of guessing.
- Tax, shipping fees, delivery dates, and real-time stock or inventory status
  are not available through the current tools. If asked, say that plainly and
  direct the shopper to checkout or the retailer product page. Do not treat a
  catalog result as proof that an item is in stock or ready to ship.
- Persona or preference context is guidance only. The current shopper request
  wins when it conflicts with previous preferences.
- Keep final answers concise and grounded in tool results. Attribute materials,
  comfort, construction, and outdoor-practicality claims to the specific items
  that support them instead of making unsupported whole-outfit claims. Avoid
  guarantee language such as "will stay comfortable all evening" unless the
  catalog evidence supports the guarantee. Before finalizing, remove or soften
  unsupported phrases about grass, gravel, water resistance, all-day comfort,
  maximum breathability, or best-in-category performance.
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

    def _append_cached_cart_images(
        self,
        retrieved: dict[str, str],
        cart: Cart,
        identity: RequestIdentity,
    ) -> None:
        if not cart.contents:
            return
        refs = self._product_refs.get(identity.conversation_id, {})
        if not refs:
            return

        products_by_key: dict[str, ProductSummary] = {}
        for product in refs.values():
            products_by_key[product.product_id] = product
            products_by_key[product.display_name] = product

        for item in cart.contents:
            product = products_by_key.get(str(item.get("product_id") or ""))
            if product is None:
                product = products_by_key.get(str(item.get("item") or ""))
            if product is not None and product.image_url:
                retrieved[product.display_name] = product.image_url

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

    def _check_safety(self, mode: str, user_id: int, text: str) -> tuple[bool, bool]:
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
            return True, False

        responses = payload.get("response") or []
        if not responses:
            return True, True
        return responses[0].get("content") == text, True

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


def _append_product_results(state: State, products: list[ProductSummary]) -> None:
    existing_ids = {
        str(product.get("product_id") or "")
        for product in state.product_results
        if isinstance(product, dict)
    }
    for product in products:
        payload = product.model_dump(mode="json")
        product_id = str(payload.get("product_id") or "")
        if product_id and product_id in existing_ids:
            continue
        state.product_results.append(payload)
        if product_id:
            existing_ids.add(product_id)


def _partial_product_results_response(state: State) -> str:
    products = [
        product for product in state.product_results if isinstance(product, dict)
    ]
    if not products:
        return ""

    lines = [
        "I found these grounded catalog options so far:",
        "",
    ]
    seen: set[str] = set()
    for product in products[:8]:
        name = str(product.get("display_name") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        parts = [f"**{name}**"]
        category = str(product.get("category") or "").strip()
        if category:
            parts.append(category)
        price = _product_result_price(product)
        if price:
            parts.append(price)
        lines.append("- " + " — ".join(parts))

    if len(lines) <= 2:
        return ""
    lines.extend(
        [
            "",
            (
                "I do not want to overstate outdoor performance, material, care, "
                "or fit details without a completed detail check. I can continue "
                "from these options or narrow one piece at a time."
            ),
        ]
    )
    return "\n".join(lines)


def _product_result_price(product: dict[str, Any]) -> str:
    price = product.get("price")
    if isinstance(price, dict):
        amount = price.get("amount")
        currency = str(price.get("currency") or "USD")
    else:
        amount = price
        currency = "USD"
    if not isinstance(amount, (int, float)):
        return ""
    return f"${float(amount):.2f} {currency}"


def _should_short_circuit_media_failure(state: State) -> bool:
    if not state.media or not state.media_analysis:
        return False
    if not any(str(item.get("type") or "").lower() == "video" for item in state.media):
        return False
    if not _media_analysis_unavailable(state.media_analysis):
        return False
    return _query_depends_on_media(state.query)


def _record_media_model_usage(state: State, config: Any) -> None:
    if not getattr(config, "vlm_enabled", False) or not getattr(config, "vlm_name", None):
        _add_model_usage(state, "vlm", status="disabled", calls=0)
        return

    status = "failed" if _media_analysis_unavailable(state.media_analysis) else "used"
    _add_model_usage(
        state,
        "vlm",
        status=status,
        calls=1,
        detail="Attached media understanding",
    )


def _record_catalog_model_usage(
    state: State,
    plan: CatalogSearchPlan,
    ok: bool,
) -> None:
    status = "used" if ok else "failed"
    uses_image_endpoint = bool(state.image) and plan.search_mode in {"image", "hybrid"}
    uses_text_embedding = bool(plan.semantic_queries) or uses_image_endpoint

    if uses_text_embedding:
        _add_model_usage(
            state,
            "text_embedding",
            status=status,
            calls=1,
            detail="Catalog text/vector retrieval",
        )
    if uses_image_endpoint:
        _add_model_usage(
            state,
            "image_embedding",
            status=status,
            calls=1,
            detail="Catalog image similarity retrieval",
        )


def _record_language_model_failure(state: State) -> None:
    _add_model_usage(
        state,
        "app_llm",
        status="failed",
        calls=1,
        detail="Planning, tool use, and response generation failed",
    )


def _record_safety_model_usage(state: State, mode: str, *, ok: bool = True) -> None:
    status = "used" if ok else "failed"
    detail = "Input and output safety checks" if ok else "Guardrails check failed open"
    if mode == "input":
        _add_model_usage(
            state,
            "content_safety",
            status=status,
            calls=1,
            detail=detail,
        )
        _add_model_usage(
            state,
            "topic_control",
            status=status,
            calls=1,
            detail="Input topic check" if ok else "Guardrails topic check failed open",
        )
        return

    _add_model_usage(
        state,
        "content_safety",
        status=status,
        calls=1,
        detail=detail,
    )


def _add_model_usage(
    state: State,
    role: str,
    *,
    status: str,
    calls: int,
    detail: str = "",
) -> None:
    existing = state.model_usage.get(role, {})
    existing_calls = _safe_int(existing.get("calls"))
    existing_status = str(existing.get("status") or "")
    merged_status = _merged_model_usage_status(existing_status, status)
    existing_detail = str(existing.get("detail") or "")
    next_detail = (
        existing_detail
        if existing_status == merged_status and existing_detail
        else detail or existing_detail
    )
    state.model_usage[role] = {
        "status": merged_status,
        "calls": existing_calls + max(0, calls),
        "detail": next_detail,
    }


def _merged_model_usage_status(existing: str, current: str) -> str:
    priority = {"failed": 4, "used": 3, "disabled": 2, "not_used": 1, "": 0}
    return existing if priority.get(existing, 0) > priority.get(current, 0) else current


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _media_analysis_unavailable(media_analysis: str) -> bool:
    try:
        parsed = json.loads(media_analysis)
    except json.JSONDecodeError:
        text = media_analysis
    else:
        if not isinstance(parsed, dict):
            text = str(parsed)
        else:
            parts = [str(parsed.get("summary") or "")]
            uncertainties = parsed.get("uncertainties")
            if isinstance(uncertainties, list):
                parts.extend(str(item) for item in uncertainties)
            text = " ".join(parts)

    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in (
            "not configured",
            "unavailable",
            "authentication failed",
            "could not authenticate",
            "not allowed to access",
            "media analysis failed",
            "vlm returned no analysis",
        )
    )


def _query_depends_on_media(query: str) -> bool:
    lowered = (query or "").strip().lower()
    if not lowered or lowered == "the user submitted visual media without additional text.":
        return True
    return any(
        phrase in lowered
        for phrase in (
            "this video",
            "the video",
            "in video",
            "from video",
            "attached video",
            "she is wearing",
            "he is wearing",
            "they are wearing",
            "person is wearing",
            "like that",
            "like this",
            "similar to that",
            "similar to this",
            "what is in this",
            "what's in this",
            "what am i wearing",
            "what are they wearing",
            "describe this",
        )
    )


def _media_failure_response(media_analysis: str) -> str:
    detail = "video/image understanding is unavailable for this turn"
    try:
        parsed = json.loads(media_analysis)
    except json.JSONDecodeError:
        parsed = {}
    if isinstance(parsed, dict):
        summary = str(parsed.get("summary") or "").strip()
        if summary:
            detail = _clean_media_failure_detail(summary)

    return (
        f"I could not analyze the attached media because {detail}. "
        "Please describe the item in text, such as color, silhouette, material, "
        "and any visible details, and I can search the catalog from that description."
    )


def _clean_media_failure_detail(summary: str) -> str:
    detail = summary.strip()
    for prefix in (
        "Media was attached, but ",
        "Video was attached, but ",
        "Image was attached, but ",
    ):
        if detail.startswith(prefix):
            detail = detail[len(prefix):]
            break
    if detail:
        detail = detail[0].lower() + detail[1:]
    return detail.rstrip(". ") or "video/image understanding is unavailable for this turn"


def _collect_token_usage(result: Any) -> dict[str, int]:
    """Collect normalized token usage from LangChain/OpenAI message metadata."""

    usage = _empty_token_usage()
    for message in _result_messages(result):
        record = _message_token_usage_record(message)
        if not record:
            continue

        input_tokens = _token_int(record, ("input_tokens", "prompt_tokens", "input"))
        output_tokens = _token_int(
            record,
            ("output_tokens", "completion_tokens", "output"),
        )
        total_tokens = _token_int(record, ("total_tokens", "total"))
        if total_tokens is None and (input_tokens is not None or output_tokens is not None):
            total_tokens = int(input_tokens or 0) + int(output_tokens or 0)

        if input_tokens is None and output_tokens is None and total_tokens is None:
            continue

        usage["input_tokens"] += int(input_tokens or 0)
        usage["output_tokens"] += int(output_tokens or 0)
        usage["total_tokens"] += int(total_tokens or 0)
        usage["model_calls"] += 1
    return usage


def _merge_token_usage(
    base: dict[str, Any] | None,
    additional: dict[str, Any] | None,
) -> dict[str, int]:
    merged = _normalized_token_usage(base)
    extra = _normalized_token_usage(additional)
    for key in merged:
        merged[key] += extra[key]
    return merged


def _collect_tool_grounding_evidence(result: Any, *, max_chars: int) -> str:
    parts: list[str] = []
    for message in _result_messages(result):
        content = _content_to_text(_value(message, "content"))
        if not content:
            continue
        if not _is_tool_evidence_message(message, content):
            continue
        parts.append(_customer_safe_tool_evidence(content))

    evidence = "\n\n---\n\n".join(parts).strip()
    if len(evidence) <= max_chars:
        return evidence
    return evidence[-max_chars:]


def _customer_safe_tool_evidence(content: str) -> str:
    if "SEARCH_RESULT_GROUNDING_NOTE" in content:
        return _summarize_product_evidence(
            content,
            heading="CUSTOMER_SAFE_SEARCH_EVIDENCE",
            note=(
                "Search results support only product names, prices, categories, "
                "image availability, and a modest styling role. They do not "
                "support length, color, print, materials, care, construction, "
                "fit, comfort, weather, grass, gravel, heat, or best-in-category "
                "claims. Treat names as display names, not attribute evidence; "
                "group claims require product-detail evidence for every item."
            ),
        )
    if "PRODUCT_DETAIL_GROUNDING_NOTE" in content:
        return _summarize_product_evidence(
            content,
            heading="CUSTOMER_SAFE_PRODUCT_DETAIL_EVIDENCE",
            note=(
                "Product details were read for these products, but the available "
                "detail data contains only the listed facts. Do "
                "not state material, care, dimensions, closures, fit, sizing, "
                "colorways, or outdoor performance unless the field appears in "
                "this evidence summary."
            ),
        )
    return _summarize_cart_evidence(content)


def _summarize_product_evidence(content: str, *, heading: str, note: str) -> str:
    products = _product_evidence_records(content)
    lines = [f"{heading}: {note}"]
    if not products:
        return "\n".join(lines + [_strip_internal_ids_from_evidence_line(content)])
    for product in products:
        summary_parts = [product["name"]]
        if product.get("category"):
            summary_parts.append(f"category: {product['category']}")
        if product.get("price"):
            summary_parts.append(f"price: {product['price']}")
        if product.get("image_url"):
            summary_parts.append("image: available")
        if product.get("details"):
            summary_parts.append("details: " + "; ".join(product["details"]))
        lines.append("- " + " | ".join(summary_parts))
    return "\n".join(lines)


def _product_evidence_records(content: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    reading_details = False
    key_map = {
        "NAME:": "name",
        "CATEGORY:": "category",
        "PRICE:": "price",
        "IMAGE_URL:": "image_url",
    }
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if line.startswith("PRODUCT_REF:"):
            if current.get("name"):
                records.append(current)
            current = {}
            reading_details = False
            continue
        if line == "DETAILS:":
            reading_details = True
            continue
        if reading_details and line.startswith("- ") and ":" in line:
            label, value = line[2:].split(":", 1)
            if label.strip() and value.strip():
                current.setdefault("details", []).append(
                    f"{label.strip()}: {value.strip()}"
                )
            continue
        for prefix, key in key_map.items():
            if line.startswith(prefix):
                current[key] = line[len(prefix) :].strip()
                break
    if current.get("name"):
        records.append(current)
    return records


def _summarize_cart_evidence(content: str) -> str:
    lines = ["CUSTOMER_SAFE_CART_EVIDENCE:"]
    for raw_line in content.splitlines():
        cleaned = _strip_internal_ids_from_evidence_line(raw_line)
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines)


def _strip_internal_ids_from_evidence_line(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return ""
    if stripped.startswith("PRODUCT_REF:") or stripped.startswith("CART_LINE_ID:"):
        return ""
    if stripped.startswith("- CART_LINE_ID:") and "|" in stripped:
        return "- " + stripped.split("|", 1)[1].strip()

    marker = "(PRODUCT_REF:"
    while marker in stripped:
        start = stripped.find(marker)
        end = stripped.find(")", start)
        if end == -1:
            stripped = stripped[:start].rstrip()
            break
        stripped = (stripped[:start] + stripped[end + 1 :]).strip()
    return stripped


def _is_tool_evidence_message(message: Any, content: str) -> bool:
    message_type = str(_value(message, "type") or "").lower()
    role = str(_value(message, "role") or "").lower()
    if message_type == "tool" or role == "tool":
        return True
    return any(
        marker in content
        for marker in (
            "SEARCH_RESULT_GROUNDING_NOTE",
            "PRODUCT_DETAIL_GROUNDING_NOTE",
            "CART_ADD_RESULT",
            "PRODUCT_REF:",
            "CART_LINE_ID:",
            "Cart total:",
        )
    )


def _normalized_token_usage(raw: dict[str, Any] | None) -> dict[str, int]:
    usage = _empty_token_usage()
    if not isinstance(raw, dict):
        return usage
    for key in usage:
        value = raw.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            usage[key] = max(0, int(value))
    return usage


def _empty_token_usage() -> dict[str, int]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "model_calls": 0,
    }


def _result_messages(result: Any) -> list[Any]:
    if isinstance(result, dict):
        messages = result.get("messages")
        if isinstance(messages, list):
            return messages
        return [result]

    messages = getattr(result, "messages", None)
    if isinstance(messages, list):
        return messages
    return [result]


def _message_token_usage_record(message: Any) -> Any:
    usage_metadata = _value(message, "usage_metadata")
    if usage_metadata:
        return usage_metadata

    response_metadata = _value(message, "response_metadata")
    if isinstance(response_metadata, dict):
        token_usage = response_metadata.get("token_usage")
        if token_usage:
            return token_usage

    for key in ("token_usage", "usage"):
        record = _value(message, key)
        if record:
            return record
    return None


def _value(source: Any, key: str) -> Any:
    if isinstance(source, dict):
        return source.get(key)
    return getattr(source, key, None)


def _token_int(record: Any, keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = _value(record, key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return max(0, int(value))
    return None


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


def _tool_search_mode(value: str | None) -> str | None:
    return value if value in {"text", "image", "hybrid"} else None


def _format_product(product: Any) -> str:
    lines = [
        f"PRODUCT_REF: {product.product_id}",
        f"NAME: {product.display_name}",
    ]
    if getattr(product, "category", None):
        lines.append(f"CATEGORY: {product.category}")
    if product.price:
        lines.append(f"PRICE: ${product.price.amount:.2f} {product.price.currency}")
    if product.image_url:
        lines.append(f"IMAGE_URL: {product.image_url}")
    lines.append(
        "DETAILS: Call get_product_details_tool with this PRODUCT_REF before "
        "stating materials, dimensions, pockets, closures, care, comfort, or "
        "outdoor-practicality claims."
    )
    return "\n".join(lines)


def _format_product_details(product: ProductDetail) -> str:
    lines = [
        _PRODUCT_DETAIL_GROUNDING_NOTE,
        f"PRODUCT_REF: {product.product_id}",
        f"NAME: {product.display_name}",
    ]
    if product.category:
        lines.append(f"CATEGORY: {product.category}")
    if product.brand:
        lines.append(f"BRAND: {product.brand}")
    if product.price:
        lines.append(f"PRICE: ${product.price.amount:.2f} {product.price.currency}")
    if product.image_url:
        lines.append(f"IMAGE_URL: {product.image_url}")
    if product.attributes:
        lines.append("DETAILS:")
        for name, value in sorted(product.attributes.items()):
            lines.append(
                f"- {name.replace('_', ' ')}: {_format_detail_value(value)}"
            )
    else:
        lines.append("NO_ADDITIONAL_STRUCTURED_DETAILS")
    return "\n".join(lines)


def _format_detail_value(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    if isinstance(value, dict):
        return ", ".join(
            f"{key}={value[key]}" for key in sorted(value)
        )
    return str(value)


def _normalize_cart_add_tool_items(
    items: list[AddCartItemsToolItemInput] | list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    for item in items or []:
        try:
            parsed = (
                item
                if isinstance(item, AddCartItemsToolItemInput)
                else AddCartItemsToolItemInput.model_validate(item)
            )
            quantity = max(1, int(parsed.quantity or 1))
        except (TypeError, ValueError, ValidationError) as exc:
            raise ValueError("each item must include a PRODUCT_REF and quantity") from exc
        entry = normalized.setdefault(
            parsed.product_ref,
            {
                "quantity": 0,
                "expected_display_name": (
                    parsed.expected_display_name.strip()
                    if parsed.expected_display_name
                    else ""
                ),
            },
        )
        entry["quantity"] += quantity
        if not entry["expected_display_name"] and parsed.expected_display_name:
            entry["expected_display_name"] = parsed.expected_display_name.strip()
    return normalized


def _cart_add_scope_failures(
    user_query: str,
    requested_products: list[tuple[str, ProductSummary]],
    cached_products: Any,
) -> list[str]:
    explicitly_named = _explicitly_named_products(user_query, cached_products)
    if not explicitly_named:
        return []

    explicit_names = {
        _normalize_product_name(product.display_name) for product in explicitly_named
    }
    failures = []
    for product_ref, product in requested_products:
        if _normalize_product_name(product.display_name) in explicit_names:
            continue
        failures.append(
            f"- PRODUCT_REF '{product_ref}': selected '{product.display_name}' is "
            "outside the current explicit add request. The current request names: "
            f"{_format_cached_product_refs(explicitly_named)}. Retry with matching "
            "PRODUCT_REF values only, or ask a clarification."
        )
    return failures


def _explicitly_named_products(text: str, cached_products: Any) -> list[ProductSummary]:
    normalized_text = _normalize_product_name(text)
    if not normalized_text:
        return []

    padded_text = f" {normalized_text} "
    matches: list[ProductSummary] = []
    seen: set[str] = set()
    products = list(cached_products)
    for product in products:
        normalized_name = _normalize_product_name(product.display_name)
        if not normalized_name:
            continue
        if f" {normalized_name} " not in padded_text:
            continue
        key = product.product_id or product.display_name
        if key in seen:
            continue
        seen.add(key)
        matches.append(product)

    query_tokens = set(_product_name_tokens(text))
    for product in products:
        key = product.product_id or product.display_name
        if key in seen:
            continue
        product_tokens = _product_name_tokens(product.display_name)
        required_overlap = 3 if len(product_tokens) > 3 and matches else 2
        if not _product_name_tokens_match(
            query_tokens,
            product_tokens,
            required_overlap=required_overlap,
        ):
            continue
        seen.add(key)
        matches.append(product)
    return matches


def _same_product_display_name(expected: str, actual: str) -> bool:
    return _normalize_product_name(expected) == _normalize_product_name(actual)


def _product_detail_failure_message(
    error: CommerceError | None,
    *,
    cart_validation: bool,
) -> str:
    if error is not None and error.code == "product_not_found":
        return (
            "The product is no longer present in the active catalog. "
            "Search again before adding it."
            if cart_validation
            else (
                "That product is no longer available in the active catalog. "
                "Search the catalog again before using its details."
            )
        )
    if error is not None and error.retryable:
        return (
            "The catalog is temporarily unavailable, so the cart was not changed. "
            "Please try again."
            if cart_validation
            else "Product details are temporarily unavailable. Please try again."
        )
    return (
        "The product could not be verified, so the cart was not changed. "
        "Search again before adding it."
        if cart_validation
        else "Product details could not be verified. Search the catalog again."
    )


def _normalize_product_name(value: str) -> str:
    chars = []
    for char in str(value or "").casefold():
        chars.append(char if char.isalnum() else " ")
    return " ".join("".join(chars).split())


def _product_name_tokens(value: str) -> list[str]:
    return [
        token
        for token in _normalize_product_name(value).split()
        if token not in _PRODUCT_NAME_STOPWORDS
    ]


def _product_name_tokens_match(
    query_tokens: set[str],
    product_tokens: list[str],
    *,
    required_overlap: int,
) -> bool:
    if len(product_tokens) < 2:
        return False
    overlap = query_tokens.intersection(product_tokens)
    required = min(required_overlap, len(set(product_tokens)))
    return len(overlap) >= required


def _format_cached_product_refs(products: list[ProductSummary]) -> str:
    return ", ".join(
        f"{product.display_name} (PRODUCT_REF: {product.product_id})"
        for product in products
    )


def _scrub_internal_shopper_language(text: str) -> str:
    scrubbed = text or ""
    for internal, replacement in _INTERNAL_SHOPPER_REPLACEMENTS:
        scrubbed = scrubbed.replace(internal, replacement)
    return scrubbed


def _format_cart_add_result(added: list[str], failed: list[str], cart: Cart) -> str:
    lines = ["CART_ADD_RESULT"]
    if added:
        lines.append("Added:")
        lines.extend(added)
    if failed:
        lines.append("Failed:")
        lines.extend(failed)
    lines.append("Current cart:")
    lines.append(_format_cart(cart))
    lines.append("Cart total:")
    lines.append(_format_cart_total(cart))
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


def _format_retrieved_images(retrieved: dict[str, str] | None) -> str:
    if not retrieved:
        return "(none)"
    lines = []
    for name, image_url in retrieved.items():
        lines.append(f"- {name}: image available")
    return "\n".join(lines)


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
