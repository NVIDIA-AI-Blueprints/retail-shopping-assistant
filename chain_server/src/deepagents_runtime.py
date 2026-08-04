# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deep Agents SDK runtime for the shopping assistant."""

from __future__ import annotations

import logging

import asyncio
import json
import os
from pathlib import Path
import time
from typing import Any, AsyncIterator, Literal

from langgraph.errors import GraphRecursionError
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
)
import requests

from .agenttypes import Cart, ShopperContext, State
from .catalog_search import SearchContext, search_catalog
from .response_format import (
    _format_availability_result,
    _format_cart,
    _format_cart_add_result,
    _format_cart_remove_result,
    _format_cart_total,
    _format_media_summary,
    _format_policy_result,
    _format_product_detail_record,
    _format_promotions_result,
    _format_retrieved_images,
    _format_shopper_context,
    _format_update_cart_result,
)
from .turn_support import (
    _search_catalog_scopes_input_model,
    AddCartItemsToolItemInput,
    RequestIdentity,
    _add_model_usage,
    _build_checkpointer,
    _cart_add_scope_failures,
    _cart_line_by_id,
    _catalog_repair_clarification_response,
    _collect_token_usage,
    _collect_tool_grounding_evidence,
    _committed_effect_receipt,
    _conversation_turn_status,
    _empty_agent_diagnostics,
    _format_search_only_response,
    _has_grounding_authority,
    _has_search_only_tool_evidence,
    _has_successful_non_search_tool_evidence,
    _media_failure_response,
    _merge_token_usage,
    _no_direct_taxonomy_response,
    _normalize_cart_add_tool_items,
    _normalized_token_usage,
    _partial_graph_messages,
    _partial_product_results_response,
    _product_detail_failure_message,
    _product_detail_record,
    _record_language_model_failure,
    _record_media_model_usage,
    _record_safety_model_usage,
    _rejected_catalog_search_response,
    _safe_collect_agent_diagnostics,
    _same_product_display_name,
    _scrub_internal_shopper_language,
    _search_catalog_tool_input_model,
    _search_guidance_evidence,
    _should_short_circuit_media_failure,
    _skill_activation_input_model,
    _store_policies_path,
)
from .catalog_capabilities import (
    CatalogCapabilitiesClient,
    format_catalog_capabilities_for_prompt,
)
from .catalog_scope import CATALOG_SEARCH_RULES
from .commerce_tools import (
    add_cart_item,
    check_active_promotions,
    check_product_availability,
    get_cart,
    get_product_details,
    get_store_policy,
    remove_cart_item,
    update_cart_item,
)
from .conversation_memory import (
    ConversationMemoryClient,
    ConversationMemoryError,
    FinalTurnStatus,
    TurnReplayOutput,
    TurnStartResult,
    build_dialogue_context,
)
from .control_signals import (
    EFFECTS_KEY,
    ControlSignal,
    committed_effect,
    committed_effects_in,
    control,
    normalize_tool_result,
)
from .conversation_products import (
    ConversationProductsClient,
    ConversationProductsError,
    ProductReferenceDescriptor,
    ResolveConversationProductsRequest,
    format_historical_product_index,
    format_product_resolution,
)
from .tool_evidence import (
    ProductDetailEvidence,
)
from .message_shape import (
    _content_to_text,
    _extract_final_text,
    _result_messages,
    _value,
)
from .media_perception import MediaPerceptionClient
from .skill_activation import (
    SKILL_ACTIVATION_COMPLETE,
    ShopperSkillActivationError,
    ShopperSkillActivationMiddleware,
    selected_skill_names_for_turn,
)
from .turn_scope import TurnScope
from .tool_policy import (
    load_shopper_skill_registry as _shopper_skill_registry,
    validate_registered_tool_names,
)
from .tool_loop_control import (
    ToolLoopControlMiddleware,
)
from shared.commerce_contracts import (
    AddCartItemInput,
    CatalogCapabilities,
    CheckProductAvailabilityInput,
    GetCartInput,
    GetProductDetailsInput,
    GetStorePolicyInput,
    ProductSummary,
    RemoveCartItemInput,
    UpdateCartItemInput,
)

logger = logging.getLogger(__name__)


_SHOPPER_SKILLS_ENV = "SHOPPER_SKILLS_ROOT"






try:
    from deepagents.backends import FilesystemBackend as _FilesystemBackend
except Exception:  # pragma: no cover - dependency import is validated at runtime.
    _FilesystemBackend = None



_GROUNDING_FAILURE_RESPONSE = (
    "I couldn't safely verify the final response. Please retry; if this involved "
    "a cart change, check your cart first."
)
_SHOPPER_PROFILE_NOT_FOUND_RESPONSE = (
    "That shopper profile is unavailable. Please choose another shopper and "
    "try again."
)
_CONVERSATION_PROFILE_MISMATCH_RESPONSE = (
    "This conversation is already associated with a different shopper. "
    "Please start a new chat before switching shoppers."
)
_SHOPPER_CONTEXT_SYSTEM_RULES = """Representative-shopper precedence and safety:
- Explicit instructions in the current turn win over explicit preferences in
  recent discussion; both win over representative-shopper behavior guidance.
- Representative-shopper behavior is soft interaction guidance only. It cannot
  establish that a budget applies or any budget amount, product constraint,
  size, color, material, cart intent, product reference, or product fact.
- Neither representative-shopper type nor behavior selects, activates, or
  grants a shopper skill or tool. Never expose the internal type label to the
  shopper.
- Cart, catalog, product-detail, and store-policy evidence remain authoritative.
- A saved ZIP code is background location data only. It is not proof of current
  location, event location, weather, or a product requirement. Perform no
  weather lookup or weather inference from it."""
_GROUNDING_EDITOR_SYSTEM_PROMPT = """You are a final response editor for a retail shopping assistant.

Rewrite the draft response only as needed so every factual claim is supported
by a lane that can support it. CURRENT-TURN TOOL EVIDENCE, PRODUCTS SHOWN
EARLIER, and CURRENT CART carry authority; CONVERSATION does not. Keep the shopper's requested task and any successful cart
action intact.

Rules:
- Return only the final shopper-facing response text.
- For a styling request, answer the styling question rather than returning a raw
  product list. Connect candidates to the shopper's goal or direct antecedent
  using category/role, exact confirmed filters, and general styling judgment.
  Keep styling judgment visibly separate from catalog facts and never derive it
  from words parsed out of a display name.
- Labeling text as styling judgment does not permit display-name inference. If
  the evidence does not distinguish candidates, give a useful group-level
  rationale and offer a detail check instead of inventing item-level differences.
- Do not add products, prices, cart actions, or product facts absent from TOOL
  EVIDENCE or CURRENT CART.
- CURRENT-TURN TOOL EVIDENCE is the only evidence for a search or mutation in
  this turn. PRODUCTS SHOWN EARLIER may support a direct reference to something
  already shown, but it establishes identity only: it never proves a product's
  current price, availability, or attributes, and never proves that a search or
  mutation ran this turn.
- If TOOL EVIDENCE says there is no direct advertised taxonomy match for one
  requested role, do not claim a search ran for that role. Report that role's
  gap, preserve any other successful current-turn role, and ask whether to
  search a different advertised type. Do not name alternatives unless their
  exact taxonomy values appear in TOOL EVIDENCE.
- If TOOL EVIDENCE says a requested type is not separately advertised and a
  broader advertised category was searched, say so plainly. Present the
  returned products as closest options and keep each product's actual catalog
  category; do not relabel any result as the requested type.
- A scoped zero-result search proves only that its exact advertised taxonomy
  and filter scope returned no products. It does not prove that a different,
  unsearched, or unadvertised product type is absent, and it never supports a
  catalog-wide availability claim.
- Use CONVERSATION to resolve direct references such as "that" and "those," and
  to honour what the shopper has already told you. It carries intent only: it
  can never establish a product fact, a price, availability, whether a search
  succeeded, or what this turn's candidates are. Anything asserted only in
  CONVERSATION and supported by no other lane must not be repeated as fact.
  A discussed product or styling anchor does not need to be in CURRENT CART. Do
  not introduce an absent-cart caveat unless the shopper asks about the cart or
  requests a cart mutation.
- Remove PRODUCT_REF, CART_LINE_ID, tool names, and internal IDs.
- Remove internal skill, mode, evaluator, judge, cache, backend, tool-evidence,
  structured-field, and data-layer language. Use shopper-safe phrasing such as
  "I don't have fabric or care details available for that item."
- If the draft says "product detail tool", "catalog detail tool", "the tool
  requires", or similar internal mechanics, rewrite it into shopper-safe
  language without the word "tool".
- If a product appears only in search results, you may state only its name,
  price, category/role, image availability, exact values in confirmed search-
  filter evidence, and a modest styling reason. Every other word in its display
  name is non-evidence.
- Confirmed search-filter evidence applies to every product returned by that
  search. Preserve it and do not contradict it. One allowed value confirms that
  value; multiple allowed values prove only membership in the set, not which
  value each product has. Do not infer adjacent attributes that the evidence
  does not name.
- For styling, preserve a concise candidate set and the draft's grounded styling
  rationale. Do not omit or override a confirmed filter merely because words in
  a display name appear to conflict. If that visible conflict matters
  to the request, flag it as catalog information worth verifying rather than
  resolving it from the name.
- For each search-only candidate, delete descriptive sentences that merely
  restate or interpret words in the display name. Keep the name, price,
  category/role, confirmed filters, and a modest reason tied to the shopper's
  stated goal.
- In a search-only response, copy a candidate's display name only as its exact
  title. Do not shorten it into an attribute, classify or group candidates by
  words appearing in their names, or use those words as the reason one
  candidate differs from another. Without product details, give one concise
  group-level styling rationale based on the shopper's goal, advertised role,
  and confirmed filters instead of item-specific attribute rationales.
- Advertised search-taxonomy evidence lists the valid product types used by that
  search. Do not call an unlisted product type advertised or offer it as an
  advertised alternative.
- Treat product names as display names, not proof of length, color, print,
  material, construction, fit, care, or vibe. Do not say a product is solid,
  floral, gingham, maxi, knee-length, woven, structured, neutral, lightweight,
  polished, or dressier unless that attribute appears in product-detail evidence.
- Material, care, dimensions, pockets, closures, fit, comfort, and outdoor
  practicality claims require matching product-detail evidence and a direct
  shopper need for that fact.
- If the shopper's requested outcome depends on a material, fit, comfort,
  durability, care, weather, or other functional property that TOOL EVIDENCE
  does not confirm, say that property is not confirmed. Frame the candidates as
  the closest catalog or styling direction, not as complete, suitable, ready,
  or proven for that outcome. Keep any missing functional element explicit
  without inventing a product.
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
















































































class AddCartItemsToolInput(BaseModel):
    items: list[AddCartItemsToolItemInput] = Field(
        ...,
        min_length=1,
        description=(
            "One or more products to add. Each must use a PRODUCT_REF "
            "established by current-turn search or historical-product resolution."
        ),
    )


class _UpdateCartItemsInput(BaseModel):
    cart_line_id: str = Field(
        description="CART_LINE_ID from get_cart_tool. Not the product name."
    )
    quantity: int = Field(
        ge=0,
        description="New total quantity. Set to 0 to remove the line.",
    )


class _GetStorePolicyInput(BaseModel):
    topic: Literal[
        "returns",
        "shipping",
        "sizing",
        "payment",
        "price_match",
        "gift_cards",
    ] = Field(description="Policy topic to look up.")


class _CheckAvailabilityInput(BaseModel):
    product_ref: str = Field(
        description=(
            "PRODUCT_REF established by current-turn search or historical-product "
            "resolution."
        )
    )
    variant_hint: str | None = Field(
        default=None,
        description="Requested size wording, such as 'size 8'.",
    )






class DeepAgentsRuntime:
    """Small adapter around the Deep Agents SDK.

    This class intentionally keeps commerce truth in existing services. Deep
    Agents gets scoped context and deterministic tools; it does not own carts,
    profiles, prices, inventory, or session identity.
    """

    def __init__(self, config: Any) -> None:
        self.config = config
        self._checkpointer = _build_checkpointer()
        self._profile_registered = False
        self._media_perception = MediaPerceptionClient(config)
        self._catalog_capabilities = CatalogCapabilitiesClient(
            config.retriever_port,
            timeout_seconds=config.catalog_search_timeout_seconds,
        )
        self._conversation_memory = ConversationMemoryClient(config.memory_port)
        self._conversation_products = ConversationProductsClient(config.memory_port)

    def catalog_capabilities(self) -> CatalogCapabilities:
        """Return the process-lifecycle catalog capability contract."""

        return self._catalog_capabilities.get()

    def _exposed_agent_diagnostics(self, output: State) -> dict[str, Any]:
        if not getattr(self.config, "expose_agent_diagnostics", False):
            return {}
        return output.agent_diagnostics

    async def astream(
        self,
        state: State,
        identity: RequestIdentity,
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
                    "agent_diagnostics": self._exposed_agent_diagnostics(output),
                },
                "timestamp": time.time(),
            }
        )

    async def ainvoke(
        self,
        state: State,
        identity: RequestIdentity,
    ) -> dict[str, Any]:
        output = await self._run_turn(state, identity)
        return {
            "response": output.response,
            "images": output.retrieved or {},
            "cart": output.cart.model_dump(mode="json"),
            "timings": output.timings,
            "token_usage": _normalized_token_usage(output.token_usage),
            "model_usage": output.model_usage,
            "agent_diagnostics": self._exposed_agent_diagnostics(output),
        }

    async def _run_turn(
        self,
        state: State,
        identity: RequestIdentity,
    ) -> State:
        state.user_id = identity.context_user_id
        state.agent_diagnostics = _empty_agent_diagnostics("not_started")
        state.previous_selected_skill_names = []
        state.selected_skill_names = []
        state.shopper_profile_id = identity.shopper_profile_id
        state.shopper_context = None
        turn = self._start_conversation_turn(state, identity)
        if turn is not None and turn.replayed:
            await self._delete_turn_checkpoint(identity)
            return self._restore_replayed_turn(state, turn)
        if turn is None and state.response:
            return state

        try:
            output = await self._execute_turn(state, identity)
        except asyncio.CancelledError:
            if turn is not None:
                if not state.response:
                    state.response = (
                        "This request was interrupted. Please check your cart "
                        "before retrying."
                    )
                finalized = self._finalize_conversation_turn(
                    state,
                    identity,
                    turn,
                    status="failed",
                    termination_reason="request_cancelled",
                    present_products=False,
                )
                if finalized:
                    await self._delete_turn_checkpoint(identity)
            raise
        except Exception:
            if turn is not None:
                finalized = self._finalize_conversation_turn(
                    state,
                    identity,
                    turn,
                    status="failed",
                    termination_reason="unexpected_runtime_error",
                    present_products=False,
                )
                if finalized:
                    await self._delete_turn_checkpoint(identity)
            raise

        if turn is not None:
            finalized = self._finalize_conversation_turn(state, identity, turn)
            if finalized:
                await self._delete_turn_checkpoint(identity)
        return output

    async def _execute_turn(
        self,
        state: State,
        identity: RequestIdentity,
    ) -> State:
        start = time.monotonic()

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
                state.agent_diagnostics = _empty_agent_diagnostics(
                    "input_guardrail_blocked"
                )
                return state

        media_start = time.monotonic()
        state.media_analysis = await self._media_perception.analyze(state)
        if state.media:
            state.timings["media_perception"] = time.monotonic() - media_start
            _record_media_model_usage(state, self.config)
        if _should_short_circuit_media_failure(state):
            state.response = _media_failure_response(state.media_analysis)
            state.timings["deepagents"] = time.monotonic() - start
            state.agent_diagnostics = _empty_agent_diagnostics("media_failure")
            return state

        turn_capabilities = await asyncio.to_thread(self._catalog_capabilities.get)
        invoke_config = {
            "configurable": {"thread_id": identity.checkpoint_thread_id},
            "recursion_limit": self.config.deepagents_recursion_limit,
        }
        agent = None
        try:
            execution_deadline = (
                time.monotonic()
                + self.config.deepagents_execution_timeout_seconds
            )
            agent = self._create_agent(
                state,
                identity,
                turn_capabilities,
            )
            input_message = self._build_user_message(state, identity)
            # The grounding editor is not optional, so its budget is reserved
            # before the agent loop runs rather than taken from what the loop
            # leaves. Measured 2026-08-04: the loop reached the full 45s turn
            # budget on six turns, leaving the editor zero seconds and sending
            # the grounding-failure message to the shopper.
            # Never take more than half the turn: a deployment with a short
            # budget must still get an agent loop, and a reserve larger than the
            # budget would fail every turn before any work happened.
            grounding_reserve = min(
                max(
                    0.0,
                    float(
                        getattr(
                            self.config, "grounding_editor_reserve_seconds", 0.0
                        )
                    ),
                ),
                max(0.0, float(self.config.deepagents_execution_timeout_seconds)) / 2,
            )
            agent_timeout = max(
                0.0,
                execution_deadline - time.monotonic() - grounding_reserve,
            )
            if agent_timeout <= 0:
                raise TimeoutError
            result = await asyncio.wait_for(
                agent.ainvoke(
                    {"messages": [{"role": "user", "content": input_message}]},
                    config=invoke_config,
                ),
                timeout=agent_timeout,
            )
            result_messages = _result_messages(result)
            state.selected_skill_names = list(
                selected_skill_names_for_turn(
                    result_messages,
                    identity.request_id,
                )
            )
            draft_response = _extract_final_text(result)
            state.token_usage = _collect_token_usage(result)
            state.agent_diagnostics = _safe_collect_agent_diagnostics(
                result_messages,
                request_id=identity.request_id,
                final_termination_reason="completed",
            )
            # An unenforceable requirement is a fact for the model to speak to,
            # not a reason for the runtime to seize the turn. Deterministic code
            # establishes that the filter is not advertised; the model decides
            # what to say about it, constrained by the grounding editor.
            state.response = _no_direct_taxonomy_response(
                result,
                request_id=identity.request_id,
            )
            if not state.response:
                # Never below the reserve, however long the loop actually ran.
                remaining_seconds = max(
                    grounding_reserve,
                    execution_deadline - time.monotonic(),
                )
                state.response = await self._rewrite_response_for_grounding(
                    state,
                    result,
                    draft_response,
                    request_id=identity.request_id,
                    timeout_seconds=remaining_seconds,
                )
            if not state.response:
                state.response = (
                    "I could not complete that shopping request. Please try again."
                )
                state.agent_diagnostics[
                    "final_termination_reason"
                ] = "incomplete_agent_response"
        except Exception as exc:  # noqa: BLE001 - keep endpoint resilient.
            partial_messages, capture_error = await _partial_graph_messages(
                agent,
                invoke_config,
            )
            state.selected_skill_names = list(
                selected_skill_names_for_turn(
                    partial_messages,
                    identity.request_id,
                )
            )
            if isinstance(exc, GraphRecursionError):
                termination_reason = "recursion_limit"
            elif isinstance(exc, ShopperSkillActivationError):
                termination_reason = "skill_activation_failed"
            elif isinstance(exc, TimeoutError):
                termination_reason = "agent_timeout"
            else:
                termination_reason = "agent_error"
            state.agent_diagnostics = _safe_collect_agent_diagnostics(
                partial_messages,
                request_id=identity.request_id,
                final_termination_reason=termination_reason,
                preserve_partial_messages=True,
            )
            if capture_error:
                state.agent_diagnostics["partial_graph_capture_error"] = capture_error
            logger.exception("DeepAgentsRuntime failed")
            # A committed cart change must never be concealed by a read-only
            # fallback. When the graph snapshot could not be read we cannot rule
            # a mutation out, so that case is treated as uncertain rather than
            # as "no mutation".
            effects = committed_effects_in(partial_messages)
            effects_unknown = bool(capture_error)
            if termination_reason == "agent_timeout":
                state.product_results = []
                state.retrieved = {}
                state.response = (
                    "This request took too long to complete. Please retry. If it "
                    "involved a cart change, check your cart first."
                )
            elif effects:
                state.product_results = []
                state.retrieved = {}
                state.response = _committed_effect_receipt(
                    effects,
                    self._safe_read_cart(identity.cart_user_id),
                )
            elif effects_unknown:
                state.product_results = []
                state.retrieved = {}
                state.response = (
                    "Something went wrong before I could confirm this request. "
                    "Please check your cart before retrying, in case a change "
                    "was already applied."
                )
            else:
                fallback_response = _partial_product_results_response(state)
                state.response = fallback_response or (
                    "I encountered an error while helping with your shopping request. "
                    "Please try again."
                )
            _record_language_model_failure(state)
            state.timings["deepagents_error"] = time.monotonic() - start
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
                state.agent_diagnostics[
                    "final_termination_reason"
                ] = "output_guardrail_blocked"

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

        skills_root = self._shopper_skills_root()
        skills_backend = self._create_skills_backend()
        if skills_backend is None:
            raise RuntimeError("Shopper skill backend is unavailable.")
        skill_registry = _shopper_skill_registry(skills_root)
        skill_activation_input = _skill_activation_input_model(
            tuple(skill_registry)
        )
        scope = TurnScope()
        state.retrieved = scope.retrieved
        search_input_model = _search_catalog_tool_input_model(turn_capabilities)
        search_tool_arguments_model = _search_catalog_scopes_input_model(
            turn_capabilities,
        )
        constraint_input_model = search_input_model.model_fields[
            "required_constraints"
        ].annotation

        search_context = SearchContext(
            config=self.config,
            state=state,
            scope=scope,
            capabilities=turn_capabilities,
            search_input_model=search_input_model,
            constraint_input_model=constraint_input_model,
        )

        def _search_catalog_impl(
            semantic_query: str,
            requested_product_type: str | None,
            taxonomy: BaseModel | dict[str, Any],
            required_constraints: BaseModel | dict[str, Any],
            shopper_guidance: str,
            scope_complete: bool = True,
            search_mode: str | None = None,
        ):
            """Execute one catalog search; may return control signals."""

            return search_catalog(
                search_context,
                semantic_query,
                requested_product_type,
                taxonomy,
                required_constraints,
                shopper_guidance,
                scope_complete,
                search_mode,
            )


        @tool(
            args_schema=search_tool_arguments_model,
            return_direct=False,
            response_format="content_and_artifact",
        )
        def search_catalog_tool(scopes):
            """Find products by description, advertised taxonomy, or constraints.

            Use for browse, search, and recommendation requests after product
            discovery or outfit styling is active. Select exact values from the
            current Catalog capabilities. Do not use for a product already
            established in this conversation, and do not repeat a completed hard-
            filter scope with different semantic wording.
            """

            # One scope for now. The list is the contract; N > 1 is enabled only
            # once the model is shown to emit the nested shape reliably.
            scope = scopes[0]
            fields = (
                scope
                if isinstance(scope, dict)
                else scope.model_dump(exclude_none=False)
            )
            return normalize_tool_result(
                _search_catalog_impl(
                    semantic_query=fields["semantic_query"],
                    requested_product_type=fields.get("requested_product_type"),
                    taxonomy=fields["taxonomy"],
                    required_constraints=fields["required_constraints"],
                    shopper_guidance=fields["shopper_guidance"],
                    scope_complete=fields.get("scope_complete", True),
                    search_mode=fields.get("search_mode"),
                )
            )

        @tool(return_direct=False)
        def get_cart_tool() -> str:
            """Read the current cart. Use before cart mutations to get
            CART_LINE_ID values, or when the shopper asks what is in their cart.
            Do NOT call again if the cart was already read this turn and no
            mutation has occurred since.
            """

            cart = self._read_cart(identity.cart_user_id)
            state.cart = cart
            self._append_product_images(
                scope.retrieved,
                cart,
                scope.product_evidence.values(),
            )
            return _format_cart(cart)

        def _get_product_details_impl(product_ref: str):
            """Get detailed facts (material, care, dimensions, closures) for a
            product established in this turn by search or historical-product
            resolution. Requires a PRODUCT_REF — not a product name. Do NOT call
            for initial recommendations. Stop immediately if STOP_TOOL_USE is
            returned.
            """

            if (
                scope.product_detail_reads
                >= self.config.max_product_detail_reads_per_turn
            ):
                return control(
                    "STOP_TOOL_USE: Product-detail read limit reached for this "
                    "turn. Do not call more tools this turn. Answer now from the "
                    "details already read and keep any other products to names, "
                    "prices, categories, image availability, and styling role.",
                    ControlSignal.STOP_TOOL_USE,
                )
            scope.product_detail_reads += 1

            cached_product = scope.product_evidence.get(product_ref)
            if cached_product is None:
                return (
                    f"No product with PRODUCT_REF '{product_ref}' is available. "
                    "Search this turn or resolve the earlier product first."
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
                scope.retrieved[product.display_name] = product.image_url
            record = _product_detail_record(product)
            evidence = ProductDetailEvidence(products=[record])
            return (
                _format_product_detail_record(record),
                evidence.as_artifact(),
            )

        @tool(return_direct=False, response_format="content_and_artifact")
        def get_product_details_tool(product_ref: str):
            """Get detailed facts (material, care, dimensions, closures) for a
            product established in this turn by search or historical-product
            resolution. Requires a PRODUCT_REF — not a product name. Do NOT call
            for initial recommendations. Stop immediately if STOP_TOOL_USE is
            returned.
            """

            return normalize_tool_result(_get_product_details_impl(product_ref))

        def _resolve_conversation_products_impl(
            references: list[ProductReferenceDescriptor],
        ):
            """Resolve products the shopper refers to from earlier in this
            conversation. Use only when a needed product was not established
            in the current turn. Submit exact descriptors from the historical
            product index. If a reference is missing or ambiguous, ask one
            concise clarification; do not guess or search for a substitute.
            """

            with scope.resolution_lock:
                if scope.product_resolution_used:
                    return control(
                        "STOP_TOOL_USE: Historical product resolution limit "
                        "reached for this turn. Use the first resolution result "
                        "and ask one concise clarification if needed.",
                        ControlSignal.STOP_TOOL_USE,
                    )
                scope.product_resolution_used = True

            try:
                result = self._conversation_products.resolve(
                    identity.conversation_id,
                    references,
                )
            except (ConversationProductsError, ValidationError):
                return (
                    "REFERENCE RESOLUTION UNAVAILABLE: Ask which earlier product "
                    "the shopper means; do not guess or search for a substitute."
                )
            scope.product_evidence.add_resolutions(result.results)
            for resolution in result.results:
                if resolution.status != "resolved":
                    continue
                product = resolution.matches[0].product
                if product.image_url:
                    scope.retrieved[product.display_name] = product.image_url
            return format_product_resolution(result)

        @tool(
            args_schema=ResolveConversationProductsRequest,
            return_direct=False,
            response_format="content_and_artifact",
        )
        def resolve_conversation_products_tool(
            references: list[ProductReferenceDescriptor],
        ):
            """Resolve products the shopper refers to from earlier in this
            conversation. Use only when a needed product was not established
            in the current turn. Submit exact descriptors from the historical
            product index. If a reference is missing or ambiguous, ask one
            concise clarification; do not guess or search for a substitute.
            """

            return normalize_tool_result(
                _resolve_conversation_products_impl(references)
            )

        def _add_cart_items_impl(items: list[AddCartItemsToolItemInput]):
            """Add products to the cart. Use ONLY on explicit shopper intent to
            add, buy, or put items in the cart. Requires PRODUCT_REF values from
            current-turn search or historical-product resolution — not names.
            Call once with all items, not once per item.
            """

            try:
                requested_items = _normalize_cart_add_tool_items(items)
            except ValueError as exc:
                return f"Cart add failed: {exc}"
            if not requested_items:
                return "Cart add failed: provide at least one PRODUCT_REF to add."

            resolved: list[tuple[str, ProductSummary, int]] = []
            failed: list[str] = []
            blocked: list[str] = []
            for product_ref, request in requested_items.items():
                product = scope.product_evidence.get(product_ref)
                if product is None:
                    failed.append(
                        f"- PRODUCT_REF '{product_ref}': Search this turn or "
                        "resolve the earlier product first."
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
                    scope.product_evidence.values(),
                )
            )
            if blocked:
                state.cart = self._read_cart(identity.cart_user_id)
                self._append_product_images(
                    scope.retrieved,
                    state.cart,
                    scope.product_evidence.values(),
                )
                return _format_cart_add_result([], failed + blocked, state.cart)

            added: list[str] = []
            committed: list[dict[str, Any]] = []
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
                    committed.append(
                        {
                            "operation": "added to cart",
                            "idempotency_key": (
                                f"{identity.request_id}:add:"
                                f"{product.product_id}:{quantity}"
                            ),
                            "product_id": product.display_name,
                            "quantity": quantity,
                        }
                    )
                    added.append(
                        f"- {quantity} x {product.display_name} "
                        f"(PRODUCT_REF: {product.product_id})"
                    )
                else:
                    message = (
                        result.error.message if result.error else "Cart add failed."
                    )
                    failed.append(f"- PRODUCT_REF '{product_ref}': {message}")

            state.cart = self._read_cart(identity.cart_user_id)
            self._append_product_images(
                scope.retrieved,
                state.cart,
                scope.product_evidence.values(),
            )
            rendered = _format_cart_add_result(added, failed, state.cart)
            if not committed:
                return rendered
            return rendered, {EFFECTS_KEY: committed}

        @tool(
            args_schema=AddCartItemsToolInput,
            return_direct=False,
            response_format="content_and_artifact",
        )
        def add_cart_items_tool(items: list[AddCartItemsToolItemInput]):
            """Add products to the cart. Use ONLY on explicit shopper intent to
            add, buy, or put items in the cart. Requires PRODUCT_REF values from
            current-turn search or historical-product resolution — not names.
            Call once with all items, not once per item.
            """

            return normalize_tool_result(_add_cart_items_impl(items))

        @tool(return_direct=False)
        def _remove_cart_item_impl(cart_line_id: str, quantity: int = 1):
            """Remove a cart line. Use ONLY on explicit shopper intent to remove
            an item. Requires CART_LINE_ID from get_cart_tool — do not guess.
            Use update_cart_items_tool to change quantity instead of removing
            and re-adding.
            """

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
            self._append_product_images(
                scope.retrieved,
                state.cart,
                scope.product_evidence.values(),
            )
            rendered = _format_cart_remove_result(
                result,
                fallback=f"Removed {quantity} {line['item']} from cart.",
            )
            if not result.ok:
                return rendered
            return committed_effect(
                rendered,
                operation="removed from cart",
                idempotency_key=(
                    f"{identity.request_id}:remove:{line['cart_line_id']}:{quantity}"
                ),
                cart_line_id=line["cart_line_id"],
                product_id=line["item"],
                quantity=quantity,
            )

        @tool(return_direct=False, response_format="content_and_artifact")
        def remove_cart_item_tool(cart_line_id: str, quantity: int = 1):
            """Remove a cart line. Use ONLY on explicit shopper intent to remove
            an item. Requires CART_LINE_ID from get_cart_tool — do not guess.
            Use update_cart_items_tool to change quantity instead of removing
            and re-adding.
            """

            return normalize_tool_result(
                _remove_cart_item_impl(cart_line_id, quantity)
            )

        def _update_cart_items_impl(cart_line_id: str, quantity: int):
            """Change the quantity of an item already in the cart, or remove it
            by setting quantity to 0. Use ONLY when the shopper explicitly asks
            to change a quantity or remove by quantity. Do NOT use for initial
            adds — use add_cart_items_tool. Do NOT guess the CART_LINE_ID; call
            get_cart_tool first if you do not have one.
            """

            result = update_cart_item(
                UpdateCartItemInput(
                    user_id=str(identity.cart_user_id),
                    cart_line_id=cart_line_id,
                    quantity=quantity,
                    idempotency_key=(
                        f"{identity.request_id}:update:{cart_line_id}:{quantity}"
                    ),
                ),
                self.config.memory_port,
            )
            state.cart = self._read_cart(identity.cart_user_id)
            self._append_product_images(
                scope.retrieved,
                state.cart,
                scope.product_evidence.values(),
            )
            rendered = _format_update_cart_result(result, state.cart)
            if not result.ok:
                return rendered
            return committed_effect(
                rendered,
                operation="cart quantity updated",
                idempotency_key=(
                    f"{identity.request_id}:update:{cart_line_id}:{quantity}"
                ),
                cart_line_id=cart_line_id,
                quantity=quantity,
            )

        @tool(
            args_schema=_UpdateCartItemsInput,
            return_direct=False,
            response_format="content_and_artifact",
        )
        def update_cart_items_tool(cart_line_id: str, quantity: int):
            """Set the exact quantity for one cart line. Use for quantity
            changes instead of removing and re-adding. Requires CART_LINE_ID
            from get_cart_tool.
            """

            return normalize_tool_result(
                _update_cart_items_impl(cart_line_id, quantity)
            )

        @tool(args_schema=_GetStorePolicyInput, return_direct=False)
        def get_store_policy_tool(
            topic: Literal[
                "returns",
                "shipping",
                "sizing",
                "payment",
                "price_match",
                "gift_cards",
            ],
        ) -> str:
            """Look up store policy for: returns, shipping, sizing, payment,
            price_match, or gift_cards. Use ONLY for these policy topics. Do
            NOT use for product facts, prices, or availability. If this tool
            returns a not-found error, relay the message to the shopper and
            direct them to the retailer's help center. Do NOT substitute model
            knowledge for a missing policy.
            """

            result = get_store_policy(
                GetStorePolicyInput(topic=topic),
                _store_policies_path(),
            )
            return _format_policy_result(result)

        @tool(args_schema=_CheckAvailabilityInput, return_direct=False)
        def check_product_availability_tool(
            product_ref: str,
            variant_hint: str | None = None,
        ) -> str:
            """Check whether a specific product is available or in stock. Use
            ONLY when the shopper explicitly asks about availability, stock,
            or a specific size. Requires a PRODUCT_REF established by search or
            historical-product resolution. Do NOT use for browsing. The
            deterministic stub reports general availability, sized availability
            for apparel and footwear, and one-size availability for other
            product categories.
            """

            product = scope.product_evidence.get(product_ref)
            if product is None:
                return (
                    f"PRODUCT_REF '{product_ref}' is unknown in this conversation. "
                    "Search this turn or resolve the earlier product first."
                )
            result = check_product_availability(
                CheckProductAvailabilityInput(
                    product_ref=product_ref,
                    variant_hint=variant_hint,
                ),
                product,
            )
            return _format_availability_result(result)

        @tool(return_direct=False)
        def check_active_promotions_tool() -> str:
            """Check whether a sale, discount, or promotion is currently active.
            Use ONLY when the shopper explicitly asks about promotion status. Do
            NOT use for ordinary affordable browsing, a price ceiling, price
            matching, or product availability. Catalog search does not establish
            sale status.
            """

            return _format_promotions_result(check_active_promotions())

        @tool(return_direct=False)
        def view_cart_total_tool() -> str:
            """Compute the cart subtotal. Use for budget checks or when the
            shopper asks for the total. Does not include tax or shipping. Use
            get_cart_tool for line contents.
            """

            cart = self._read_cart(identity.cart_user_id)
            state.cart = cart
            self._append_product_images(
                scope.retrieved,
                cart,
                scope.product_evidence.values(),
            )
            return _format_cart_total(cart)

        shopping_tools = [
            search_catalog_tool,
            get_product_details_tool,
            resolve_conversation_products_tool,
            get_cart_tool,
            add_cart_items_tool,
            remove_cart_item_tool,
            update_cart_items_tool,
            view_cart_total_tool,
            get_store_policy_tool,
            check_product_availability_tool,
            check_active_promotions_tool,
        ]
        validate_registered_tool_names(
            {
                str(
                    getattr(candidate, "name", None)
                    or getattr(candidate, "__name__", "")
                )
                for candidate in shopping_tools
            }
        )
        skill_gate = ShopperSkillActivationMiddleware(
            request_id=identity.request_id,
            skill_descriptions={
                name: skill.description
                for name, skill in skill_registry.items()
            },
            skill_tool_grants={
                name: skill.tools_granted
                for name, skill in skill_registry.items()
            },
            previous_selected_skills=state.previous_selected_skill_names,
        )
        tool_loop_control = ToolLoopControlMiddleware(
            catalog_context=format_catalog_capabilities_for_prompt(
                turn_capabilities
            ),
            shopper_statements=(
                state.query,
                *(turn.shopper_text for turn in state.dialogue),
            ),
        )

        @tool(args_schema=skill_activation_input, return_direct=False)
        def activate_shopper_skills_tool(
            skill_names: list[str],
        ) -> str:
            """Select and load shopper behavior skills for this turn. This is
            the required first step before answering or calling shopping tools.
            Select the smallest set whose registered descriptions cover the
            complete current intent. Use outfit-styling for outfit building,
            completion, refinement, or mid-browse styling questions; style-led
            single-piece selection, including a statement or balancing piece,
            also uses outfit-styling. Use product-discovery for search and browse
            without styling intent. These are alternative primary procedures,
            never a pair. Add budget-shopping only as a modifier when the shopper
            states a budget. Keep outfit-styling as the primary skill throughout
            an active outfit-building thread, including its single-piece follow-up
            searches.

            """

            selected_names = list(dict.fromkeys(skill_names))
            try:
                selected_files = {
                    skill_registry[name].path: skill_registry[name].content
                    for name in selected_names
                }
                activated = skill_gate.activate(selected_files, selected_names)
            except (KeyError, ValueError):
                skill_gate.fail()
                return (
                    "SHOPPER_SKILL_ACTIVATION_FAILED: Registered skill "
                    "instructions could not be loaded."
                )
            if not activated:
                return "SHOPPER_SKILL_ACTIVATION_ALREADY_COMPLETE"
            return (
                f"{SKILL_ACTIVATION_COMPLETE} "
                + ", ".join(selected_files)
            )

        activate_shopper_skills_tool.handle_validation_error = (
            skill_gate.handle_activation_validation_error
        )

        return create_deep_agent(
            model=self._create_chat_model(),
            tools=[activate_shopper_skills_tool, *shopping_tools],
            system_prompt=self._system_prompt(
                turn_capabilities,
                shopper_context=state.shopper_context,
            ),
            middleware=[tool_loop_control, skill_gate],
            backend=skills_backend,
            checkpointer=self._checkpointer,
        )

    async def _delete_turn_checkpoint(self, identity: RequestIdentity) -> None:
        try:
            delete_thread = getattr(self._checkpointer, "adelete_thread", None)
            if delete_thread is not None:
                await delete_thread(identity.checkpoint_thread_id)
            else:
                self._checkpointer.delete_thread(identity.checkpoint_thread_id)
        except Exception as exc:  # pragma: no cover - cleanup is best effort.
            logger.warning("Could not delete Deep Agents turn checkpoint: %s", exc)

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

    async def _rewrite_response_for_grounding(
        self,
        state: State,
        result: Any,
        draft_response: str,
        *,
        request_id: str,
        timeout_seconds: float | None = None,
    ) -> str:
        clarification = _catalog_repair_clarification_response(
            result,
            request_id=request_id,
        )
        if clarification:
            if _has_search_only_tool_evidence(
                result,
                request_id=request_id,
            ):
                grounded_response = self._rewrite_search_only_response(
                    state,
                    result,
                    request_id=request_id,
                )
                return _scrub_internal_shopper_language(
                    f"{grounded_response}\n\n{clarification}"
                )
            if not _has_successful_non_search_tool_evidence(
                result,
                request_id=request_id,
            ):
                return _scrub_internal_shopper_language(clarification)
            draft_response = clarification
        rejected_search_response = _rejected_catalog_search_response(
            result,
            request_id=request_id,
        )
        if rejected_search_response:
            return rejected_search_response

        max_evidence_chars = getattr(
            self.config,
            "grounding_rewrite_max_evidence_chars",
            12000,
        )
        current_evidence = _collect_tool_grounding_evidence(
            result,
            max_chars=max_evidence_chars,
            request_id=request_id,
        )
        search_only = bool(current_evidence) and _has_search_only_tool_evidence(
            result,
            request_id=request_id,
        )
        if not getattr(self.config, "grounding_rewrite_enabled", True):
            if search_only:
                return self._rewrite_search_only_response(
                    state,
                    result,
                    request_id=request_id,
                )
            return _scrub_internal_shopper_language(draft_response)
        if not draft_response:
            if not search_only:
                return draft_response
            return self._rewrite_search_only_response(
                state,
                result,
                request_id=request_id,
            )
        # No evidence is a reason to run the editor, not to skip it. A turn with
        # nothing to ground against is exactly the turn where the model
        # improvises about its own machinery -- the leak that reached a shopper
        # was "I don't have access to a catalog search tool (only cart tools)"
        # on a turn with no evidence, no cart, and no history. Skipping here left
        # a fixed list of literal replacements as the only guard, and a list can
        # never cover what a model might say. These turns also run no tools, so
        # the editor has the most deadline available, not the least.

        # Fail-closed exists to stop a draft asserting product facts that the
        # turn's evidence cannot support. A turn with no authority has no such
        # facts, so an editor failure there must not cost the shopper the whole
        # reply -- it degrades to the unedited draft, which is what shipped
        # before the editor ran on these turns at all.
        has_grounding_authority = _has_grounding_authority(state, current_evidence)
        termination_reason_before_editor = state.agent_diagnostics.get(
            "final_termination_reason"
        )

        start = time.monotonic()
        # Separated authority lanes. Each is labelled with what it may be used
        # for, because a lane that mixes intent with identity gives the editor
        # no way to tell which half can support a factual claim.
        prompt = (
            f"USER QUERY:\n{state.query}\n\n"
            "CURRENT-TURN TOOL EVIDENCE (facts established this turn):\n"
            f"{current_evidence or '(none)'}\n\n"
            "PRODUCTS SHOWN EARLIER (identity only — not current facts):\n"
            f"{format_historical_product_index(state.historical_product_sets) or '(none)'}\n\n"
            f"CURRENT CART (authoritative):\n{_format_cart(state.cart)}\n\n"
            "CONVERSATION (shopper intent only — never a product fact):\n"
            f"{state.dialogue_context or '(none)'}\n\n"
            f"AVAILABLE IMAGES:\n{_format_retrieved_images(state.retrieved)}\n\n"
            f"DRAFT RESPONSE:\n{draft_response}"
        )
        try:
            active_timeout = (
                self.config.deepagents_execution_timeout_seconds
                if timeout_seconds is None
                else timeout_seconds
            )
            if active_timeout <= 0:
                raise TimeoutError
            rewrite_result = await asyncio.wait_for(
                self._create_chat_model().ainvoke(
                    [
                        {
                            "role": "system",
                            "content": _GROUNDING_EDITOR_SYSTEM_PROMPT,
                        },
                        {"role": "user", "content": prompt},
                    ]
                ),
                timeout=active_timeout,
            )
        except TimeoutError:
            logger.warning(
                "Grounding response editor timed out for request %s",
                request_id,
            )
            state.timings["grounding_rewrite"] = time.monotonic() - start
            state.agent_diagnostics[
                "final_termination_reason"
            ] = "grounding_timeout"
            _add_model_usage(
                state,
                "app_llm_grounding_editor",
                status="failed",
                calls=1,
                detail="Final response grounding rewrite timed out",
            )
            if search_only:
                return self._rewrite_search_only_response(
                    state,
                    result,
                    request_id=request_id,
                )
            if not has_grounding_authority:
                # The turn itself completed; only the optional tidy-up failed,
                # and that is already recorded in model usage. Leaving an error
                # termination reason here would mark a good turn as failed.
                if termination_reason_before_editor is None:
                    state.agent_diagnostics.pop("final_termination_reason", None)
                else:
                    state.agent_diagnostics["final_termination_reason"] = (
                        termination_reason_before_editor
                    )
                return _scrub_internal_shopper_language(draft_response)
            return _GROUNDING_FAILURE_RESPONSE
        except Exception:  # noqa: BLE001 - response editor has a safe fallback.
            logger.exception("Grounding response editor failed")
            state.timings["grounding_rewrite"] = time.monotonic() - start
            state.agent_diagnostics["final_termination_reason"] = "grounding_error"
            _add_model_usage(
                state,
                "app_llm_grounding_editor",
                status="failed",
                calls=1,
                detail="Final response grounding rewrite failed closed",
            )
            if search_only:
                return self._rewrite_search_only_response(
                    state,
                    result,
                    request_id=request_id,
                )
            if not has_grounding_authority:
                # The turn itself completed; only the optional tidy-up failed,
                # and that is already recorded in model usage. Leaving an error
                # termination reason here would mark a good turn as failed.
                if termination_reason_before_editor is None:
                    state.agent_diagnostics.pop("final_termination_reason", None)
                else:
                    state.agent_diagnostics["final_termination_reason"] = (
                        termination_reason_before_editor
                    )
                return _scrub_internal_shopper_language(draft_response)
            return _GROUNDING_FAILURE_RESPONSE

        state.timings["grounding_rewrite"] = time.monotonic() - start
        state.token_usage = _merge_token_usage(
            state.token_usage,
            _collect_token_usage(rewrite_result),
        )

        rewritten = _content_to_text(_value(rewrite_result, "content"))
        if not rewritten:
            rewritten = _content_to_text(rewrite_result)
        if not rewritten.strip():
            state.agent_diagnostics["final_termination_reason"] = "grounding_error"
            _add_model_usage(
                state,
                "app_llm_grounding_editor",
                status="failed",
                calls=1,
                detail="Final response grounding rewrite returned empty output",
            )
            if search_only:
                return self._rewrite_search_only_response(
                    state,
                    result,
                    request_id=request_id,
                )
            if not has_grounding_authority:
                # The turn itself completed; only the optional tidy-up failed,
                # and that is already recorded in model usage. Leaving an error
                # termination reason here would mark a good turn as failed.
                if termination_reason_before_editor is None:
                    state.agent_diagnostics.pop("final_termination_reason", None)
                else:
                    state.agent_diagnostics["final_termination_reason"] = (
                        termination_reason_before_editor
                    )
                return _scrub_internal_shopper_language(draft_response)
            return _GROUNDING_FAILURE_RESPONSE
        _add_model_usage(
            state,
            "app_llm_grounding_editor",
            status="used",
            calls=1,
            detail="Final response grounding rewrite",
        )
        return _scrub_internal_shopper_language(rewritten)

    def _rewrite_search_only_response(
        self,
        state: State,
        result: Any,
        *,
        request_id: str,
    ) -> str:
        """Return the deterministic fallback for a search-only turn."""

        shopper_guidance = _search_guidance_evidence(
            result,
            request_id=request_id,
        )
        return self._build_search_only_response(
            state,
            result,
            request_id=request_id,
            intro=(
                "\n".join(shopper_guidance)
                or self._active_skill_response_guidance(state)
            ),
        )

    def _active_skill_response_guidance(self, state: State) -> str:
        active_paths = set(
            state.agent_diagnostics.get("skill_files_read", [])
        )
        registry = _shopper_skill_registry(self._shopper_skills_root())
        guidance = [
            skill.response_guidance
            for skill in registry.values()
            if skill.path in active_paths
        ]
        return "\n".join(guidance)

    def _build_search_only_response(
        self,
        state: State,
        result: Any,
        *,
        request_id: str,
        intro: str = "",
    ) -> str:
        """Return a deterministic facts-only fallback for search evidence."""

        return _format_search_only_response(
            state,
            result,
            request_id=request_id,
            intro=intro,
        )

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

    def _system_prompt(
        self,
        capabilities: CatalogCapabilities,
        *,
        shopper_context: ShopperContext | None = None,
    ) -> str:
        catalog_context = format_catalog_capabilities_for_prompt(capabilities)
        shopper_context_rules = (
            f"\n{_SHOPPER_CONTEXT_SYSTEM_RULES}\n"
            if shopper_context is not None
            else ""
        )
        prompt = f"""You are a retail shopping assistant for the products advertised by the active catalog.

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
{shopper_context_rules}

Catalog capabilities:
{catalog_context}

Rules:
- Every turn begins with shopper-skill activation. Select the smallest set of
  registered skills that covers the complete current intent, then follow the
  injected full instructions before constructing shopping-tool arguments or a
  final response. Never combine activation and shopping calls in one response.
- Outfit styling and product discovery are alternative primary procedures;
  never activate both. Budget shopping may accompany either only as a modifier
  when the shopper states a budget.
- Style-led fashion selection belongs to outfit styling even when the shopper
  asks for one statement or balancing piece. Product discovery is for browse
  and filter intent without styling judgment.
- Keep the primary skill aligned with the active conversation task. An outfit-
  building or styling thread continues to use outfit styling for piece-by-piece
  searches until the shopper changes tasks; do not reclassify a follow-up as
  product discovery merely because it asks for one product type.
{CATALOG_SEARCH_RULES}
- Put every non-taxonomy shopper must-have in `required_constraints`, including
  requirements whose fields are semantic/detail-only or absent from Catalog
  capabilities. Do not omit an unsupported must-have or rely on semantic search
  to enforce it; deterministic validation must refuse that search instead of
  weakening it. Preserve performance requirements such as water resistance or
  machine washability here as well. An attribute that defines the requested
  products is required even without the words "must have." Semantic relevance cannot guarantee
  a must-have requirement. Recommendation adjectives such as comfortable,
  relaxed, soft, breathable, lightweight, casual, dressy, bold, bright,
  vibrant, or sporty always remain semantic preferences, never objective hard
  filters. Before every search, compare each modifier on the target product
  with the hard filters advertised in Catalog capabilities and copy every exact matching value into
  `required_constraints`; never leave the object empty when one applies. Use only advertised
  filter properties directly. For "Do you have
  water-resistant bags?", include
  `{{"unadvertised_requirements": ["water resistance"]}}`; do not send an
  empty object.
- Apply hard constraints only to the target products named in the current turn.
  An anchor's confirmed color, material, or other attribute is styling context,
  not a hard filter for a complementary role, unless the shopper explicitly asks
  for the same value or palette.
- For required constraints advertised as hard filters, enum values must exactly
  match listed values and numeric values use an object with `min` and/or `max`.
  When one shopper constraint includes multiple applicable advertised enum
  values, include all of them in one list and one search rather than trying one
  value at a time.
- Media-only or descriptive media requests such as "what's in this look",
  "describe this outfit", "what am I wearing", or "what colors are here" must
  be answered from MEDIA ANALYSIS. Do not call search_catalog_tool and do not
  show catalog products unless the shopper explicitly asks to find, shop,
  recommend, compare, price-check, check availability, or add an item.
- Use at most {self.config.max_catalog_searches_per_turn} catalog search calls per
  user turn. One normalized taxonomy-and-required-constraint scope can execute
  only once in a turn; do not retry the same hard-filter scope with different
  semantic wording. For outfit requests
  with multiple required item types, run one focused search per distinct
  taxonomy scope, then stop and synthesize from those results.
- An outfit request with a season, weather need, occasion, or style/vibe already
  has enough direction to begin with a grounded partial outfit. Do not answer
  only with a questionnaire; search the most useful core role first and ask at
  most one concise follow-up while presenting the grounded result.
- An unspecified request for one style-led piece, such as a statement piece,
  does not identify a product role. Ask one concise category or occasion question
  before searching. This does not apply to an outfit or complete-look request,
  where the named vibe, occasion, season, or weather need is enough to begin.
- A whole or complete outfit remains incomplete until current or directly
  referenced prior evidence covers multiple complementary roles, or the search
  cap is reached and the missing role is disclosed. A one-piece dress may be the
  clothing core, but does not by itself complete an outfit request.
- For a broad style/vibe request, select a useful core role from exact taxonomy
  values currently advertised by the catalog. Do not invent an unadvertised
  product type from the vibe or copy a generic styling example into taxonomy.
- Treat broad weather or occasion context as styling direction, not automatically
  as a product-attribute guarantee. A "rainy day outfit" or "wet-weather outfit"
  should search practical
  advertised roles without adding an unadvertised requirement; add water
  resistance to `unadvertised_requirements` only when the shopper directly
  requires it for a product. "Rainy day outfit" does not imply water resistance;
  "water-resistant bags" directly requires it.
- Subjective style/vibe language is semantic direction unless the shopper makes
  it an explicit hard requirement. Objective product attributes such as material,
  weather performance, or a specific shade remain must-haves when they define
  the requested product.
- For alternatives joined by "or", search the faithful advertised branch and
  preserve every named branch. If another branch cannot be mapped faithfully,
  present the supported result and ask one concise clarification before any
  adjacent search. Do not reject a supported branch merely because another
  alternative is unresolved, and never list the supported taxonomy value in
  `unadvertised_requirements`.
- When every named alternative is advertised, include all of them in one call.
  Do not narrow an explicit umbrella or alternatives to one convenient type.
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
- A request for one product role gets one inclusive search scope containing all
  faithful advertised types for that role. Do not use remaining search budget
  on adjacent categories, one-piece substitutes, or unrelated product types.
  A dress is not a bottom and does not satisfy a request for separates.
- In an active styling thread, a group of recent candidates can be the direct
  antecedent. If they share a confirmed constraint that is sufficient for the
  next request, use it as the provisional anchor and search the requested role.
  Do not require one exact product selection or an occasion when the shopper is
  asking for generally compatible options and the shared anchor is sufficient.
- In the final response to a follow-up search, explicitly connect the new
  candidates to the named antecedent or to that candidate group's shared
  confirmed constraint. Do not return an unexplained product list.
- Product-detail or research questions about a product already returned by
  search_catalog_tool should use get_product_details_tool with that
  PRODUCT_REF. Do not run another broad catalog search for known-product facts.
- Initial recommendations should use product name, price, category or role,
  and one styling reason. Do not enumerate materials, dimensions, pockets,
  closures, care, or construction details unless the shopper asks for those
  details and you have called get_product_details_tool.
- For search-only recommendations, keep every product line to name, price,
  category/role, image availability when useful, and exact confirmed filters.
  Put the styling reason in a separate sentence based on role and shopper
  context; never derive it by interpreting words in the display name.
- A successful search may report confirmed filters. Every returned product
  passed each reported predicate. A single allowed value confirms that value;
  multiple allowed values confirm membership in the set, not which value each
  product has. Do not infer an adjacent attribute from a confirmed filter.
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
  product names and prices. Catalog results are not inventory evidence.
- If MEDIA ANALYSIS says media analysis failed, VLM authentication failed, the
  VLM is unavailable, or video understanding is not configured, say so plainly.
  Do not infer video-similar products from the media; ask the shopper for a
  text description or search only from explicit text in the shopper request.
  If an image is attached, image embedding search through search_catalog_tool is
  still available even when MEDIA ANALYSIS is unavailable.
- Cart reads require get_cart_tool. Cart totals require view_cart_total_tool.
- Use recent discussion, not CURRENT CART, to resolve ordinary product and
  styling references such as "that" and "those." A discussed anchor does not
  need to be in the cart for styling advice. Mention that an item is absent from
  the cart only when the shopper asks about cart contents or a cart mutation.
- Cart mutations require explicit shopper intent and must use
  add_cart_items_tool, remove_cart_item_tool, or update_cart_items_tool. Never
  claim a cart mutation unless the tool reports success.
- Cart mutation scope must match the shopper's explicit add or remove request.
  Selection, approval, or styling preference is not cart intent by itself.
  If the shopper asks to "add those", add only the items named in that add
  request or its direct antecedent. Do not add earlier anchor, core outfit, or
  optional pieces unless the shopper explicitly includes them in the cart
  request.
- For an explicit cart swap, finish the whole swap before the final response:
  remove the rejected cart line, add the selected replacement when a valid
  PRODUCT_REF is already available, then summarize the updated cart. If the
  replacement is from an earlier turn, resolve it first. Search only for a new
  replacement that has not already been presented.
- If cart mutation scope is ambiguous, ask one concise clarification before
  calling any cart mutation tool. Example: "Do you want me to add just the bag,
  layer, and earrings, or the full outfit including the dress and sandals?"
- For cart styling requests, inspect CURRENT CART or call get_cart_tool first.
  Do not search for products already named as cart contents just to verify them.
  If the cart is empty but the shopper names items, say you do not see those
  items in the cart yet, then give provisional styling advice from the named
  items without claiming cart truth. Search at most once for a missing piece
  only after identifying the gap.
- Use PRODUCT_REF established by current-turn search or
  resolve_conversation_products_tool when adding items. Do not pass display
  names as product_ref values to add_cart_items_tool. Include
  expected_display_name for each item so the tool can verify that the selected
  PRODUCT_REF resolves to the shopper-facing product name you intend to add.
- When the shopper asks to add multiple selected products, call
  add_cart_items_tool once with an item list. The tool may report partial
  success; the final answer must clearly distinguish added items from failures.
- Use PRODUCT_REF established by current-turn search or historical-product
  resolution when requesting product details. Do not pass display names to
  get_product_details_tool.
- Use internal identifiers only in tool calls. Do not expose PRODUCT_REF or
  CART_LINE_ID in customer-facing responses.
- Use CART_LINE_ID from CURRENT CART or get_cart_tool when removing an item. Do
  not guess cart line IDs from product names.
- Use update_cart_items_tool for quantity changes. Set quantity to zero only
  when the shopper explicitly asks to remove that line.
- Store policy questions about returns, shipping, sizing, payment, price
  matching, or gift cards require get_store_policy_tool. Never substitute model
  knowledge for policy content that the tool does not return.
- Explicit stock, inventory, or size availability questions
  require check_product_availability_tool with a PRODUCT_REF from a prior
  search. Relay its deterministic result rather than guessing from catalog
  presence.
- Explicit sale, discount, or promotion questions require
  check_active_promotions_tool. Catalog results and prices cannot establish sale
  status. If no promotion is active and sale status is required, do not search
  regular-price products without the shopper's agreement; continue any separate
  requested work from the same turn.
- If the shopper asks for anything under a budget without a product type,
  category, occasion, style, outfit goal, or image, ask one concise clarifying
  question instead of guessing.
- Tax and delivery dates are not available through the current tools. Do not
  treat a catalog result alone as proof that an item is in stock or ready to
  ship; availability claims require check_product_availability_tool.
- Previous preference context is guidance only. The current shopper request
  wins when it conflicts with previous preferences.
- Keep final answers concise and grounded in tool results. Attribute materials,
  comfort, construction, and outdoor-practicality claims to the specific items
  that support them instead of making unsupported whole-outfit claims. Avoid
  guarantee language such as "will stay comfortable all evening" unless the
  catalog evidence supports the guarantee. Before finalizing, remove or soften
  unsupported phrases about grass, gravel, water resistance, all-day comfort,
  maximum breathability, or best-in-category performance.
"""
        return prompt

    def _build_user_message(self, state: State, identity: RequestIdentity) -> str:
        sections = [
            (
                f"REQUEST ID: {identity.request_id}\n"
                f"SESSION ID: {identity.session_id}\n"
                f"CONVERSATION ID: {identity.conversation_id}\n"
                f"CART ID: {identity.cart_id}"
            )
        ]
        shopper_context = _format_shopper_context(state.shopper_context)
        if shopper_context:
            sections.append(shopper_context)
        sections.extend(
            [
                (
                    f"USER QUERY: {state.query}\n"
                    f"IMAGE ATTACHED: {'yes' if state.image else 'no'}"
                ),
                f"MEDIA ATTACHED:\n{_format_media_summary(state.media)}",
                f"MEDIA ANALYSIS:\n{state.media_analysis or '(none)'}",
                f"CURRENT CART:\n{_format_cart(state.cart)}",
                f"RECENT DISCUSSION:\n{state.context or '(none)'}",
            ]
        )
        return "\n\n".join(sections)

    @staticmethod
    def _append_product_images(
        retrieved: dict[str, str],
        cart: Cart,
        products: tuple[ProductSummary, ...],
    ) -> None:
        if not cart.contents or not products:
            return

        products_by_key: dict[str, ProductSummary] = {}
        for product in products:
            products_by_key[product.product_id] = product
            products_by_key[product.display_name] = product

        for item in cart.contents:
            product = products_by_key.get(str(item.get("product_id") or ""))
            if product is None:
                product = products_by_key.get(str(item.get("item") or ""))
            if product is not None and product.image_url:
                retrieved[product.display_name] = product.image_url

    def _safe_read_cart(self, user_id: int) -> Cart | None:
        """Read the cart for a failure receipt without raising again."""

        try:
            return self._read_cart(user_id)
        except Exception:  # noqa: BLE001 - a receipt must not fail closed twice.
            logger.exception("Could not read cart for mutation receipt")
            return None

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

    def _start_conversation_turn(
        self,
        state: State,
        identity: RequestIdentity,
    ) -> TurnStartResult | None:
        """Start one durable turn and apply its context/cart snapshot."""

        start = time.monotonic()
        try:
            turn = self._conversation_memory.start_turn(
                identity.conversation_id,
                request_id=identity.request_id,
                shopper_text=state.query,
                media=state.media,
                cart_user_id=identity.cart_user_id,
                shopper_profile_id=identity.shopper_profile_id,
            )
            if (identity.shopper_profile_id is None) != (
                turn.shopper_context is None
            ) or (
                turn.shopper_context is not None
                and not isinstance(turn.shopper_context, ShopperContext)
            ):
                raise ConversationMemoryError(
                    "shopper_context_invalid",
                    "Conversation memory returned mismatched shopper context.",
                )
        except (ConversationMemoryError, ValidationError) as exc:
            logger.error("Failed to start durable conversation turn: %s", exc)
            state.dialogue = []
            state.historical_product_sets = []
            state.dialogue_context = ""
            state.context = ""
            state.cart = Cart()
            state.shopper_context = None
            error_code = getattr(exc, "code", "memory_start_payload_invalid")
            if error_code == "shopper_profile_not_found":
                state.response = _SHOPPER_PROFILE_NOT_FOUND_RESPONSE
                state.agent_diagnostics = _empty_agent_diagnostics(error_code)
            elif error_code == "conversation_profile_mismatch":
                state.response = _CONVERSATION_PROFILE_MISMATCH_RESPONSE
                state.agent_diagnostics = _empty_agent_diagnostics(error_code)
            elif getattr(exc, "status_code", None) == 409:
                if error_code in {"turn_in_progress", "conversation_turn_in_progress"}:
                    state.response = (
                        "This conversation is still processing another request. "
                        "Please retry shortly."
                    )
                elif error_code == "turn_abandoned":
                    state.response = (
                        "That earlier request was interrupted. Please retry with "
                        "a new request."
                    )
                elif error_code == "turn_superseded":
                    state.response = (
                        "That interrupted request was superseded by a newer turn. "
                        "Please continue from the latest conversation state."
                    )
                else:
                    state.response = (
                        "That request identifier was already used for different "
                        "input. Please retry with a new request."
                    )
                state.agent_diagnostics = _empty_agent_diagnostics(error_code)
            else:
                state.response = (
                    "I cannot safely load this conversation right now. "
                    "Please retry shortly."
                )
                state.agent_diagnostics = _empty_agent_diagnostics(
                    "memory_start_failed"
                )
                state.agent_diagnostics["memory_start_error"] = getattr(
                    exc,
                    "code",
                    "memory_start_payload_invalid",
                )
            return None
        finally:
            state.timings["memory"] = time.monotonic() - start

        state.dialogue, state.dialogue_context = build_dialogue_context(
            turn.recent_turns,
            max_chars=max(1000, int(self.config.memory_length)),
        )
        state.context = state.dialogue_context
        state.previous_selected_skill_names = list(
            turn.previous_selected_skill_names
        )
        state.shopper_context = turn.shopper_context
        state.historical_product_sets = [
            entry
            for entry in (turn.projection.product_reference_index or [])
            if isinstance(entry, dict)
        ]
        historical_products = format_historical_product_index(
            turn.projection.product_reference_index
        )
        if historical_products:
            state.context = "\n\n".join(
                value for value in (state.context, historical_products) if value
            )
        state.cart = Cart(
            contents=[
                item.model_dump(mode="json", exclude_none=True) for item in turn.cart
            ]
        )
        return turn

    def _restore_replayed_turn(
        self,
        state: State,
        turn: TurnStartResult,
    ) -> State:
        """Restore a finalized turn without repeating model or tool work."""

        state.response = turn.assistant_text or (
            "That earlier request did not complete. Please retry with a new request."
        )
        if turn.output is not None:
            state.product_results = [
                product.model_dump(mode="json")
                for product in turn.output.product_results
            ]
            state.retrieved = dict(turn.output.retrieved)
            state.agent_diagnostics = dict(turn.output.agent_diagnostics)
            state.selected_skill_names = list(turn.output.selected_skill_names)
        else:
            state.agent_diagnostics = _empty_agent_diagnostics(
                turn.termination_reason or "durable_turn_replayed"
            )
        return state

    def _finalize_conversation_turn(
        self,
        state: State,
        identity: RequestIdentity,
        turn: TurnStartResult,
        *,
        status: FinalTurnStatus | None = None,
        termination_reason: str | None = None,
        present_products: bool = True,
    ) -> bool:
        """Persist one terminal turn without changing its shopper response."""

        reason = termination_reason or str(
            state.agent_diagnostics.get("final_termination_reason") or "completed"
        )
        final_status = status or _conversation_turn_status(reason)
        state.agent_diagnostics["final_termination_reason"] = reason
        start = time.monotonic()
        finalized = False
        try:
            output = TurnReplayOutput(
                product_results=(state.product_results if present_products else []),
                retrieved=(state.retrieved if present_products else {}),
                agent_diagnostics=state.agent_diagnostics,
                selected_skill_names=state.selected_skill_names,
            )
            self._conversation_memory.finalize_turn(
                identity.conversation_id,
                turn.turn_id,
                request_id=identity.request_id,
                attempt_id=turn.attempt_id,
                assistant_text=state.response,
                status=final_status,
                termination_reason=reason,
                output=output,
            )
            finalized = True
        except (ConversationMemoryError, ValidationError) as exc:
            logger.error("Failed to finalize durable conversation turn: %s", exc)
            error_code = getattr(
                exc,
                "code",
                "memory_finalize_payload_invalid",
            )
            state.agent_diagnostics["memory_finalize_error"] = error_code
            if error_code == "turn_attempt_superseded":
                state.response = (
                    "This request was superseded by a newer attempt. "
                    "Please use the latest response."
                )
                state.product_results = []
                state.retrieved = {}
                state.agent_diagnostics["final_termination_reason"] = error_code
        finally:
            state.timings["memory"] = state.timings.get("memory", 0.0) + (
                time.monotonic() - start
            )
        return finalized

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














































































































































































#: Keys the catalog returns alongside the declared detail fields that are not
#: product attributes: retrieval bookkeeping, and taxonomy which has its own
#: lane. catalog_text is the prose serialisation of the same attributes and is
#: deliberately not forwarded -- it carries a marketing summary, and separating
#: the two would mean parsing prose.






























































