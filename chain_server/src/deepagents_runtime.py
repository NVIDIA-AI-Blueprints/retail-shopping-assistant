# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deep Agents SDK runtime for the shopping assistant."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date as CalendarDate, datetime, timezone

import logging

import asyncio
import contextlib
import json
import os
from pathlib import Path
import time
from typing import Any, AsyncIterator, Literal

from langgraph.errors import GraphRecursionError
from pydantic import (
    field_validator,
    BaseModel,
    Field,
    ValidationError,
)
import requests

from .agenttypes import Cart, ShopperContext, State
from .catalog_search import SearchContext, search_catalog
from .response_format import (
    format_catalog_shape,
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
    _format_store_date,
    _format_weather_result,
    WEATHER_BUDGET_EXHAUSTED,
    WEATHER_NO_DATE,
    WeatherForecastInput as _WeatherForecastInput,
    weather_call_needs_a_date,
    claim_weather_call,
    _format_wearer_audience,
    _format_update_cart_result,
    format_cart_change,
)
from .catalog_execution import execute_catalog_search
from .catalog_request import CatalogSearchPlan
from .turn_support import (
    _a_list_written_as_json_text,
    _append_product_results,
    _detail_fields_already_held,
    _search_catalog_scopes_input_model,
    _system_identification_events,
    _turn_audience_events,
    AddCartItemsToolItemInput,
    RequestIdentity,
    _add_model_usage,
    _cart_size_issue,
    _cart_line_size,
    _cart_product_choice_note,
    format_most_recent_subject,
    _identified_in_the_current_showing,
    _most_recently_shown,
    _images_in_product_order,
    _in_presentation_order,
    _shopper_words_this_conversation,
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
    _products_found_receipt,
    _advertised_sizes,
    _ONE_SIZE,
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
from .media_summary import summarize_media_analysis
from .media_perception import MediaPerceptionClient
from .skill_activation import (
    SKILL_ACTIVATION_COMPLETE,
    ShopperSkillActivationError,
    ShopperSkillActivationMiddleware,
    selected_skill_names_for_turn,
)
from .turn_scope import TurnScope
from .weather import WeatherConfig, WeatherRequest, build_weather_client
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


def _emit_media_analysis(on_progress: Any, state: State) -> None:
    """Send what the vision model saw, if anything and if anyone is listening.

    Failing here must never cost a turn: this is a progress message, and a turn
    that answered correctly but could not describe its own perception step is
    still a turn that answered correctly.
    """

    if on_progress is None or not getattr(state, "media", None):
        return
    try:
        summary = summarize_media_analysis(state.media_analysis or "")
        if not summary:
            return
        on_progress(
            json.dumps(
                {
                    "type": "media_analysis",
                    "payload": summary,
                    "timestamp": time.time(),
                }
            )
        )
    except Exception as exc:  # noqa: BLE001 - progress never breaks a turn.
        logger.warning("Could not emit media analysis: %s", type(exc).__name__)


def _turn_trace_session(identity: RequestIdentity):
    """Bind this turn's spans to its conversation, not to its graph thread.

    The LangChain instrumentor derives a session from LangGraph's ``thread_id``,
    which is ``[conversation_id, request_id]`` -- unique per turn. Left alone,
    a twenty-turn conversation becomes twenty single-turn sessions and the whole
    point of a session view is lost. Binding ``conversation_id`` explicitly is
    what makes a trace session and a durable conversation the same set of turns.

    Returns a null context when tracing is not installed, so this costs nothing
    in a deployment that never exports a span.
    """

    try:
        from openinference.instrumentation import using_attributes
    except Exception:  # noqa: BLE001 - tracing must never break a turn.
        return contextlib.nullcontext()

    try:
        return using_attributes(
            session_id=identity.conversation_id,
            user_id=str(identity.context_user_id),
        )
    except Exception as exc:  # noqa: BLE001 - same.
        logger.warning("Could not bind trace session: %s", type(exc).__name__)
        return contextlib.nullcontext()


@contextlib.contextmanager
def _turn_span(identity: RequestIdentity):
    """Open one span per turn, parenting the graph and the grounding editor.

    The graph and the grounding editor are two separate top-level invocations,
    so without this a turn arrives as two unrelated traces. This also gives the
    finished diagnostics somewhere to hang: every span the instrumentor raises
    has closed by the time the blob settles.
    """

    try:
        from opentelemetry import trace
    except Exception:  # noqa: BLE001 - tracing must never break a turn.
        yield None
        return

    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("turn") as span:
        try:
            # The instrumentor's context attributes reach the spans it raises,
            # not one opened by hand, so the root states its own kind and
            # session or it renders as UNKNOWN and sits outside the session.
            span.set_attribute("openinference.span.kind", "CHAIN")
            span.set_attribute("session.id", identity.conversation_id)
            span.set_attribute("conversation.id", identity.conversation_id)
            span.set_attribute("request.id", identity.request_id)
        except Exception:  # noqa: BLE001 - same.
            pass
        yield span


def _record_turn_diagnostics(span: Any, state: State) -> None:
    """Attach the settled diagnostics to the turn span as metadata.

    Everything rides the one supported ``metadata`` channel rather than a
    private attribute namespace, including the scalar counts -- viewers filter
    on metadata subkeys, so the countables stay filterable without inventing
    names nothing else understands.

    ``partial_graph_messages`` is excluded: it is the largest field by far and
    the span tree already holds what it would say.
    """

    if span is None:
        return
    try:
        diagnostics = dict(getattr(state, "agent_diagnostics", None) or {})
        diagnostics.pop("partial_graph_messages", None)
        tool_calls = diagnostics.get("tool_calls") or []
        payload = {
            "termination_reason": diagnostics.get("final_termination_reason"),
            "tool_calls": len(tool_calls),
            "tool_calls_rejected": len(diagnostics.get("rejected_tool_calls") or []),
            "products_shown": len(diagnostics.get("product_evidence") or []),
            "zero_result_scopes": len(
                diagnostics.get("catalog_scope_outcomes") or []
            ),
            "skills": diagnostics.get("skill_files_read") or [],
            "tools": [call.get("tool_name") for call in tool_calls],
            # Serialised, not nested. A viewer flattens nested metadata into one
            # attribute per leaf, and the full blob explodes into ~120 keys --
            # `product_evidence.0.facts.heel_type` and the like -- which sorts
            # the handful of fields worth reading to the bottom of a wall. The
            # per-tool detail is already on the tool spans; this is the derived
            # view, kept whole and out of the way.
            "diagnostics_json": json.dumps(diagnostics, default=str),
        }
        span.set_attribute("metadata", json.dumps(payload, default=str))
    except Exception as exc:  # noqa: BLE001 - diagnostics never break a turn.
        logger.warning("Could not record turn diagnostics: %s", type(exc).__name__)


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
- Never infer a shopper's location, the weather, or a seasonal need. Nothing in
  this context establishes any of them, and naming one is an invented fact."""
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
- A live forecast in TOOL EVIDENCE supports what the weather will be. Keep the
  provider attribution and its link whenever any of it survives into the reply:
  the provider's terms require it wherever weather or anything derived from it
  is shown, so removing it as clutter is not an option available to you. A
  forecast never confirms a product property.
- Styling judgement about an occasion is not a product claim, and must be kept
  rather than removed. "A stiletto will sink into grass" reasons from a
  confirmed heel type about the setting; "these are stable on grass" asserts a
  property of the shoe. Remove the second, keep the first. Advice that the
  shopper will need something the catalog does not stock is also judgement, not
  a claim, and stays.
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
_MEDIA_TURN_RULES = """- Media-only or descriptive media requests such as "what's in this look",
  "describe this outfit", "what am I wearing", or "what colors are here" must
  be answered from MEDIA ANALYSIS. Do not call search_catalog_tool and do not
  show catalog products unless the shopper explicitly asks to find, shop,
  recommend, compare, price-check, check availability, or add an item.
- If an image is attached, the current image is already available to
  search_catalog_tool. Use that tool for "this", "similar", and image-price
  refinement requests.
- If MEDIA ANALYSIS is present, use it as the visual/video understanding of
  the attached media. It can guide search_catalog_tool queries and follow-up
  pronoun resolution, but catalog results remain the source of truth for
  product names and prices. Catalog results are not inventory evidence.
- MEDIA ANALYSIS is what the media actually showed. It is your sight of the
  attached image or video: speak from it with confidence, name what it saw, and
  never tell the shopper you could not view their media when an analysis is
  present.
- What it is not is catalog vocabulary. Its words describe what was seen, not
  what the catalog can filter on, so treat each term as you would the same word
  from the shopper: map it to an advertised value before placing it in
  required_constraints, and carry what has no advertised value in
  unadvertised_requirements. Never copy a term out of MEDIA ANALYSIS into
  required_constraints unchanged, whatever field it came from -- including
  constraints_detected, which records what was observed and not what may be
  filtered on.
- When the media shows several garments and the shopper asked about one, search
  for the one they asked about. Name what else you saw; do not search it unasked.
- If MEDIA ANALYSIS says media analysis failed, VLM authentication failed, the
  VLM is unavailable, or video understanding is not configured, say so plainly.
  Do not infer video-similar products from the media; ask the shopper for a
  text description or search only from explicit text in the shopper request.
  If an image is attached, image embedding search through search_catalog_tool is
  still available even when MEDIA ANALYSIS is unavailable.
"""
def _today_for_the_shopper() -> str:
    """The date the shopper is shopping on, written the way they would say it.

    Read at request time rather than build time: an image that has been running
    a week would otherwise date every conversation to the day it was built.
    """

    return datetime.now(timezone.utc).strftime("%A %d %B %Y")


_EXCLUDED_DEEP_AGENT_TOOLS = frozenset(
    {"write_todos", "ls", "write_file", "edit_file", "glob", "grep", "execute"}
)
#: `BASE_AGENT_PROMPT` is written for a coding agent: it teaches a todo list, a
#: filesystem, "read files before editing", and a task-completion protocol. None
#: of it applies to a shopping assistant, and every tool it names is in
#: `_EXCLUDED_DEEP_AGENT_TOOLS`, so it spent 3,862 characters instructing the
#: model to call tools it had not been given. `base_system_prompt` is the
#: framework's own slot for replacing it; the shopping instructions the agent
#: does need are assembled in `_system_prompt` and passed as `system_prompt`,
#: which sits ahead of this base.
_DEEP_AGENT_BASE_PROMPT = """You have no filesystem, no shell, and no todo list.
The tools you are given are the only ones that exist; there is no planning or
bookkeeping step before using them."""
#: Historical-product resolutions allowed per turn while none has resolved. A
#: resolution that succeeds ends the budget immediately; this only bounds the
#: corrections a failing one may attempt.
_MAX_PRODUCT_RESOLUTION_ATTEMPTS = 2
#: Characters a model may wrap an opaque identifier in, matching the resolver's
#: own tolerance so both lanes read a ref the same way.
_REFERENCE_WRAPPERS = "<>[]{}\"'`"
#: Names looked up in the catalog when a reference resolves to nothing. Bounded
#: because each is a retrieval, and a turn that names more than two products the
#: assistant never showed is a conversation to have, not a batch to satisfy.
#: The longest one model request may take before it is abandoned, whatever the
#: deployment's turn budget. A request slower than this has stalled rather than
#: thought: the median turn costs ten seconds end to end.
_MODEL_REQUEST_TIMEOUT_CEILING_SECONDS = 40.0
_MAX_NAME_LOOKUPS = 2
















































































class AddCartItemsToolInput(BaseModel):
    _accept_items_as_text = field_validator("items", mode="before")(
        _a_list_written_as_json_text
    )

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
    size: str | None = Field(
        default=None,
        description=(
            "Do not use. A size is a different cart line, not a property of "
            "one, and this tool cannot change it. Declared here only so the "
            "attempt is refused with the correct sequence rather than "
            "silently ignored."
        ),
    )

    @field_validator("size", mode="before")
    @classmethod
    def _a_number_is_a_size_too(cls, value: Any) -> Any:
        """Take a size written as a number, so the refusal can be reached.

        This field exists for one reason: to turn "change it to a 7" into the
        add-then-remove sequence instead of a silent no-op. A model that sent
        `size: 7` rather than `size: "7"` never got there -- pydantic refused
        the call for the type, three times running, with a validation error
        that says nothing about carts. It then gave up, sent the quantity
        alone, and told the shopper it had updated a dress it had never been
        asked about.

        Sizes are "2" and "onesize" in this catalog, so a bare number is the
        obvious slip. Coercing it costs nothing and delivers the guidance the
        field was declared for.
        """

        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return str(int(value) if float(value).is_integer() else value)
        return value


class _DescribeCatalogInput(BaseModel):
    """No arguments. The shape is published; there is nothing to narrow."""


class _GetStorePolicyInput(BaseModel):
    topic: Literal[
        "returns",
        "shipping",
        "sizing",
        "payment",
        "price_match",
        "gift_cards",
    ] = Field(description="Policy topic to look up.")


class _AvailabilityItemInput(BaseModel):
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


class _CheckAvailabilityInput(BaseModel):
    items: list[_AvailabilityItemInput] = Field(
        ...,
        min_length=1,
        max_length=20,
        description=(
            "Every product the shopper asked about, in one call. They are "
            "checked together, so four products cost one round trip, not four."
        ),
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
        # Fails closed when weather is disabled or unconfigured: every call
        # returns a typed failure, which the reply degrades into occasion
        # styling. So an operator who never sets a key sees a shop that styles
        # without weather, not a shop that breaks.
        self._weather_client = build_weather_client(
            getattr(config, "weather", None) or WeatherConfig()
        )

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
        # The turn runs as a task so progress can be emitted while it is still
        # working. Everything below still waits for it: this adds events during
        # the turn, it does not change what is sent at the end of one.
        progress: asyncio.Queue = asyncio.Queue()
        _FINISHED = object()

        async def _run() -> State:
            try:
                return await self._run_turn(
                    state, identity, on_progress=progress.put_nowait
                )
            finally:
                # In a finally so a failed turn cannot leave the drain below
                # waiting on a queue nothing will ever write to again.
                progress.put_nowait(_FINISHED)

        turn = asyncio.create_task(_run())
        try:
            while True:
                event = await progress.get()
                if event is _FINISHED:
                    break
                yield event
        except BaseException:
            # A shopper who closes the tab or resets stops this generator. The
            # turn is a task now, so it does not hear that on its own: before
            # the queue it was awaited inline and the cancellation reached it.
            # Measured without this: a client that hung up at 8s still cost
            # three more LLM calls. Cancel it and let the timeout own the rest.
            turn.cancel()
            raise
        # Re-raises whatever the turn raised, so failure handling is unchanged.
        output = await turn
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
        on_progress: Any = None,
    ) -> State:
        """Run one turn inside its conversation's trace session."""

        with _turn_trace_session(identity), _turn_span(identity) as span:
            try:
                return await self._run_turn_inner(
                    state, identity, on_progress=on_progress
                )
            finally:
                # In a finally, not on the success path: a failed turn is
                # exactly when the trace is worth having.
                _record_turn_diagnostics(span, state)

    async def _run_turn_inner(
        self,
        state: State,
        identity: RequestIdentity,
        on_progress: Any = None,
    ) -> State:
        state.user_id = identity.context_user_id
        state.agent_diagnostics = _empty_agent_diagnostics("not_started")
        state.previous_selected_skill_names = []
        state.selected_skill_names = []
        state.shopper_profile_id = identity.shopper_profile_id
        state.shopper_context = None
        state.wearer_audience = []
        state.assumed_audience = []
        state.disclosed_audience = []
        turn = self._start_conversation_turn(state, identity)
        if turn is not None and turn.replayed:
            await self._delete_turn_checkpoint(identity)
            return self._restore_replayed_turn(state, turn)
        if turn is None and state.response:
            return state

        # Snapshot the cart before any tool can touch it. The effect of the
        # turn is then a computed fact rather than something a model infers.
        state.cart_at_turn_start = state.cart.model_copy(deep=True)

        try:
            output = await self._execute_turn(
                state, identity, on_progress=on_progress
            )
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
        on_progress: Any = None,
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
        # Emitted here rather than with the rest of the turn: the analysis is
        # complete and the catalog work has not started, so a shopper sees what
        # was seen in their media seconds before the products arrive.
        _emit_media_analysis(on_progress, state)
        if _should_short_circuit_media_failure(state):
            state.response = _media_failure_response(state.media_analysis)
            state.timings["deepagents"] = time.monotonic() - start
            state.agent_diagnostics = _empty_agent_diagnostics("media_failure")
            return state

        turn_capabilities = await asyncio.to_thread(self._catalog_capabilities.get)
        invoke_config = {
            "configurable": {"thread_id": identity.checkpoint_thread_id},
            "recursion_limit": self.config.deepagents_recursion_limit,
            # Trace-only. The checkpoint thread is deliberately request-scoped,
            # so a tracer left to infer a session from it files every turn as
            # its own conversation. This names the conversation for spans raised
            # inside the graph; it is read by tracing and by nothing else.
            "metadata": {"session_id": identity.conversation_id},
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
            # Attribute the agent loop to app_llm. Without this the shopper-
            # facing model list shows embeddings and guardrails with call
            # counts and the LLM with none, which reads as the LLM not running.
            if state.token_usage.get("model_calls"):
                _add_model_usage(
                    state,
                    "app_llm",
                    status="used",
                    calls=int(state.token_usage.get("model_calls") or 0),
                    detail="Planning, tool use, and response generation",
                    tokens=int(state.token_usage.get("total_tokens") or 0),
                )
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
                # The work is already done and paid for. Answering from it
                # beats asking the shopper to run the turn again.
                state.response = _products_found_receipt(state)
                state.agent_diagnostics["final_termination_reason"] = (
                    "incomplete_agent_response_answered_from_evidence"
                    if state.response
                    else "incomplete_agent_response"
                )
            if not state.response:
                state.response = (
                    "I could not complete that shopping request. Please try again."
                )
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
                    base_system_prompt=_DEEP_AGENT_BASE_PROMPT,
                    excluded_tools=_EXCLUDED_DEEP_AGENT_TOOLS,
                    excluded_middleware=frozenset({"TodoListMiddleware"}),
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
        wearer_audience_field = str(
            getattr(self.config, "wearer_audience_field", "") or ""
        )
        search_input_model = _search_catalog_tool_input_model(
            turn_capabilities,
            wearer_audience_field=wearer_audience_field,
        )
        search_tool_arguments_model = _search_catalog_scopes_input_model(
            turn_capabilities,
            max_scopes=max(
                1, int(getattr(self.config, "max_search_scopes_per_call", 1) or 1)
            ),
            wearer_audience_field=wearer_audience_field,
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

        def _search_catalog_impl(scopes, not_covered=None):
            """Execute one catalog search per product role; may return signals."""

            return search_catalog(search_context, scopes, not_covered=not_covered)


        @tool(
            args_schema=search_tool_arguments_model,
            return_direct=False,
            response_format="content_and_artifact",
        )
        def search_catalog_tool(scopes, not_covered=None):
            """Find products by description, advertised taxonomy, or constraints.

            Use for browse, search, and recommendation requests after product
            discovery or outfit styling is active. Select exact values from the
            current Catalog capabilities. Do not use for a product already
            established in this conversation, and do not repeat a completed hard-
            filter scope with different semantic wording.
            """

            return normalize_tool_result(
                _search_catalog_impl(scopes, not_covered)
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
            cached_product = scope.product_evidence.get(product_ref)
            if cached_product is None:
                return (
                    f"No product with PRODUCT_REF '{product_ref}' is available. "
                    "Search this turn or resolve the earlier product first."
                )
            if _detail_fields_already_held(cached_product, turn_capabilities):
                # The search that produced this product already returned every
                # detail field its category advertises. Answering from that
                # evidence is not a guess; it is the same data, without an ~8.7s
                # round trip. A product recovered from the historical index
                # carries identity only, so it fails this check and still reads.
                record = _product_detail_record(cached_product)
                evidence = ProductDetailEvidence(products=[record])
                return (
                    _format_product_detail_record(record),
                    evidence.as_artifact(),
                )
            scope.product_detail_reads += 1
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

        def _descriptor_field(descriptor: Any, name: str) -> Any:
            """Read one descriptor field whether it arrived typed or as a dict."""

            if isinstance(descriptor, dict):
                return descriptor.get(name)
            return getattr(descriptor, name, None)

        def _established_this_turn(references: Any) -> list[str]:
            """Answer from this turn's evidence, for refs and names alike.

            The shopper asked about something this turn already searched for and
            found. Nothing needs resolving: the answer is in hand, and going to
            history for it returns nothing because the durable index is written
            when the turn ends.
            """

            answers: list[str] = []
            for descriptor in references or []:
                product_ref = _descriptor_field(descriptor, "product_ref")
                display_name = _descriptor_field(descriptor, "display_name")
                product = None
                if product_ref:
                    product = scope.product_evidence.get(
                        str(product_ref).strip().strip(_REFERENCE_WRAPPERS)
                    )
                if product is None and display_name:
                    wanted = " ".join(str(display_name).casefold().split())
                    product = next(
                        (
                            item
                            for item in scope.product_evidence.values()
                            if " ".join(str(item.display_name).casefold().split())
                            == wanted
                        ),
                        None,
                    )
                if product is None:
                    return []
                answers.append(
                    f"REFERENCE {_descriptor_field(descriptor, 'reference_id')}: "
                    f"ALREADY ESTABLISHED THIS TURN. "
                    f"PRODUCT_REF: {product.product_id}. "
                    f"NAME: {product.display_name}. "
                    "Use it directly; it needs no resolution and no search."
                )
            return answers

        def _catalog_name_lookup(result: Any, references: Any) -> str:
            """Look the shopper's product name up in the catalog, and say so.

            Nothing in this conversation matched, which means the shopper named
            a product the assistant never showed -- and that is a search
            request. Telling the model to search was a sentence in a tool
            result, so it was advisory: measured across full conversations it
            was obeyed most of the time and, when it was not, the assistant
            offered products it had shown earlier and the shopper never got the
            one they asked for.

            So the runtime does it. A name lookup needs no taxonomy and no
            filters -- nothing that belongs to the model -- so it can be
            composed here without deciding anything on the model's behalf.

            What comes back is labelled for what it is: found by name, not
            shown before. Whether one of these IS the product the shopper named
            or merely resembles it is a judgement about language, which the
            model makes; supplying honest facts to judge from is our job.
            """

            unresolved = {
                item.reference_id
                for item in result.results
                if item.status == "not_found" and not item.blocking_field
            }
            names: list[str] = []
            for descriptor in references or []:
                reference_id = _descriptor_field(descriptor, "reference_id")
                display_name = _descriptor_field(descriptor, "display_name")
                if reference_id not in unresolved:
                    continue
                # The name is usually in `display_name`. When the model puts it
                # in `reference_id` instead -- "Southwest Bracelet" as the label
                # rather than the name -- the lookup used to collect nothing and
                # do nothing, and the shopper was told the product could not be
                # found while its name sat one field over. Both fields are the
                # model's own free text; either may carry it.
                text = str(display_name or reference_id or "").strip()
                if text and text not in names:
                    names.append(text)
            if not names:
                return ""

            sections: list[str] = []
            for name in names[:_MAX_NAME_LOOKUPS]:
                try:
                    execution = execute_catalog_search(
                        CatalogSearchPlan(
                            should_search=True,
                            semantic_queries=[name],
                            hard_filters={},
                            search_mode="text",
                            top_k=4,
                        ),
                        self.config.retriever_port,
                        timeout_seconds=getattr(
                            self.config, "catalog_search_timeout_seconds", None
                        ),
                    )
                except Exception:  # pragma: no cover - retrieval already degrades
                    continue
                found = execution.result
                if not found.ok or not found.products:
                    sections.append(
                        f'CATALOG NAME LOOKUP "{name}": the catalog returned '
                        "nothing for that name. Tell the shopper it is not "
                        "carried. Do not offer a different product as though it "
                        "were the one they named."
                    )
                    continue
                # Registered exactly as a search result is, so these are
                # addable this turn and resolvable in the next one.
                scope.product_evidence.add(found.products)
                _append_product_results(state, found.products)
                for product in found.products:
                    if product.image_url:
                        scope.retrieved[product.display_name] = product.image_url
                lines = [
                    f'CATALOG NAME LOOKUP "{name}": not shown earlier in this '
                    "conversation. The catalog was searched by that name; these "
                    "are the closest matches in rank order, none previously "
                    "shown.",
                ]
                for rank, product in enumerate(found.products, start=1):
                    price = (
                        f" - ${product.price.amount:.2f} {product.price.currency}"
                        if getattr(product, "price", None)
                        else ""
                    )
                    lines.append(
                        f"{rank}. {product.display_name}{price} "
                        f"[PRODUCT_REF {product.product_id}]"
                    )
                exact = [
                    product
                    for product in found.products
                    if _same_product_display_name(name, product.display_name)
                ]
                if len(exact) == 1:
                    # Naming a product by the name the catalog gives it is not
                    # a resemblance to be judged -- it is the same product, and
                    # choosing it is the shopper's to do. Told to "offer it and
                    # ask which size", the assistant answered "add the
                    # Southwest Bracelet" with "I found a Southwest Bracelet
                    # for $169.99. Would you like me to add that?" -- asking
                    # permission for the thing it had just been asked to do,
                    # about a bracelet that has no size to ask about.
                    match = exact[0]
                    sizes = _advertised_sizes(match)
                    lines.append(
                        f"'{match.display_name}' is the product they named, by "
                        "the catalog's own name for it. They have chosen it. "
                        "Say plainly that it was not among the ones you had "
                        "shown, then "
                        + (
                            # Only a catalog that says "onesize" settles it.
                            # Silence about sizes is not evidence of having
                            # none, and a garment added in a size nobody chose
                            # is the failure this must not reintroduce.
                            "add it."
                            if sizes == [_ONE_SIZE]
                            else "ask which size"
                            + (
                                ", offering " + ", ".join(sizes)
                                if sizes
                                else ""
                            )
                            + " -- unless they already said one, in which case "
                            "add it."
                        )
                        + " Do not ask whether to add what they asked you to add."
                    )
                else:
                    lines.append(
                        "Say plainly that this was not something you had shown. "
                        "If one of these is the product the shopper named, offer "
                        "it and ask which size before adding. If none is, say "
                        "you do not carry that one and name the closest you do "
                        "-- never present a different product as the one they "
                        "asked for."
                    )
                sections.append("\n".join(lines))
            return "\n\n".join(section for section in sections if section)

        def _resolve_conversation_products_impl(
            references: list[ProductReferenceDescriptor],
        ):
            """Resolve products the shopper refers to from earlier in this
            conversation. Use only when a needed product was not established
            in the current turn. Submit exact descriptors from the historical
            product index. If a reference is ambiguous, ask one concise
            clarification and do not guess. If nothing matches at all, the
            result says what to do next.
            """

            # This turn's own evidence first, before the memory service is
            # called at all. A product this turn searched for is not in the
            # durable index yet -- that is written when the turn finalizes --
            # so asking history about it returns nothing, and the turn spends a
            # round trip rediscovering what it already holds. The two records
            # disagree only inside the turn that created one of them; reading
            # the nearer one first is what makes them agree.
            established = _established_this_turn(references)
            if established:
                return "\n".join(established)

            with scope.resolution_lock:
                # A resolution that found something ends the budget, as before.
                # One that found nothing no longer does: the failure itself
                # says "correct that field and retry", and the retry used to be
                # refused with an instruction to stop and ask -- so the turn was
                # spent asking the shopper to name a product the assistant had
                # named a turn earlier. Attempts are still counted, so a call
                # that keeps missing terminates.
                if (
                    scope.product_resolution_used
                    or scope.product_resolution_attempts
                    >= _MAX_PRODUCT_RESOLUTION_ATTEMPTS
                ):
                    return control(
                        "STOP_TOOL_USE: Historical product resolution limit "
                        "reached for this turn. Use the resolution results you "
                        "have and ask one concise clarification if needed.",
                        ControlSignal.STOP_TOOL_USE,
                    )
                scope.product_resolution_attempts += 1

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
            scope.product_evidence.add_resolutions(result.results, references)
            state.system_identified_products = list(
                scope.product_evidence.system_identified()
            )
            for resolution in result.results:
                if resolution.status != "resolved":
                    continue
                product = resolution.matches[0].product
                if product.image_url:
                    scope.retrieved[product.display_name] = product.image_url
            if any(item.status == "resolved" for item in result.results):
                scope.product_resolution_used = True
                return format_product_resolution(result)
            # Nothing resolved. The products this conversation has shown are
            # already recorded, so the next attempt can be a lookup in that
            # record rather than another guess at a descriptor.
            #
            # Only for a near miss. A blocking field means the call pointed at
            # something it had seen and got one field wrong, and the record is
            # what corrects it. When nothing matches at all, the shopper named a
            # product that was never shown -- that is a search request, and
            # handing back a list of earlier products reads as a menu and
            # suppresses the search: asked for a dress by name, the assistant
            # offered four it had shown before and never looked in the catalog.
            near_miss = any(item.blocking_field for item in result.results)
            if not near_miss:
                looked_up = _catalog_name_lookup(result, references)
                if looked_up:
                    return "\n\n".join(
                        (format_product_resolution(result), looked_up)
                    )
            return "\n\n".join(
                value
                for value in (
                    format_product_resolution(result),
                    format_historical_product_index(state.historical_product_sets)
                    if near_miss
                    else "",
                )
                if value
            )

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
            product index. If a reference is ambiguous, ask one concise
            clarification and do not guess. If nothing matches at all, the
            result says what to do next.
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
                choices_from_a_description: list[str] = []
            except ValueError as exc:
                return f"Cart add failed: {exc}"
            if not requested_items:
                return "Cart add failed: provide at least one PRODUCT_REF to add."

            def _resolve_from_conversation_index(product_ref: str):
                """Look one ref up in the conversation's durable product index."""

                descriptors = [
                    ProductReferenceDescriptor(
                        reference_id="cart-add",
                        product_ref=product_ref,
                    )
                ]
                try:
                    result = self._conversation_products.resolve(
                        identity.conversation_id,
                        descriptors,
                    )
                except (ConversationProductsError, ValidationError):
                    return None
                scope.product_evidence.add_resolutions(result.results, descriptors)
                state.system_identified_products = list(
                    scope.product_evidence.system_identified()
                )
                return scope.product_evidence.get(product_ref)

            resolved: list[tuple[str, ProductSummary, int]] = []
            failed: list[str] = []
            blocked: list[str] = []
            for (product_ref, size), request in requested_items.items():
                product = scope.product_evidence.get(product_ref)
                if product is None:
                    # The conversation's product index is the identity lane: it
                    # is durable, scoped to this conversation, and printed into
                    # the prompt every turn -- which is where the model read
                    # this ref. Evidence is rebuilt per turn, so a ref shown two
                    # turns ago is absent from it, and refusing on that basis
                    # rejected a ref the shopper had genuinely been shown.
                    # This is a lookup in the conversation's own record, not a
                    # catalog search.
                    product = _resolve_from_conversation_index(product_ref)
                if product is None:
                    # Two different situations reach here, and naming only one
                    # of them stranded the other.
                    #
                    # A ref the shopper was never shown: the model passes the
                    # product's name, and told only that a name is not a ref it
                    # asked the shopper for the exact catalogue name -- the
                    # assistant's own job, with a search budget unspent.
                    #
                    # A ref shown in an *earlier* turn: evidence is rebuilt per
                    # turn, so a real ref from last turn is absent from this
                    # one. "Add it in a 10 as well" carried the correct ref for
                    # a dress added moments earlier, was told to go searching,
                    # and gave up -- so a shopper asking for a second size got
                    # "the add didn't go through".
                    failed.append(
                        f"- PRODUCT_REF '{product_ref}': not established in "
                        "this turn. If this product was shown earlier in the "
                        "conversation, resolve it first and add the PRODUCT_REF "
                        "that comes back -- evidence is per turn, so a "
                        "reference from an earlier turn has to be resolved "
                        "again. If it is a product name rather than a "
                        "reference, or was never shown at all, search the "
                        "catalog now and show the closest matches, then ask "
                        "which to add. Never add a product the shopper has not "
                        "been shown, and do not ask them for a catalogue name, "
                        "a link, or a price."
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
                # Whether the catalog sells this size is a fact and is still
                # checked. Whether the shopper chose it is a reading, and the
                # model reads the conversation better than any matcher here
                # could: it resolved the right heel and picked size 7 from "add
                # the Jade Suede Heels in a 7", then was refused for not also
                # quoting the shopper back into a field. The size and quantity
                # now travel into the result instead, where a wrong one is
                # visible on the turn it happens.
                size_issue = _cart_size_issue(active_detail.product, size)
                if size_issue:
                    blocked.append(f"- PRODUCT_REF '{product_ref}': {size_issue}")
                    continue
                # Disclosed, not refused. A description the model read one way
                # is added and said out loud, because the cart is on screen and
                # a wrong line is one click away -- where a refusal costs a
                # turn on every request it misjudges, and it misjudged plenty.
                choice_note = _cart_product_choice_note(
                    active_detail.product,
                    _shopper_words_this_conversation(state),
                    scope.product_evidence,
                    _most_recently_shown(state),
                    _identified_in_the_current_showing(state),
                    size,
                )
                if choice_note:
                    choices_from_a_description.append(
                        f"- {active_detail.product.display_name}: {choice_note}"
                    )
                resolved.append(
                    (
                        product_ref,
                        active_detail.product,
                        int(request["quantity"]),
                        size,
                    )
                )

            scope_failures = _cart_add_scope_failures(
                state.query,
                [(product_ref, product) for product_ref, product, _, _ in resolved],
                scope.product_evidence.values(),
            )
            blocked.extend(message for _ref, message in scope_failures)
            out_of_scope = {ref for ref, _message in scope_failures}
            if blocked:
                state.cart = self._read_cart(identity.cart_user_id)
                self._append_product_images(
                    scope.retrieved,
                    state.cart,
                    scope.product_evidence.values(),
                )
                # Nothing is written -- the add is all or nothing. But the items
                # that were established travel with the refusal, so the question
                # put to the shopper is only the one still open.
                # An item can pass every per-item gate and still be refused
                # below as outside this turn's request. Listing it as settled
                # would tell the shopper not to ask again about the very thing
                # that failed.
                ready = [
                    f"- {product.display_name}"
                    + (f", size {size}" if size else "")
                    + f", qty {quantity}"
                    for ref, product, quantity, size in resolved
                    if ref not in out_of_scope
                ]
                return _format_cart_add_result(
                    [], failed + blocked, state.cart, ready
                )

            added: list[str] = []
            committed: list[dict[str, Any]] = []
            for product_ref, product, quantity, size in resolved:
                result = add_cart_item(
                    AddCartItemInput(
                        user_id=str(identity.cart_user_id),
                        product_id=product.product_id,
                        display_name=product.display_name,
                        quantity=quantity,
                        size=size,
                        unit_price=product.price,
                        image_url=product.image_url,
                        # Size is part of the key: adding a 6 and an 8 in one
                        # turn are two mutations, not a retry of one.
                        idempotency_key=(
                            f"{identity.request_id}:add:{product.product_id}"
                            f":{size or 'onesize'}:{quantity}"
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
                    # The size travels with the line it went in as. Nothing
                    # now refuses a size the shopper did not choose, so the
                    # whole safety story is that it is visible -- to the model
                    # writing the reply, and through it to the shopper, on the
                    # turn it happens rather than at checkout.
                    added.append(
                        f"- {quantity} x {product.display_name}"
                        + (f", size {size}" if size else "")
                        + f" (PRODUCT_REF: {product.product_id})"
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
            if choices_from_a_description:
                rendered += "\n\n" + "\n".join(choices_from_a_description)
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

        # Not a tool. `remove_cart_item_tool` calls this directly, and a
        # decorated function is a StructuredTool, which is not callable -- so
        # every removal raised `'StructuredTool' object is not callable` and
        # the turn died. Its sibling `_add_cart_items_impl` is undecorated for
        # the same reason.
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

        def _update_cart_items_impl(
            cart_line_id: str,
            quantity: int,
            size: str | None = None,
        ):
            """Change the quantity of an item already in the cart. Use ONLY when
            the shopper explicitly asks to change a quantity. Do NOT use for
            initial adds — use add_cart_items_tool. Do NOT guess the
            CART_LINE_ID; call get_cart_tool first if you do not have one.
            """

            if size is not None:
                # `size` is accepted only so it can be refused. The parameter
                # does not exist on the cart, and there is no route behind it:
                # asked to "change the heels to an 8" the model sent
                # quantity 1 with size 8, the argument went nowhere, the
                # quantity was set 1 -> 1, and the assistant told the shopper
                # the size had changed while the cart kept the old one. A
                # silently dropped argument is worse than a rejected one.
                return (
                    "CART_UPDATE_REFUSED: this tool changes quantities only, and "
                    "a size is a different line rather than a property of one. "
                    "To change a size: add the new size with add_cart_items_tool "
                    "FIRST, confirm it is in the cart, then remove the old line "
                    "with remove_cart_item_tool. Never the other way round -- a "
                    "failure between the two must leave the shopper with an "
                    "extra line, never with nothing."
                )

            if quantity == 0:
                # A size is a different line, not a different quantity, and the
                # cart has no operation for changing one. Asked for a size 8
                # against a line added as a size 2, the model reached for the
                # only move available -- quantity 0 -- which deleted the line,
                # and then never added the replacement. The shopper corrected
                # their size and lost the item.
                return (
                    "CART_UPDATE_REFUSED: quantity 0 would delete this line, and "
                    "this tool changes quantities. If the shopper is changing a "
                    "SIZE, a size is a different line: add the new size with "
                    "add_cart_items_tool FIRST, confirm it is in the cart, then "
                    "remove the old line with remove_cart_item_tool. Never the "
                    "other way round -- a failure between the two must leave the "
                    "shopper with an extra line, never with nothing. If they "
                    "simply want the line gone, use remove_cart_item_tool."
                )

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
        def update_cart_items_tool(
            cart_line_id: str,
            quantity: int,
            size: str | None = None,
        ):
            """Set the exact quantity for one cart line. Use for quantity
            changes instead of removing and re-adding. Requires CART_LINE_ID
            from get_cart_tool.
            """

            return normalize_tool_result(
                _update_cart_items_impl(cart_line_id, quantity, size)
            )

        @tool(args_schema=_DescribeCatalogInput, return_direct=False)
        def describe_catalog_tool() -> str:
            """What this shop holds: how many products, which categories, the
            price range of each, and their subcategories. Use for questions
            about the SHOP rather than about a product -- the most or least
            expensive thing, whether anything falls in a price range, what
            departments exist. Takes no arguments and searches nothing.

            A fact about the catalog comes from here, never from the results of
            one search: the dearest item a search happened to return is that
            search's maximum, not the shop's. To name the actual item, read the
            range here and then search that category at that bound.
            """

            return format_catalog_shape(self._catalog_capabilities.get())

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

        @tool(args_schema=_WeatherForecastInput, return_direct=False)
        def get_weather_forecast_tool(
            city: str,
            date: CalendarDate | None = None,
            start_date: CalendarDate | None = None,
            end_date: CalendarDate | None = None,
        ) -> str:
            """Live daily forecast for one place, for the dates being dressed
            for.

            Call it, without being asked, when all three hold: the shopper
            named a CITY, town or postal code; they named a date or window;
            and that window is within about 15 days of TODAY. A destination
            wedding, a trip, an outdoor event. Conditions change what to wear
            more than anything else about a destination.

            The `city` argument takes a city, town or postal code. A country
            or region has no single weather, so prefer asking which city over
            calling with one. If you do call with something broad, the reply
            must say the numbers cover that whole area and ask which city --
            never present them as the weather where the shopper will be.

            Do not call it otherwise. Specifically:
            - No date. Today is not what they are dressing for; ask instead.
            - A date further out than about 15 days. There is no forecast that
              far ahead, so a call cannot produce anything true.
            - No place, or nothing they are dressing for.

            A country or region does not stop you. "We're going to Italy at the
            weekend" was answered with no forecast at all and a flat assertion
            that the weather would be warm -- worse than either asking or
            calling. Call it for the place they named, using its capital or
            largest city when they named a country, then say which city the
            numbers are for and ask whether that is where they will be. What
            you may never do is describe weather you did not fetch.

            Dress the date they are dressing for, not the one they travel on:
            "flying to Rome tomorrow, what do I wear at the weekend" is a
            forecast for the weekend. Resolve relative dates against TODAY
            first -- one exact ISO date, or a complete inclusive ISO start/end
            range -- and never send a relative date or invent a place.
            """

            if weather_call_needs_a_date(date, start_date, end_date):
                # The library treats a missing date as local today, which is
                # right for "what is it like there now" and wrong for the only
                # thing a shopper asks: "a wedding in Cancun" would silently
                # get today's weather for an event months away. Ask instead.
                return WEATHER_NO_DATE
            if not claim_weather_call(scope):
                return WEATHER_BUDGET_EXHAUSTED
            return _format_weather_result(
                self._weather_client.get_forecast(
                    WeatherRequest(
                        location=city,
                        date=date,
                        start_date=start_date,
                        end_date=end_date,
                    )
                )
            )

        @tool(args_schema=_CheckAvailabilityInput, return_direct=False)
        def check_product_availability_tool(items) -> str:
            """Check whether products are available or in stock. Use ONLY when
            the shopper explicitly asks about availability, stock, or a specific
            size. Requires a PRODUCT_REF established by search or
            historical-product resolution. Do NOT use for browsing. Pass every
            product being asked about in one call. The deterministic stub
            reports general availability, sized availability for apparel and
            footwear, and one-size availability for other product categories.
            """

            requests = [
                item if isinstance(item, dict) else item.model_dump()
                for item in items
            ]

            def _one(entry: dict[str, Any]) -> str:
                product_ref = entry.get("product_ref") or ""
                product = scope.product_evidence.get(product_ref)
                if product is None:
                    return (
                        f"PRODUCT_REF '{product_ref}' is unknown in this "
                        "conversation. Search this turn or resolve the earlier "
                        "product first."
                    )
                return _format_availability_result(
                    check_product_availability(
                        CheckProductAvailabilityInput(
                            product_ref=product_ref,
                            variant_hint=entry.get("variant_hint"),
                        ),
                        product,
                    )
                )

            # Each check stands in for an inventory-system lookup, so they go out
            # together. Asking about four products cost four model round trips at
            # roughly 8.7s each -- enough to exhaust a turn's step budget before
            # the shopper got an answer.
            if len(requests) == 1:
                return _one(requests[0])
            with ThreadPoolExecutor(max_workers=len(requests)) as pool:
                return "\n\n".join(pool.map(_one, requests))

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
            describe_catalog_tool,
            check_active_promotions_tool,
        ]
        # Off means absent, not present-and-failing. An unregistered tool
        # cannot be called, so a shop without weather simply styles the
        # occasion instead of explaining a capability nobody asked about.
        weather_off = not getattr(
            getattr(self.config, "weather", None), "enabled", False
        )
        if not weather_off:
            shopping_tools.append(get_weather_forecast_tool)
        validate_registered_tool_names(
            {
                str(
                    getattr(candidate, "name", None)
                    or getattr(candidate, "__name__", "")
                )
                for candidate in shopping_tools
            },
            disabled=("get_weather_forecast_tool",) if weather_off else (),
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
            never a pair. Adding a product the shopper names that no search in
            this conversation has shown is a discovery request as well as a cart
            one: select product-discovery with cart-management, because the cart
            skill cannot search and the product must be found and shown before
            it can be added. Add budget-shopping only as a modifier when the
            shopper states a budget. Keep outfit-styling as the primary skill throughout
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
                media=bool(state.media),
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

    def _model_request_timeout(self) -> float:
        """How long one request may take, derived from this turn's budget.

        A fixed ceiling was wrong: forty seconds fits a 150-second deployment
        and swallows the whole loop of a 45-second one. Two attempts have to
        fit inside whatever the loop is actually given, or the retry the
        deadline exists to enable is itself cut off by the turn.
        """

        budget = max(0.0, float(self.config.deepagents_execution_timeout_seconds))
        reserve = min(
            max(0.0, float(getattr(self.config, "grounding_editor_reserve_seconds", 0.0))),
            budget / 2,
        )
        allowance = max(budget - reserve, budget / 2)
        return max(5.0, min(_MODEL_REQUEST_TIMEOUT_CEILING_SECONDS, allowance / 2))

    def _create_chat_model(self):
        from langchain_openai import ChatOpenAI

        api_key_env = getattr(self.config, "llm_api_key_env", None)
        api_key = os.environ.get(api_key_env, "") if api_key_env else "not-needed"
        return ChatOpenAI(
            model=self.config.llm_name,
            base_url=self.config.llm_port,
            api_key=api_key or "not-needed",
            temperature=0,
            # One request hung for 133.8 seconds and took the whole turn with
            # it: the shopper asked to add a bracelet and got "this request
            # took too long to complete" after two and a quarter minutes, on a
            # turn that normally costs ten seconds. Without a deadline here the
            # only limit was the turn budget, so a single stalled call spent
            # everything the turn had.
            #
            # This also switches on the retry that was already configured.
            # `max_retries` defaults to 2 and fires on errors -- and a hang is
            # not an error, so nothing ever retried. A call that has not
            # answered in forty seconds is not going to; failing it leaves time
            # to ask again and still finish inside the turn.
            timeout=self._model_request_timeout(),
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
            "WHAT THIS TURN DID TO THE CART (authoritative, computed):\n"
            f"{format_cart_change(state.cart_at_turn_start, state.cart)}\n\n"
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
        rewrite_usage = _collect_token_usage(rewrite_result)
        state.token_usage = _merge_token_usage(state.token_usage, rewrite_usage)
        rewrite_tokens = int(rewrite_usage.get("total_tokens") or 0)

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
                tokens=rewrite_tokens,
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
            tokens=rewrite_tokens,
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
        media: bool = False,
    ) -> str:
        catalog_context = format_catalog_capabilities_for_prompt(capabilities)
        # Media rules are only reachable on a turn that carries media, so they
        # are only assembled then.
        media_rules = _MEDIA_TURN_RULES if media else ""
        shopper_context_rules = (
            f"\n{_SHOPPER_CONTEXT_SYSTEM_RULES}\n"
            if shopper_context is not None
            else ""
        )
        prompt = f"""You are a retail shopping assistant for the products advertised by the active catalog.

TODAY IS {_today_for_the_shopper()}.
That is the only date you know. Every "this weekend", "next week", "in three
days" is counted from it, and a forecast may only be asked for a window within
about fifteen days of it. Without this the weather tool could not be called for
anything a shopper phrased in their own words: "we're going to Italy first,
what do I wear at the weekend" fetched no forecast and the reply asserted warm
weather anyway.

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
Every rule above bans a claim about a product. None of them bans thinking about
the occasion, and a shopper dressing for one is owed that thinking. Reason from
a confirmed attribute to a judgement about the situation: a stiletto heel sinks
into grass and sand, a floor-length hem drags on a lawn, a suede upper is wrong
for a beach, an open toe is cold in November. Say it as a judgement about the
occasion, never as a property of the item. "A stiletto will sink into grass" is
judgement and is welcome; "these are stable on grass" is a claim and stays
forbidden.
Ask when something material is missing, and ask it as a shopper would. The
occasion is usually what is missing: brunch, a wedding and errands need
different answers whatever else is true. The place is worth asking for once the
answer depends on it -- an outdoor event, a trip, or a question about the
weather -- but never ask where someone lives as a matter of course. Give the
styling reason first: "is it on the beach or indoors? sand and a lawn need
different shoes". Show a grounded starting point in the same reply; never
answer with only questions.
If the shopper asks you directly about the weather and you have no forecast,
say so and answer from what you know, as typical rather than predicted:
"I don't have a live forecast for Cancun, but August there is usually hot and
humid." Do not volunteer this when they did not ask, never give a temperature
or a condition for a specific date without a forecast in TOOL EVIDENCE, and
never conclude anything about the weather where the shopper is.
When the occasion needs something this catalog does not stock, say so as advice
and keep showing what was found: "you'll want a proper coat over this; we don't
carry outerwear." Never offer the nearest item as though it served the purpose,
and never invent a need the shopper's own words or the conditions they stated do
not support.
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
- Not every turn is a request for products. A shopper who names a place, a
  date, a companion, a mood or a size is adding context to what they already
  asked for, and answering a question you just asked is not a new request at
  all. Fold it into what is already on screen and reply from that: refine the
  recommendations you have made, or act on the item under discussion. Do not
  invent a product type from it. "Florence", one turn after "we're going to
  Italy first", became `requested_product_type: "Florence"` and a search that
  returned floral maxi dresses; "size 2 please", answering your own question
  about which size, became a fresh dress search instead of the add the shopper
  was waiting for. Search only when the shopper is asking to see something
  they have not been shown.
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
- Use at most {self.config.max_catalog_searches_per_turn} product roles across
  all catalog searches in one user turn, and carry them in as few calls as
  possible: roles in one call retrieve together and cost one round trip.
  One normalized taxonomy-and-required-constraint scope can execute
  only once in a turn; do not retry the same hard-filter scope with different
  semantic wording. For outfit requests
  with multiple required item types, send one focused role per distinct
  taxonomy scope in the same call, then stop and synthesize from those results.
  If the shopper named a place and a date you could forecast, look the weather
  up BEFORE that fan-out, not after: once the roles are out you are told to
  stop and synthesize, and the forecast never gets asked for. Conditions change
  which pieces you would even search for, so they belong first. Measured: the
  same sentence about a trip fetched a forecast on its own and skipped it
  entirely once it arrived mid-conversation and read as an outfit request.
- Advice is not an answer on its own either. A layering formula, a packing list
  or a list of what to look for, with no pieces from this shop beside it, is a
  wardrobe lecture rather than shopping. Search and show real items in every
  styling reply, including when conditions are hard, the date is unknown, or
  the shop cannot cover the whole need -- show what it does have and say what
  is missing. Measured: "it's going to snow this weekend" and "a wedding in
  Cancun, date not fixed yet" both returned formulas and nothing to buy.
- Treat broad weather or occasion context as styling direction, not automatically
  as a product-attribute guarantee. A "rainy day outfit" or "wet-weather outfit"
  should search practical
  advertised roles without adding an unadvertised requirement; add water
  resistance to `unadvertised_requirements` only when the shopper directly
  requires it for a product. "Rainy day outfit" does not imply water resistance;
  "water-resistant bags" directly requires it.
- For alternatives joined by "or", search the faithful advertised branch and
  preserve every named branch. If another branch cannot be mapped faithfully,
  present the supported result and ask one concise clarification before any
  adjacent search. Do not reject a supported branch merely because another
  alternative is unresolved, and never list the supported taxonomy value in
  `unadvertised_requirements`.
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
- Shopper wording is not product evidence. If the shopper mentions an
  unverified attribute such as heel shape, material, colorway, fit, or care,
  verify it with tools or refer to it as the shopper's preference, not as a
  catalog fact.
- When the shopper asks to add an item that has not already been searched in
  this conversation, call search_catalog_tool first, then call
  add_cart_items_tool with a one-item list containing the selected PRODUCT_REF.
- Cart mutations require explicit shopper intent and must use
  add_cart_items_tool, remove_cart_item_tool, or update_cart_items_tool. Never
  claim a cart mutation unless the tool reports success.
- Use PRODUCT_REF established by current-turn search or historical-product
  resolution when requesting product details. Do not pass display names to
  get_product_details_tool.
- Previous preference context is guidance only. The current shopper request
  wins when it conflicts with previous preferences.
- Keep final answers concise and grounded in tool results. Attribute materials,
  comfort, construction, and outdoor-practicality claims to the specific items
  that support them instead of making unsupported whole-outfit claims. Avoid
  guarantee language such as "will stay comfortable all evening" unless the
  catalog evidence supports the guarantee. Before finalizing, remove or soften
  unsupported phrases about grass, gravel, water resistance, all-day comfort,
  maximum breathability, or best-in-category performance.
{CATALOG_SEARCH_RULES}
{media_rules}
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
        sections.append(_format_store_date())
        shopper_context = _format_shopper_context(state.shopper_context)
        wearer = _format_wearer_audience(state.wearer_audience)
        if shopper_context:
            sections.append(shopper_context)
        if wearer:
            sections.append(wearer)
        sections.extend(
            [
                (
                    f"USER QUERY: {state.query}\n"
                    f"IMAGE ATTACHED: {'yes' if state.image else 'no'}"
                ),
                f"MEDIA ATTACHED:\n{_format_media_summary(state.media)}",
                f"MEDIA ANALYSIS:\n{state.media_analysis or '(none)'}",
                f"CURRENT CART:\n{_format_cart(state.cart)}",
                format_most_recent_subject(state),
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
                    **({"size": line.size} if line.size else {}),
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
        state.wearer_audience = list(turn.wearer_audience)
        state.assumed_audience = list(turn.assumed_audience)
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

        # Ordered here, before the record is written and before the events are
        # emitted, so the shopper, the durable index and the resolver all count
        # the same list. Ordering only at the stream would have left "the second
        # one" meaning the second shown to the shopper and the second ranked to
        # the resolver.
        state.product_results = _in_presentation_order(
            state.product_results or [], state.response or ""
        )
        state.retrieved = _images_in_product_order(
            state.retrieved or {}, state.product_results
        )

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
            receipt = self._conversation_memory.finalize_turn(
                identity.conversation_id,
                turn.turn_id,
                request_id=identity.request_id,
                attempt_id=turn.attempt_id,
                assistant_text=state.response,
                status=final_status,
                termination_reason=reason,
                events=[
                    *_turn_audience_events(
                        state,
                        identity,
                        field_name=getattr(
                            self.config, "wearer_audience_field", ""
                        ),
                    ),
                    *_system_identification_events(state, identity),
                ],
                output=output,
            )
            finalized = True
            if receipt.dropped_event_types:
                # The turn is safe; only the enrichment was lost. Surfaced
                # rather than swallowed, because the same signal means a typo
                # here and an older memory service in a rolling deploy.
                logger.warning(
                    "Conversation memory dropped unknown event types: %s",
                    ", ".join(receipt.dropped_event_types),
                )
                state.agent_diagnostics["memory_dropped_event_types"] = list(
                    receipt.dropped_event_types
                )
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






























































