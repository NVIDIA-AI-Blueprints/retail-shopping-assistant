# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deep Agents SDK runtime for the shopping assistant."""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import inspect
import json
import logging
import os
import sys
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import requests
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from pydantic import (
    ValidationError,
)
from shared.commerce_contracts import (
    CatalogCapabilities,
    GetCartInput,
    ProductSummary,
)

from ..agenttypes import Cart, ShopperContext, State
from ..cart_format import _format_cart, format_cart_change
from ..catalog_capabilities import (
    CatalogCapabilitiesClient,
    format_catalog_capabilities_for_prompt,
)
from ..commerce_tools import (
    get_cart,
)
from ..conversation_memory import (
    ConversationMemoryClient,
    ConversationMemoryError,
    FinalTurnStatus,
    TurnReplayOutput,
    TurnStartResult,
    build_dialogue_context,
)
from ..conversation_products import (
    ConversationProductsClient,
    format_historical_product_index,
)
from ..fencing import MEDIA_FENCE
from ..media_perception import MediaPerceptionClient
from ..media_summary import summarize_media_analysis
from ..search_replies import (
    _format_search_only_response,
    _partial_product_results_response,
    _scrub_internal_shopper_language,
    _search_guidance_evidence,
)
from ..tools.cart import build_cart_tools
from ..tools.catalog import build_catalog_tools
from ..tools.loop_control import (
    ToolLoopControlMiddleware,
)
from ..tools.policy import (
    load_shopper_skill_registry as _shopper_skill_registry,
)
from ..tools.policy import (
    validate_registered_tool_names,
)
from ..tools.skill_gate import (
    ShopperSkillActivationError,
    selected_skill_names_for_turn,
)
from ..tools.skills import build_skill_activation
from ..tools.store import build_store_tools
from ..tools.weather import build_weather_tool
from ..vocabulary_judge import CatalogVocabularyJudge
from ..weather import WeatherConfig, build_weather_client
from .audience_events import _system_identification_events, _turn_audience_events
from .control_signals import (
    committed_effects_in,
)
from .grounding_evidence import (
    _collect_tool_grounding_evidence,
)
from .identity import RequestIdentity
from .message_shape import (
    _content_to_text,
    _extract_final_text,
    _result_messages,
    _value,
)
from .model_usage import (
    _add_model_usage,
    _collect_token_usage,
    _merge_token_usage,
    _normalized_token_usage,
    _record_language_model_failure,
    _record_media_model_usage,
    _record_safety_model_usage,
    _should_short_circuit_media_failure,
)
from .prompts import (
    _DEEP_AGENT_BASE_PROMPT,
    _GROUNDING_EDITOR_SYSTEM_PROMPT,
    _MEDIA_TURN_RULES,
    _SHOPPER_CONTEXT_SYSTEM_RULES,
    _format_media_summary,
    _format_retrieved_images,
    _format_shopper_context,
    _format_store_date,
    _format_wearer_audience,
    _today_for_the_shopper,
    format_most_recent_subject,
)
from .replies import (
    _CONVERSATION_PROFILE_MISMATCH_RESPONSE,
    _GROUNDING_FAILURE_RESPONSE,
    _SHOPPER_PROFILE_NOT_FOUND_RESPONSE,
    _committed_effect_receipt,
    _has_grounding_authority,
    _has_search_only_tool_evidence,
    _images_in_product_order,
    _in_presentation_order,
    _media_failure_response,
    _products_found_receipt,
)
from .turn_diagnostics import (
    _catalog_repair_clarification_response,
    _empty_agent_diagnostics,
    _has_successful_non_search_tool_evidence,
    _rejected_catalog_search_response,
    _safe_collect_agent_diagnostics,
)
from .turn_scope import TurnScope

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
def the_showing(
    products: Sequence[Any],
    groups: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """This turn's products as the shopper sees them: a list of headed groups.

    A showing is not a queue. Asked for dresses and shoes, the shopper reads
    two headed lists and counts from one inside each, so "the second shoes" is
    a question the structure can answer. Stored flat, it was eight products in
    a row and that question had no answer at all.

    Numbering restarts inside each group, because that is how the shopper
    counts. Everything downstream -- the screen, the reply, resolution -- is
    handed this same structure, so none of them has to derive an order and
    none of them can derive a different one.

    Products no group claims are one unheaded group at the end, which is also
    how a conversation recorded before groups existed reads back.
    """

    held = {
        str(product.get("product_id") or ""): product
        for product in products
        if isinstance(product, Mapping) and str(product.get("product_id") or "")
    }
    shown: list[dict[str, Any]] = []
    placed: set[str] = set()
    for group in groups or []:
        members = []
        for product_id in group.get("product_ids") or []:
            product = held.get(str(product_id))
            if product is not None and str(product_id) not in placed:
                members.append(product)
                placed.add(str(product_id))
        if members:
            shown.append(
                {
                    "heading": _a_heading_that_is_not_a_product(
                        str(group.get("heading") or ""), products
                    ),
                    "products": _numbered_within_the_group(members),
                }
            )
    unclaimed = [
        product
        for product in products
        if isinstance(product, Mapping)
        and str(product.get("product_id") or "") not in placed
    ]
    if unclaimed:
        shown.append(
            {"heading": "", "products": _numbered_within_the_group(unclaimed)}
        )
    return shown


def _streamed_in_group_order(
    showing: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """The showing flattened for the wire, every product naming its group."""

    return [
        {**product, "group": group.get("heading") or ""}
        for group in showing
        for product in group.get("products") or []
    ]


def _a_heading_that_is_not_a_product(heading: str, products: Sequence[Any]) -> str:
    """This heading, unless it is the name of a product being shown under it.

    Asked to add one tote by name, the model sends that name as the product
    type, and the group of four totes comes out headed "Ombre Canvas Tote Bag"
    -- three of which are not that. An equality check against the showing's own
    products, not a word list: the data to settle it is already in hand.
    """

    wanted = _normalized_display_name(heading)
    if not wanted:
        return ""
    for product in products:
        if not isinstance(product, Mapping):
            continue
        if _normalized_display_name(str(product.get("display_name") or "")) == wanted:
            return ""
    return heading


def _normalized_display_name(name: str) -> str:
    return " ".join(name.split()).casefold()


def _numbered_within_the_group(
    products: Sequence[Any],
) -> list[dict[str, Any]]:
    """The group's products, each carrying the place it holds under its heading.

    So the client renders a given order instead of deriving one. It had been
    deriving one: the chat row was built from a name-keyed image map, the
    product list was matched back into it by display name, and the panel kept
    its own ordering state. Three mechanisms standing in for a number.

    Stamped here rather than on `state.product_results` because that list is
    re-parsed as `ProductSummary`, which forbids unknown fields. It belongs at
    this boundary regardless: where a product sits on a screen is a fact about
    how this turn was presented, not a fact about the product.
    """

    return [
        {**product, "position": position}
        for position, product in enumerate(products, 1)
        if isinstance(product, dict)
    ]


#: `read_file` joined this list once the base prompt below stopped being a lie.
#: It said "You have no filesystem" while the model could and did call
#: read_file -- observed on live turns across the run archive, including one
#: with ten consecutive calls, and one that also reached for `glob`. Every one
#: of those was a model round trip spent fetching a skill file the activation
#: middleware had already injected in full, which is why the injection prompt
#: had to ask it not to. Removing the tool answers that where a request could
#: only ask.
_EXCLUDED_DEEP_AGENT_TOOLS = frozenset(
    {
        "write_todos",
        "ls",
        "read_file",
        "write_file",
        "edit_file",
        "glob",
        "grep",
        "execute",
    }
)
#: Names looked up in the catalog when a reference resolves to nothing. Bounded
#: because each is a retrieval, and a turn that names more than two products the
#: assistant never showed is a conversation to have, not a batch to satisfy.
#: The longest one model request may take before it is abandoned, whatever the
#: deployment's turn budget. A request slower than this has stalled rather than
#: thought: the median turn costs ten seconds end to end.
_MODEL_REQUEST_TIMEOUT_CEILING_SECONDS = 40.0


"""Ceiling on one availability batch, tied to what one search hands over.

These are two halves of one rule and were two independent numbers. A search
returns up to `search_products_per_call` products; the field below asks for
every product in one call. When the ceiling was the smaller of the two, the
call that obeyed the instruction was the call that failed validation.

It cost a turn. A shopper dressing for a wedding got twenty-one products back,
the model batched all twenty-one exactly as asked, and pydantic refused it for
holding one more than twenty. Recovering, the model split the batch and then
re-sent one half eighteen times until the graph hit its recursion limit and
the turn died before composing -- so a search that had found four dresses and
confirmed every one of them in stock returned a fallback with no dresses in it.
"""


_RELAY_SUBSCRIBER = "chain-server"

_relay_warnings_said: set[str] = set()


def _relay_warn_once(key: str, message: str, *args: Any) -> None:
    """Say a Relay problem once, not once a turn.

    The agent is rebuilt every turn, so a warning raised on the build path is a
    warning per turn -- and the conditions here (a missing package, a release
    that changed the arguments) are settled at startup and never change.
    """

    if key in _relay_warnings_said:
        return
    _relay_warnings_said.add(key)
    logger.warning(message, *args)


def configure_relay_tracing(config: Any) -> bool:
    """Export NeMo Relay's lifecycle events, or export nothing and say why.

    Relay's middleware emits into an in-process runtime, and nothing leaves the
    service until a subscriber is registered -- so attaching the middleware
    alone is observable to nobody. This is the other half, and it lives beside
    ``_relay_instrumented`` because neither half is any use without the other.

    It speaks ``openinference`` rather than ``gen_ai`` because that is the
    dialect Phoenix reads, and it reuses ``OTEL_EXPORTER_OTLP_ENDPOINT`` rather
    than inventing a second endpoint setting: one collector address, whichever
    producer is speaking. Relay carries its own exporter, so this adds a second
    OTLP client to the process, not a second backend.

    Returns whether events are being exported, so a caller can say so.
    """

    if not getattr(config, "relay_enabled", False):
        return False

    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if not endpoint:
        logger.warning(
            "RELAY_ENABLED is set but OTEL_EXPORTER_OTLP_ENDPOINT is not; "
            "Relay events have nowhere to go and will not be exported."
        )
        return False

    try:
        from nemo_relay import OpenTelemetryConfig, OpenTelemetrySubscriber

        relay_config = OpenTelemetryConfig(
            "openinference", f"{endpoint.rstrip('/')}/v1/traces"
        )
        relay_config.service_name = os.environ.get("OTEL_SERVICE_NAME", "chain-server")
        subscriber = OpenTelemetrySubscriber(relay_config)
        # Registration is global and refuses a duplicate name, so a second
        # runtime would raise and read as "tracing broke" when the first
        # subscriber is still exporting perfectly well.
        subscriber.deregister(_RELAY_SUBSCRIBER)
        subscriber.register(_RELAY_SUBSCRIBER)
        # Spans are batched, so a container stopped without a flush loses the
        # tail of the last conversation -- the part worth reading.
        atexit.register(subscriber.force_flush)
    except ImportError:
        _relay_warn_once(
            "not-installed",
            "RELAY_ENABLED is set but nemo-relay is not installed; "
            "install requirements-relay.txt to trace.",
        )
        return False
    except Exception as exc:  # noqa: BLE001 - tracing must never break startup.
        logger.warning("Could not configure Relay tracing: %s", type(exc).__name__)
        return False

    logger.info("Relay tracing enabled, exporting to %s", endpoint)
    return True


@contextlib.contextmanager
def _relay_turn_scope(config: Any, conversation_id: str):
    """Open one Relay scope around the turn, so its events have somewhere to go.

    Relay does not read OpenTelemetry's context. Its events go into a runtime
    with its own scope stack, so a session set with ``using_attributes`` reaches
    the LangChain spans and never reaches Relay's -- and with no Relay scope
    open, every Relay span is emitted as its own root.

    Both of those are the same absence. One scope here gives Relay's spans a
    parent to nest under and a place to carry the conversation, and it is the
    supported way to do it: no LangGraph internals are touched.

    A no-op when Relay is off or absent, and a no-op when the scope will not
    open -- a trace must never cost a turn.
    """

    if not getattr(config, "relay_enabled", False):
        yield
        return

    try:
        from nemo_relay import ScopeType
        from nemo_relay import scope as relay_scope

        # Not "turn": our own OpenTelemetry span already owns that name, and two
        # different producers emitting a span called "turn" is unreadable.
        opened = relay_scope.scope(
            "relay-turn",
            ScopeType.Agent,
            metadata={"session.id": conversation_id},
        )
        opened.__enter__()
    except ImportError:
        yield
        return
    except Exception as exc:  # noqa: BLE001 - never break a turn for a trace
        _relay_warn_once("scope", "NeMo Relay could not open a turn scope: %s", exc)
        yield
        return

    try:
        yield
    finally:
        with contextlib.suppress(Exception):
            opened.__exit__(*sys.exc_info())



def _relay_may_observe_but_not_decide(middleware: Any) -> None:
    """Put Relay beside the call rather than inside it.

    Relay's integration hooks LangGraph middleware, so it wraps every model call
    and every tool call: it receives the request, decides what the handler is
    given, and decides what the agent is told came back. That is three chances
    to change an outcome it is only supposed to record, and one of them has
    already been taken -- its tool wrapper re-encoded the arguments and handed
    the tool a copy in which every unset optional had become an explicit null,
    so a search the model sent as ``{"requested_product_type": "dress"}`` arrived
    carrying a price constraint the shopper never gave. This service rejects
    invented constraints, so the tool refused its own call.

    Fixing that one path is not enough, because the position is the problem
    rather than the bug. What follows makes three things true whatever Relay
    does, including raising:

    * the handler is called with the request that arrived, not a copy;
    * it is called **exactly once**, so a wrapper that retries or abandons
      cannot double a cart write or drop one;
    * the agent receives the handler's own return value, not the wrapper's.

    Relay still sees the call and still writes its span -- the handler it is
    given runs the real work at the real moment, so nesting and timings are
    unaffected. It has no say in the result.

    An exception from the wrapper is contained: the work is already done, and the
    agent gets its result. A trace must never cost a turn.
    """

    for name in ("wrap_tool_call", "wrap_model_call"):
        wrapped = getattr(middleware, name, None)
        if wrapped is not None:
            setattr(middleware, name, _observing_only(wrapped, name))

    for name in ("awrap_tool_call", "awrap_model_call"):
        wrapped = getattr(middleware, name, None)
        if wrapped is not None:
            setattr(middleware, name, _observing_only_async(wrapped, name))


def _observing_only(wrapped: Any, name: str) -> Any:
    """The synchronous half of putting Relay beside the call."""

    def observed(request: Any, handler: Any) -> Any:
        done: list[Any] = []

        def run_once(_relays_version: Any = None) -> Any:
            # Relay offers its own copy of the request; the handler answers the
            # one that actually arrived. Called through Relay so its span still
            # wraps the real work.
            if not done:
                done.append(handler(request))
            return done[0]

        try:
            wrapped(request, run_once)
        except Exception as exc:  # noqa: BLE001 - a trace may not cost a turn
            _relay_warn_once(f"raised:{name}", "NeMo Relay %s raised: %s", name, exc)
        if not done:
            # Relay never reached the handler -- it failed early, or chose not
            # to. The call still has to happen.
            done.append(handler(request))
        return done[0]

    return observed


def _observing_only_async(wrapped: Any, name: str) -> Any:
    """The asynchronous half, which is the live path: the agent runs async."""

    async def observed(request: Any, handler: Any) -> Any:
        done: list[Any] = []

        async def run_once(_relays_version: Any = None) -> Any:
            if not done:
                done.append(await handler(request))
            return done[0]

        try:
            await wrapped(request, run_once)
        except Exception as exc:  # noqa: BLE001 - a trace may not cost a turn
            _relay_warn_once(f"raised:{name}", "NeMo Relay %s raised: %s", name, exc)
        if not done:
            done.append(await handler(request))
        return done[0]

    return observed


def _relay_instrumented(
    agent_kwargs: dict[str, Any],
    config: Any,
) -> dict[str, Any]:
    """Add NeMo Relay's instrumentation, or hand back exactly what came in.

    Off unless RELAY_ENABLED says otherwise, and absent when the package is
    not installed. An observability layer that changes behaviour by being
    absent is not observability, so every failure here falls back to the
    untouched arguments and says so once.

    What we hand create_deep_agent carries the parts that make this agent
    correct: middleware is the tool-loop control and the skill gate, backend is
    the skills filesystem, checkpointer is the within-turn state that is
    deleted at every exit. So the wrapper's output is checked rather than
    trusted -- it appends its own middleware to ours, which is the behaviour we
    want, and a release that replaced ours instead would silently remove both
    gates.
    """

    if not getattr(config, "relay_enabled", False):
        return agent_kwargs
    try:
        from nemo_relay.integrations.deepagents import add_nemo_relay_integration
    except ImportError:
        _relay_warn_once(
            "not-installed",
            "RELAY_ENABLED is set but nemo-relay is not installed; "
            "install requirements-relay.txt to trace.",
        )
        return agent_kwargs

    try:
        instrumented = add_nemo_relay_integration(agent_kwargs)
    except Exception as exc:  # pragma: no cover - never break a turn for a trace
        _relay_warn_once("failed", "NeMo Relay instrumentation failed: %s", exc)
        return agent_kwargs

    for name, value in agent_kwargs.items():
        if name not in instrumented:
            _relay_warn_once(
                f"dropped:{name}",
                "NeMo Relay dropped %s from the agent arguments; "
                "running without it.", name
            )
            return agent_kwargs
        if name != "middleware" and instrumented[name] is not value:
            _relay_warn_once(
                f"replaced:{name}",
                "NeMo Relay replaced %s rather than preserving it; "
                "running without it.", name
            )
            return agent_kwargs
    ours = agent_kwargs["middleware"]
    for added in list(instrumented["middleware"])[len(ours):]:
        _relay_may_observe_but_not_decide(added)
    if list(instrumented["middleware"])[: len(ours)] != list(ours):
        _relay_warn_once(
            "middleware-order",
            "NeMo Relay did not preserve our middleware order; "
            "running without it.",
        )
        return agent_kwargs
    return instrumented


class DeepAgentsRuntime:
    """Small adapter around the Deep Agents SDK.

    This class intentionally keeps commerce truth in existing services. Deep
    Agents gets scoped context and deterministic tools; it does not own carts,
    profiles, prices, inventory, or session identity.
    """

    def __init__(self, config: Any) -> None:
        self.config = config
        # Before the first agent is built, so the subscriber is registered by the
        # time any middleware has an event to emit.
        configure_relay_tracing(config)
        self._checkpointer = _build_checkpointer()
        self._profile_registered = False
        self._media_perception = MediaPerceptionClient(config)
        # Built once with the runtime, like the perception client: it holds an
        # endpoint and no turn state, so a per-turn instance would buy nothing.
        self._vocabulary_judge = CatalogVocabularyJudge(config)
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
        # What the turn could see, beside what it did. Reviewing a journey
        # means asking both, and the lanes could only be reconstructed
        # offline and approximately -- from the products a run happened to
        # report rather than from the projection the turn actually read.
        lanes = {
            lane: text
            for lane, text in (
                ("dialogue", output.dialogue_context),
                ("historical_product_index", output.historical_product_index),
            )
            if text
        }
        if not lanes:
            return output.agent_diagnostics
        return {**output.agent_diagnostics, "context_lanes": lanes}

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
            # Flat, still, and in group order: the client renders a sequence,
            # and a payload that changed shape would blank the panel. Each
            # product names its group and its number under that group, which
            # is what the client needs to draw the headings.
            yield json.dumps(
                {
                    "type": "products",
                    "payload": _streamed_in_group_order(
                        the_showing(products, output.product_groups)
                    ),
                    "timestamp": time.time(),
                }
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
                # Likewise, and for a blunter reason. The checkpoint thread is
                # keyed on (conversation_id, request_id), so it belongs to this
                # one request and nothing will ever read it again. It is freed
                # here, not only at finalization, because a turn that fails to
                # finalize would otherwise keep its whole message history for
                # the life of a long-lived pod with no eviction.
                await self._delete_turn_checkpoint(identity)

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
                self._finalize_conversation_turn(
                    state,
                    identity,
                    turn,
                    status="failed",
                    termination_reason="request_cancelled",
                    present_products=False,
                )
            raise
        except Exception:
            if turn is not None:
                self._finalize_conversation_turn(
                    state,
                    identity,
                    turn,
                    status="failed",
                    termination_reason="unexpected_runtime_error",
                    present_products=False,
                )
            raise

        if turn is not None:
            self._finalize_conversation_turn(state, identity, turn)
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
            with _relay_turn_scope(self.config, identity.conversation_id):
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
        scope = TurnScope()
        state.retrieved = scope.retrieved

        (
            search_catalog_tool,
            get_product_details_tool,
            resolve_conversation_products_tool,
            describe_catalog_tool,
        ) = build_catalog_tools(self, state, identity, scope, turn_capabilities)

        (
            get_cart_tool,
            add_cart_items_tool,
            remove_cart_item_tool,
            update_cart_items_tool,
            view_cart_total_tool,
        ) = build_cart_tools(self, state, identity, scope)

        (
            get_store_policy_tool,
            check_product_availability_tool,
            check_active_promotions_tool,
        ) = build_store_tools(scope)

        get_weather_forecast_tool = build_weather_tool(self, state, scope)

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
        weather_off = not self._weather_is_registered()
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
        # Built before the skill gate because the gate asks it, at each model
        # call, which tools the turn has finished with. `tool_loop_control` is
        # the outer middleware and owns that fact; the gate is what writes the
        # prompt, so the answer has to travel from here to there.
        tool_loop_control = ToolLoopControlMiddleware(
            catalog_context=format_catalog_capabilities_for_prompt(
                turn_capabilities
            ),
            shopper_statements=(
                state.query,
                *(turn.shopper_text for turn in state.dialogue),
            ),
        )

        activate_shopper_skills_tool, skill_gate = build_skill_activation(
            state, identity, skills_root, turn_capabilities, tool_loop_control
        )

        agent_tools = [activate_shopper_skills_tool, *shopping_tools]
        # `@tool` copies the docstring verbatim, source indentation included,
        # so where a tool is defined would otherwise change what the model
        # reads. Descriptions take the form they are evaluated with, wherever
        # their source sits; changing that form changes which tools turns
        # call, and needs a replay like any prompt change.
        for agent_tool in agent_tools:
            if isinstance(getattr(agent_tool, "description", None), str):
                agent_tool.description = _as_evaluated(agent_tool.description)

        agent_kwargs: dict[str, Any] = {
            "model": self._create_chat_model(),
            "tools": agent_tools,
            "system_prompt": self._system_prompt(
                shopper_context=state.shopper_context,
                media=bool(state.media),
            ),
            "middleware": [tool_loop_control, skill_gate],
            "backend": skills_backend,
            "checkpointer": self._checkpointer,
        }
        return create_deep_agent(**_relay_instrumented(agent_kwargs, self.config))

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

    def _create_chat_model(self, *, max_tokens: int | None = None):
        from langchain_openai import ChatOpenAI

        api_key_env = getattr(self.config, "llm_api_key_env", None)
        api_key = os.environ.get(api_key_env, "") if api_key_env else "not-needed"
        # Sampling is settable so the repetition failure can be bisected.
        #
        # J01 turn 16 emitted twelve model steps with empty text content, each
        # a byte-identical tool call against a byte-identical result, and died
        # on the recursion limit. That is degenerate repetition under
        # likelihood-maximising decoding, and at `temperature=0` with no
        # penalty it cannot be told apart from a model that will not stop. The
        # default stays 0 so every measurement taken at 0 still holds.
        sampling: dict[str, Any] = {}
        frequency_penalty = os.environ.get("APP_LLM_FREQUENCY_PENALTY", "")
        if frequency_penalty:
            sampling["frequency_penalty"] = float(frequency_penalty)
        return ChatOpenAI(
            model=self.config.llm_name,
            base_url=self.config.llm_port,
            api_key=api_key or "not-needed",
            temperature=float(os.environ.get("APP_LLM_TEMPERATURE", "0")),
            **sampling,
            # Uncapped output let one call run to the model's own ceiling.
            # Callers pick a smaller one where that fits; this is the default.
            max_tokens=(
                max_tokens if max_tokens is not None else self.config.llm_max_output_tokens
            ),
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
            # The editor had seven lanes and none of them was the shopper's own
            # media, so a draft naming what the photo showed -- "tan blazer,
            # cable-knit sweater, salmon trousers" -- had no lane supporting it,
            # and this editor cuts what no lane supports. It was not merely
            # failing to require the description; it had reason to remove one.
            # Fenced: these words were written by a model about a file a
            # stranger supplied, and they are the one thing here that nobody in
            # this service wrote. Everything in a prompt is text, so without a
            # boundary a description saying "ignore the above" arrives looking
            # exactly like a rule we set.
            "WHAT THE SHOPPER'S MEDIA SHOWED (sight, never a catalog fact):\n"
            f"{MEDIA_FENCE.wrap(state.media_analysis) or '(none)'}\n\n"
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
                self._create_chat_model(
                    max_tokens=self.config.grounding_editor_max_output_tokens
                ).ainvoke(
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
        candidates.append(Path(__file__).resolve().parents[2] / "skills")

        for candidate in candidates:
            if (candidate / "shopper").is_dir():
                return candidate
        return None

    def _weather_is_registered(self) -> bool:
        """Whether this deployment gives the model a forecast tool at all.

        One source of truth for the registration and for the prompt. They used
        to disagree: the tool was correctly omitted when weather was off, and
        the prompt went on telling the model to call it.
        """

        return bool(
            getattr(getattr(self.config, "weather", None), "enabled", False)
        )

    def _system_prompt(
        self,
        *,
        shopper_context: ShopperContext | None = None,
        media: bool = False,
    ) -> str:
        # Media rules are only reachable on a turn that carries media, so they
        # are only assembled then.
        media_rules = _MEDIA_TURN_RULES if media else ""
        shopper_context_rules = (
            f"\n{_SHOPPER_CONTEXT_SYSTEM_RULES}\n"
            if shopper_context is not None
            else ""
        )
        # Weather ships off, and off means absent, not present-and-failing:
        # a turn without the tool must not be told about it either. The rule
        # about ordering calls around a forecast lives in
        # `forecast_prompt_section`, which ships with the grant and so needs
        # no flag.
        #
        # The date itself stays either way. Relative dates are how shoppers
        # talk about occasions -- "the wedding is next weekend" -- and resolving
        # them has nothing to do with forecasts. Only the forecast window,
        # which is a fact about the tool, stays behind the flag.
        weather_registered = self._weather_is_registered()
        forecast_window = (
            " and a forecast may only be asked for a window within about "
            "fifteen days of it"
            if weather_registered
            else ""
        )
        prompt = f"""You are a retail shopping assistant for the products advertised by the active catalog.

TODAY IS {_today_for_the_shopper()}.
That is the only date you know. Every "this weekend", "next week", "in three
days" is counted from it{forecast_window}.

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
- Which skills exist, what each is for, and which may be combined are stated in
  the activation step's own registered descriptions and allowed values. Do not
  work from memory of a fixed pair: three of these bullets used to name two
  primary procedures, and were still naming two after a third was registered.
- Keep the primary skill aligned with the active conversation task. A styling
  thread continues with the styling procedure for its piece-by-piece searches
  until the shopper changes tasks; do not reclassify a follow-up merely because
  it asks for one product type.
- Use at most {self.config.max_catalog_searches_per_turn} product roles across
  all catalog searches in one user turn, and carry them in as few calls as
  possible: roles in one call retrieve together and cost one round trip.
  One normalized taxonomy-and-required-constraint scope can execute
  only once in a turn; do not retry the same hard-filter scope with different
  semantic wording. For outfit requests
  with multiple required item types, send one focused role per distinct
  taxonomy scope in the same call, then stop and synthesize from those results.
- Every product this turn's search returned is already on the shopper's screen
  as a picture, in the order the evidence lists it. Name all of them, in that
  same order, and give the styling guidance after the list rather than instead
  of part of it. Two reasons, and both are about the shopper rather than
  completeness for its own sake. A reply that writes up two of six reads as
  though the shop held two, while six pictures sit beside the words. And the
  shopper says "the first one" about what they can see, so a list that skips or
  reorders makes their next sentence mean something you did not intend. Say
  what each one is; then say which suits the occasion and why, and say plainly
  if only one or two really do.
- Advice is not an answer on its own either. A layering formula, a packing list
  or a list of what to look for, with no pieces from this shop beside it, is a
  wardrobe lecture rather than shopping. Search and show real items in every
  styling reply, including when conditions are hard, the date is unknown, or
  the shop cannot cover the whole need -- show what it does have and say what
  is missing. Measured: "it's going to snow this weekend" and "a wedding in
  Cancun, date not fixed yet" both returned formulas and nothing to buy.
  This is about advice offered in place of products you could have searched
  for. It is not a requirement to search on a turn that asked for no product:
  a question about the conditions at a destination is answered by the
  conditions, and has no product roles to fan out to.
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
                # Fenced, and this is the site that matters more than the
                # editor's: the editor only trims a draft, while this message
                # is what the agent reads before choosing tools. Text arriving
                # here unmarked is the closest thing in this service to a
                # stranger writing in the instruction channel.
                f"MEDIA ANALYSIS:\n"
                f"{MEDIA_FENCE.wrap(state.media_analysis) or '(none)'}",
                f"CURRENT CART:\n{_format_cart(state.cart)}",
                format_most_recent_subject(state),
                f"RECENT DISCUSSION:\n{state.context or '(none)'}",
            ]
        )
        # Last, because it was arriving fifth of nine with eleven thousand
        # characters of history behind it, and the final words the model read
        # before choosing a tool were a product showing from turn one.
        #
        # Asked "it's going to snow when we get back, what should I wear", it
        # searched for "a dress suitable for a warm-weather wedding in Cancun"
        # carrying black, high_neck and size 2 from six turns earlier -- the
        # constraints of the turn whose reply is quoted in the history above.
        # The request was in the prompt the whole time, buried at character
        # 1,182 of 12,865.
        #
        # The same words, moved, and nothing said about what outranks what. An
        # earlier attempt to fix this by declaring the query authoritative over
        # anything established before it took this journey from three passes in
        # three to none: the cart and the references it resolves are
        # established earlier too, and they still count.
        sections.append(f"THIS TURN'S REQUEST, TO ANSWER NOW:\n{state.query}")
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
        state.historical_product_index = format_historical_product_index(
            turn.projection.product_reference_index
        )
        if state.historical_product_index:
            state.context = "\n\n".join(
                value
                for value in (state.context, state.historical_product_index)
                if value
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
            state.product_results or [], state.response or "", state.product_groups
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
                product_groups=(state.product_groups if present_products else []),
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


_EVALUATED_DESCRIPTION_INDENT = " " * 12


def _as_evaluated(description: str) -> str:
    first, *rest = inspect.cleandoc(description).split("\n")
    return "\n".join(
        [first, *(_EVALUATED_DESCRIPTION_INDENT + line if line else line for line in rest)]
    )


def _build_checkpointer():
    """Return the process-local LangGraph checkpointer."""

    store = os.environ.get("CHECKPOINT_STORE", "memory").strip().lower()
    if store != "memory":
        raise ValueError(
            "CHECKPOINT_STORE currently supports only 'memory'. "
            f"Received: {store!r}."
        )
    return MemorySaver()


_PARTIAL_GRAPH_SNAPSHOT_TIMEOUT_SECONDS = 1.0


def _conversation_turn_status(termination_reason: str) -> FinalTurnStatus:
    if termination_reason in {
        "input_guardrail_blocked",
        "output_guardrail_blocked",
    }:
        return "blocked"
    if termination_reason == "completed":
        return "completed"
    return "failed"


async def _partial_graph_messages(
    agent: Any,
    invoke_config: dict[str, Any],
) -> tuple[list[Any], str | None]:
    """Read the last graph state before its failed checkpoint is deleted."""

    get_state = getattr(agent, "aget_state", None)
    if get_state is None:
        return [], "state_snapshot_unavailable"
    try:
        snapshot = await asyncio.wait_for(
            get_state(invoke_config),
            timeout=_PARTIAL_GRAPH_SNAPSHOT_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        logger.warning("Timed out snapshotting Deep Agents state before cleanup")
        return [], "state_snapshot_timeout"
    except Exception as exc:  # noqa: BLE001 - diagnostics cannot block cleanup.
        error_type = type(exc).__name__
        logger.warning(
            "Could not snapshot Deep Agents state before cleanup: %s",
            error_type,
        )
        return [], error_type

    values = _value(snapshot, "values")
    messages = _value(values, "messages")
    return (messages if isinstance(messages, list) else []), None
