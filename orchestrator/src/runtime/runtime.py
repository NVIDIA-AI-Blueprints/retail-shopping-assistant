# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deep Agents SDK runtime for the shopping assistant."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from shared.commerce_contracts import (
    CatalogCapabilities,
    GetCartInput,
)

from ..agenttypes import Cart, State
from ..catalog_capabilities import (
    CatalogCapabilitiesClient,
    format_catalog_capabilities_for_prompt,
)
from ..category_resolver import CatalogVocabularyJudge
from ..commerce_tools import (
    get_cart,
)
from ..conversation_memory import (
    ConversationMemoryClient,
)
from ..conversation_products import (
    ConversationProductsClient,
)
from ..guardrails import GuardrailDecision, GuardrailServiceClient, stops_turn
from ..media_perception import MEDIA_ONLY_QUERY, MediaPerceptionClient
from ..tools.cart import build_cart_tools
from ..tools.catalog import build_catalog_tools
from ..tools.loop_control import (
    ToolLoopControlMiddleware,
)
from ..tools.policy import (
    validate_registered_tool_names,
)
from ..tools.skill_gate import (
    InputGuardrailToolGate,
    ShopperSkillActivationError,
    selected_skill_names_for_turn,
)
from ..tools.skills import build_skill_activation
from ..tools.store import build_store_tools
from ..tools.weather import build_weather_tool
from ..weather import WeatherConfig, build_weather_client
from .control_signals import (
    committed_effects_in,
)
from .conversation_turns import ConversationTurnMixin, _partial_graph_messages
from .grounding import GroundingMixin
from .identity import RequestIdentity
from .message_shape import (
    _extract_final_text,
    _result_messages,
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
from .prompt_assembly import PromptAssemblyMixin
from .prompts import (
    _DEEP_AGENT_BASE_PROMPT,
)
from .replies import (
    _committed_effect_receipt,
    _media_failure_response,
    _partial_product_results_response,
    _products_found_receipt,
)
from .streaming import (
    CHECKING_MESSAGE_LABEL,
    CHECKING_REPLY_LABEL,
    WRITING_REPLY_LABEL,
    ToolProgressMiddleware,
    _emit_media_progress,
    _emit_progress_label,
    _streamed_in_group_order,
    media_progress_label,
    the_showing,
)
from .tracing import (
    _record_turn_diagnostics,
    _relay_instrumented,
    _relay_turn_scope,
    _turn_span,
    _turn_trace_session,
    configure_relay_tracing,
)
from .turn_diagnostics import (
    _empty_agent_diagnostics,
    _safe_collect_agent_diagnostics,
)
from .turn_scope import TurnScope

logger = logging.getLogger(__name__)


def _record_guardrail_result(state: State, decision: GuardrailDecision) -> None:
    """Keep only the provider's sanitized decision metadata for the UI."""

    state.guardrail_results.append(
        {
            "stage": decision.stage,
            "status": decision.status,
            "violated_categories": decision.violated_categories,
            "latency_ms": decision.latency_ms,
            "model_calls": decision.model_calls,
        }
    )


_SHOPPER_SKILLS_ENV = "SHOPPER_SKILLS_ROOT"


try:
    from deepagents.backends import FilesystemBackend as _FilesystemBackend
except Exception:  # pragma: no cover - dependency import is validated at runtime.
    _FilesystemBackend = None


#: Must match the base prompt's "You have no filesystem". Skill files are
#: already injected in full, so `read_file` would only waste a round trip.
_EXCLUDED_DEEP_AGENT_TOOLS = frozenset(
    {
        "write_todos",
        "ls",
        "read_file",
        "write_file",
        "edit_file",
        "delete",
        "glob",
        "grep",
        "execute",
    }
)
#: Names looked up in the catalog when a reference resolves to nothing. Bounded
#: because each is a retrieval.
#: The longest one model request may take before it is abandoned, whatever the
#: deployment's turn budget. A request slower than this has stalled.
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


async def _cancel_task(task: asyncio.Task[Any] | None) -> None:
    """Cancel and drain a request-local task without masking the caller outcome."""

    if task is None:
        return
    if not task.done():
        task.cancel()
    await asyncio.gather(task, return_exceptions=True)


def _guardrail_conversation(state: State) -> list[dict[str, str]]:
    """Return a small dialogue-only topic-classification context."""

    messages: list[dict[str, str]] = []
    for turn in state.dialogue[-4:]:
        for role, content in (
            ("user", turn.shopper_text),
            ("assistant", turn.assistant_text),
        ):
            if content:
                messages.append({"role": role, "content": content[:1_000]})
    return messages


def _record_discarded_speculative_usage(
    state: State,
    task: asyncio.Task[Any] | None,
) -> None:
    """Expose cost evidence for model work discarded by an input block."""

    usage: dict[str, int] = {}
    detail = (
        "Input guardrails stopped the turn; in-flight speculative model work "
        "was cancelled and may still be billed by the provider"
    )
    if task is not None and task.done() and not task.cancelled():
        try:
            result = task.result()
        except BaseException:
            detail = (
                "Speculative model work failed before the input-guardrail "
                "decision; provider billing is unknown"
            )
        else:
            usage = _collect_token_usage(result)
            detail = (
                "Completed speculative model output was discarded because "
                "input guardrails stopped the turn"
            )
            state.token_usage = _merge_token_usage(state.token_usage, usage)
    _add_model_usage(
        state,
        "app_llm",
        status="failed",
        calls=max(1, int(usage.get("model_calls") or 0)),
        detail=detail,
        tokens=int(usage.get("total_tokens") or 0),
    )


class DeepAgentsRuntime(GroundingMixin, ConversationTurnMixin, PromptAssemblyMixin):
    """Small adapter around the Deep Agents SDK.

    This class intentionally keeps commerce truth in existing services. Deep
    Agents gets scoped context and deterministic tools; it does not own carts,
    prices, inventory, or session identity.
    """

    def __init__(self, config: Any, *, guardrails: Any = None) -> None:
        self.config = config
        # Before the first agent is built, so the subscriber is registered by the
        # time any middleware has an event to emit.
        configure_relay_tracing(config)
        self._checkpointer = _build_checkpointer()
        self._profile_registered = False
        self._media_perception = MediaPerceptionClient(config)
        self._guardrails = guardrails or GuardrailServiceClient(
            config.guardrails_url,
            timeout_seconds=config.guardrails_timeout_seconds,
        )
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
        # styling.
        self._weather_client = build_weather_client(
            getattr(config, "weather", None) or WeatherConfig()
        )

    def catalog_capabilities(self) -> CatalogCapabilities:
        """Return the process-lifecycle catalog capability contract."""

        return self._catalog_capabilities.get()

    def _exposed_agent_diagnostics(self, output: State) -> dict[str, Any]:
        if not getattr(self.config, "expose_agent_diagnostics", False):
            return {}
        # What the turn could see, beside what it did. The lanes cannot be
        # reconstructed accurately offline, so they are recorded here.
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

    def _guardrail_report(self, output: State) -> dict[str, Any]:
        return {
            "enabled": output.guardrails,
            "failure_mode": self.config.guardrails_failure_mode,
            "checks": output.guardrail_results,
        }

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
            # A shopper who closes the tab or resets stops this generator, but
            # the turn is a separate task and must be cancelled explicitly.
            turn.cancel()
            raise
        # Re-raises whatever the turn raised, so failure handling is unchanged.
        output = await turn
        products = output.product_results or []
        if products:
            # Flat and in group order: a payload that changed shape would blank
            # the client panel. Each product names its group and index in it.
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
                    "guardrail_report": self._guardrail_report(output),
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
            "guardrail_report": self._guardrail_report(output),
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
                # The checkpoint thread is request-scoped and never read again.
                # Freed here too, or a turn that fails to finalize leaks it.
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
        turn_capabilities: CatalogCapabilities | None = None
        invoke_config: dict[str, Any] | None = None
        agent = None
        execution_deadline: float | None = None
        grounding_reserve: float | None = None
        speculative_agent_task: asyncio.Task[Any] | None = None
        speculative_gate: InputGuardrailToolGate | None = None
        speculative_agent_started_at: float | None = None
        speculative_agent_finished_at: float | None = None

        if state.guardrails:
            safety_start = time.monotonic()
            _emit_progress_label(on_progress, CHECKING_MESSAGE_LABEL)
            guardrail_conversation = _guardrail_conversation(state)
            speculative_enabled = (
                self.config.guardrails_speculative_main_model_enabled
                and not state.media
            )
            input_guardrail_task: asyncio.Task[Any] | None = None
            if speculative_enabled:
                speculative_gate = InputGuardrailToolGate()
                input_guardrail_task = asyncio.create_task(
                    self._guardrails.check_input(
                        text="" if state.query == MEDIA_ONLY_QUERY else state.query,
                        media=state.media,
                        conversation=guardrail_conversation,
                    )
                )
                try:
                    turn_capabilities = await asyncio.to_thread(
                        self._catalog_capabilities.get
                    )
                    invoke_config = self._agent_invoke_config(identity)
                    execution_deadline = (
                        time.monotonic()
                        + self.config.deepagents_execution_timeout_seconds
                    )
                    grounding_reserve = self._grounding_reserve()
                    agent = self._create_agent(
                        state,
                        identity,
                        turn_capabilities,
                        input_guardrail_tool_gate=speculative_gate,
                        on_progress=on_progress,
                    )
                    input_message = self._build_user_message(state, identity)
                    agent_timeout = max(
                        0.0,
                        execution_deadline
                        - time.monotonic()
                        - grounding_reserve,
                    )
                    if agent_timeout <= 0:
                        raise TimeoutError

                    async def invoke_speculatively() -> Any:
                        nonlocal speculative_agent_finished_at
                        try:
                            with _relay_turn_scope(
                                self.config, identity.conversation_id
                            ):
                                return await asyncio.wait_for(
                                    agent.ainvoke(
                                        {
                                            "messages": [
                                                {
                                                    "role": "user",
                                                    "content": input_message,
                                                }
                                            ]
                                        },
                                        config=invoke_config,
                                    ),
                                    timeout=agent_timeout,
                                )
                        finally:
                            speculative_agent_finished_at = time.monotonic()

                    speculative_agent_started_at = time.monotonic()
                    speculative_agent_task = asyncio.create_task(
                        invoke_speculatively()
                    )
                    input_decision = await input_guardrail_task
                except BaseException:
                    speculative_gate.deny()
                    await _cancel_task(speculative_agent_task)
                    await _cancel_task(input_guardrail_task)
                    raise
            else:
                input_decision = await self._guardrails.check_input(
                    text="" if state.query == MEDIA_ONLY_QUERY else state.query,
                    media=state.media,
                    conversation=guardrail_conversation,
                )
            state.timings["safety_input"] = time.monotonic() - safety_start
            if speculative_agent_started_at is not None:
                overlap_end = min(
                    time.monotonic(),
                    speculative_agent_finished_at or time.monotonic(),
                )
                state.timings["input_guardrail_model_overlap"] = max(
                    0.0,
                    overlap_end - speculative_agent_started_at,
                )
            _record_safety_model_usage(
                state,
                "input",
                model_calls=input_decision.model_calls,
                ok=input_decision.status != "error",
            )
            _record_guardrail_result(state, input_decision)
            if stops_turn(
                input_decision,
                self.config.guardrails_failure_mode,
            ):
                if speculative_gate is not None:
                    speculative_gate.deny()
                    _record_discarded_speculative_usage(
                        state,
                        speculative_agent_task,
                    )
                    await _cancel_task(speculative_agent_task)
                state.response = (
                    self.config.unsafe_message
                    if input_decision.status == "block"
                    else self.config.guardrails_unavailable_message
                )
                state.timings["deepagents"] = time.monotonic() - start
                state.agent_diagnostics = _empty_agent_diagnostics(
                    "input_guardrail_blocked"
                    if input_decision.status == "block"
                    else "input_guardrail_error"
                )
                return state
            if speculative_gate is not None:
                speculative_gate.allow()

        media_start = time.monotonic()
        if state.media:
            _emit_progress_label(on_progress, media_progress_label(state.media))
        try:
            state.media_analysis = await self._media_perception.analyze(state)
        except BaseException:
            # Text-only perception is normally a no-op, but a provider/client
            # regression here must not orphan a speculative graph task.
            await _cancel_task(speculative_agent_task)
            raise
        if state.media:
            state.timings["media_perception"] = time.monotonic() - media_start
            _record_media_model_usage(state, self.config)
        _emit_media_progress(on_progress, state)
        if _should_short_circuit_media_failure(state):
            state.response = _media_failure_response(state.media_analysis)
            state.timings["deepagents"] = time.monotonic() - start
            state.agent_diagnostics = _empty_agent_diagnostics("media_failure")
            return state

        if turn_capabilities is None:
            turn_capabilities = await asyncio.to_thread(
                self._catalog_capabilities.get
            )
        if invoke_config is None:
            invoke_config = self._agent_invoke_config(identity)
        try:
            if execution_deadline is None:
                execution_deadline = (
                    time.monotonic()
                    + self.config.deepagents_execution_timeout_seconds
                )
            if agent is None:
                agent = self._create_agent(
                    state,
                    identity,
                    turn_capabilities,
                    on_progress=on_progress,
                )
            input_message = self._build_user_message(state, identity)
            # The grounding editor is not optional, so its budget is reserved
            # before the agent loop runs rather than taken from what it leaves.
            # Never take more than half the turn, so a short budget still gets
            # an agent loop.
            if grounding_reserve is None:
                grounding_reserve = self._grounding_reserve()
            agent_timeout = max(
                0.0,
                execution_deadline - time.monotonic() - grounding_reserve,
            )
            if agent_timeout <= 0:
                raise TimeoutError
            if speculative_agent_task is not None:
                result = await speculative_agent_task
            else:
                with _relay_turn_scope(self.config, identity.conversation_id):
                    result = await asyncio.wait_for(
                        agent.ainvoke(
                            {
                                "messages": [
                                    {"role": "user", "content": input_message}
                                ]
                            },
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
            # not a reason for the runtime to seize the turn.
            # Never below the reserve, however long the loop actually ran.
            remaining_seconds = max(
                grounding_reserve,
                execution_deadline - time.monotonic(),
            )
            if self.config.grounding_rewrite_enabled:
                _emit_progress_label(on_progress, WRITING_REPLY_LABEL)
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
            _emit_progress_label(on_progress, CHECKING_REPLY_LABEL)
            output_decision = await self._guardrails.check_output(text=state.response)
            state.timings["safety_output"] = time.monotonic() - safety_start
            _record_safety_model_usage(
                state,
                "output",
                model_calls=output_decision.model_calls,
                ok=output_decision.status != "error",
            )
            _record_guardrail_result(state, output_decision)
            if stops_turn(
                output_decision,
                self.config.guardrails_failure_mode,
            ):
                state.product_results = []
                state.retrieved = {}
                state.response = (
                    self.config.unsafe_message
                    if output_decision.status == "block"
                    else self.config.guardrails_unavailable_message
                )
                state.response += (
                    " If this request may have changed your cart, please check "
                    "the cart before retrying."
                )
                state.agent_diagnostics[
                    "final_termination_reason"
                ] = (
                    "output_guardrail_blocked"
                    if output_decision.status == "block"
                    else "output_guardrail_error"
                )

        state.timings["deepagents"] = time.monotonic() - start
        return state

    def _create_agent(
        self,
        state: State,
        identity: RequestIdentity,
        turn_capabilities: CatalogCapabilities | None = None,
        *,
        input_guardrail_tool_gate: InputGuardrailToolGate | None = None,
        on_progress: Any = None,
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
        # Built before the skill gate, which asks it at each model call which
        # tools the turn has finished with.
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
            state,
            identity,
            skills_root,
            turn_capabilities,
            tool_loop_control,
            input_guardrail_tool_gate=input_guardrail_tool_gate,
        )

        agent_tools = [activate_shopper_skills_tool, *shopping_tools]
        # `@tool` copies the docstring verbatim, indentation included. Changing
        # this normalisation changes what the model reads; replay it like any
        # prompt change.
        for agent_tool in agent_tools:
            if isinstance(getattr(agent_tool, "description", None), str):
                agent_tool.description = _as_evaluated(agent_tool.description)

        agent_kwargs: dict[str, Any] = {
            "model": self._create_chat_model(),
            "tools": agent_tools,
            "system_prompt": self._system_prompt(media=bool(state.media)),
            "middleware": [
                tool_loop_control,
                skill_gate,
                *([ToolProgressMiddleware(on_progress)] if on_progress else []),
            ],
            "backend": skills_backend,
            "checkpointer": self._checkpointer,
        }
        return create_deep_agent(**_relay_instrumented(agent_kwargs, self.config))

    def _agent_invoke_config(self, identity: RequestIdentity) -> dict[str, Any]:
        return {
            "configurable": {"thread_id": identity.checkpoint_thread_id},
            "recursion_limit": self.config.deepagents_recursion_limit,
            # Trace-only: the checkpoint thread is request-scoped, so tracing
            # needs the conversation named explicitly. Nothing else reads it.
            "metadata": {"session_id": identity.conversation_id},
        }

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
        # Sampling is settable so degenerate tool-call repetition can be
        # bisected. Measurements taken at temperature 0 do not carry over.
        sampling: dict[str, Any] = {}
        frequency_penalty = os.environ.get("APP_LLM_FREQUENCY_PENALTY", "")
        if frequency_penalty:
            sampling["frequency_penalty"] = float(frequency_penalty)
        return ChatOpenAI(
            model=self.config.llm_name,
            base_url=self.config.llm_port,
            api_key=api_key or "not-needed",
            temperature=self.config.llm_temperature,
            **sampling,
            # Uncapped output let one call run to the model's own ceiling.
            # Callers pick a smaller one where that fits; this is the default.
            max_tokens=(
                max_tokens if max_tokens is not None else self.config.llm_max_output_tokens
            ),
            # Without a deadline a single stalled call can spend the whole turn
            # budget. Timing out also lets `max_retries` fire, since a hang is
            # not an error.
            timeout=self._model_request_timeout(),
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
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


#: Keys the catalog returns alongside the declared detail fields that are not
#: product attributes: retrieval bookkeeping, and taxonomy which has its own
#: lane. catalog_text is deliberately not forwarded: it mixes the same
#: attributes with a marketing summary.


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
