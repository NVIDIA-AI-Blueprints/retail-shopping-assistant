# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One durable conversation turn: started, replayed, or finalized."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from pydantic import ValidationError

from ..agenttypes import Cart, State
from ..conversation_memory import (
    ConversationMemoryError,
    FinalTurnStatus,
    TurnReplayOutput,
    TurnStartResult,
    build_dialogue_context,
)
from ..conversation_products import format_historical_product_index
from .audience_events import _system_identification_events, _turn_audience_events
from .identity import RequestIdentity
from .message_shape import _value
from .replies import _images_in_product_order, _in_presentation_order
from .turn_diagnostics import _empty_agent_diagnostics

logger = logging.getLogger(__name__)


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


class ConversationTurnMixin:
    """The `DeepAgentsRuntime` methods that persist a turn in conversation memory."""

    async def _delete_turn_checkpoint(self, identity: RequestIdentity) -> None:
        try:
            delete_thread = getattr(self._checkpointer, "adelete_thread", None)
            if delete_thread is not None:
                await delete_thread(identity.checkpoint_thread_id)
            else:
                self._checkpointer.delete_thread(identity.checkpoint_thread_id)
        except Exception as exc:  # pragma: no cover - cleanup is best effort.
            logger.warning("Could not delete Deep Agents turn checkpoint: %s", exc)

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
                guardrails_enabled=state.guardrails,
            )
        except (ConversationMemoryError, ValidationError) as exc:
            logger.error("Failed to start durable conversation turn: %s", exc)
            state.dialogue = []
            state.historical_product_sets = []
            state.dialogue_context = ""
            state.context = ""
            state.cart = Cart()
            error_code = getattr(exc, "code", "memory_start_payload_invalid")
            if getattr(exc, "status_code", None) == 409:
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
            state.guardrails = turn.output.guardrails_enabled
            state.guardrail_results = list(turn.output.guardrail_results)
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
        # the same list.
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
                guardrails_enabled=state.guardrails,
                guardrail_results=state.guardrail_results,
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
