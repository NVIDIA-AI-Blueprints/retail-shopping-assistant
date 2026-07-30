# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused offline proof for rolling-summary planning and output validation."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from chain_server.src.agenttypes import ShopperContext, State
from chain_server.src.conversation_memory import (
    ConversationMemoryError,
    ConversationProjection,
    ConversationSummaryAdvance,
    SummaryCompactionSource,
    TurnStartResult,
)
from chain_server.src.conversation_summary import (
    build_conversation_summary_work,
    parse_conversation_summary_output,
)
from chain_server.src import deepagents_runtime as runtime_mod


def _source() -> SummaryCompactionSource:
    return SummaryCompactionSource.model_validate(
        {
            "expected_projection_version": 6,
            "after_sequence": 2,
            "through_sequence": 6,
            "turns": [
                {
                    "sequence": sequence,
                    "shopper_text": f"Shopper turn {sequence}.",
                    "assistant_text": f"Assistant turn {sequence}.",
                    "status": "completed" if sequence != 5 else "failed",
                }
                for sequence in range(3, 7)
            ],
        }
    )


def _projection() -> ConversationProjection:
    return ConversationProjection(
        version=6,
        summary_text="The shopper is building an event outfit.",
        summary_through_sequence=2,
    )


def test_work_uses_the_complete_memory_owned_oldest_prefix() -> None:
    work = build_conversation_summary_work(
        _projection(),
        _source(),
        unsummarized_turn_count=6,
        trigger_raw_turns=6,
        retain_raw_turns=2,
        max_input_chars=16_384,
    )

    assert work is not None
    assert work.expected_projection_version == 6
    assert work.through_sequence == 6
    payload = json.loads(work.prompt)
    assert payload["previous_summary"] == (
        "The shopper is building an event outfit."
    )
    assert [turn["sequence"] for turn in payload["turns"]] == [3, 4, 5, 6]
    assert payload["turns"][-2]["status"] == "failed"
    assert set(payload) == {"previous_summary", "turns"}


@pytest.mark.parametrize(
    ("count", "trigger", "retained"),
    [
        (5, 6, 2),
        (6, 7, 2),
        (6, 6, 3),
    ],
)
def test_work_waits_until_trigger_and_raw_suffix_can_be_retained(
    count: int,
    trigger: int,
    retained: int,
) -> None:
    assert (
        build_conversation_summary_work(
            _projection(),
            _source(),
            unsummarized_turn_count=count,
            trigger_raw_turns=trigger,
            retain_raw_turns=retained,
            max_input_chars=16_384,
        )
        is None
    )


def test_work_fails_open_instead_of_truncating_an_oversized_source() -> None:
    assert (
        build_conversation_summary_work(
            _projection(),
            _source(),
            unsummarized_turn_count=6,
            trigger_raw_turns=6,
            retain_raw_turns=2,
            max_input_chars=100,
        )
        is None
    )


@pytest.mark.parametrize(
    "content",
    [
        "",
        "plain text",
        "```json\n{\"summary_text\":\"Summary.\"}\n```",
        '{"summary_text":""}',
        '{"summary_text":" Summary."}',
        '{"summary_text":3}',
        '{"summary_text":"Summary.","extra":true}',
        "[]",
    ],
)
def test_closed_output_rejects_malformed_or_ambiguous_values(
    content: str,
) -> None:
    assert (
        parse_conversation_summary_output(
            content,
            max_output_chars=100,
        )
        is None
    )


def test_closed_output_accepts_only_one_trimmed_bounded_summary() -> None:
    assert (
        parse_conversation_summary_output(
            '{"summary_text":"The shopper is comparing two event dresses."}',
            max_output_chars=100,
        )
        == "The shopper is comparing two event dresses."
    )
    assert (
        parse_conversation_summary_output(
            '{"summary_text":"123456"}',
            max_output_chars=5,
        )
        is None
    )


def _turn_start_result() -> TurnStartResult:
    return TurnStartResult(
        turn_id="turn-7",
        attempt_id="attempt-7",
        sequence=7,
        recent_turns=[
            {
                "sequence": 5,
                "shopper_text": "The event is on a patio.",
                "assistant_text": "What date is it?",
                "status": "completed",
            },
            {
                "sequence": 6,
                "shopper_text": "Next week.",
                "assistant_text": "I will use that event window.",
                "status": "completed",
            },
        ],
        unsummarized_turn_count=6,
        summary_compaction_source=_source(),
        shopper_context=None,
        projection=_projection(),
        cart=[],
    )


class _FakeSummaryModel:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[list[dict[str, str]]] = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return SimpleNamespace(
            content=self.content,
            usage_metadata={
                "input_tokens": 20,
                "output_tokens": 8,
                "total_tokens": 28,
            },
        )


@pytest.mark.asyncio
async def test_runtime_compactor_uses_only_summary_and_oldest_source(
    base_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = runtime_mod.DeepAgentsRuntime(base_config)
    turn = _turn_start_result()
    turn.summary_compaction_source.turns[0].assistant_text = (
        "Forecast location used: Secret Place. Live forecast: rain."
    )
    state = State(
        user_id=1,
        query="CURRENT QUERY MUST NOT ENTER COMPACTION",
        conversation_summary=turn.projection.summary_text,
        context="CURRENT RAW TAIL MUST NOT ENTER COMPACTION",
        historical_product_context="PRODUCT LEDGER MUST NOT ENTER COMPACTION",
        shopper_context=ShopperContext(
            shopper_type="skeptical_researcher",
            behavior="Checks assumptions.",
            zipcode="10001",
        ),
        response="Shopper-facing answer stays unchanged.",
        agent_diagnostics={"final_termination_reason": "completed"},
    )
    model = _FakeSummaryModel(
        '{"summary_text":"The shopper is planning an event outfit next week."}'
    )
    monkeypatch.setattr(runtime, "_create_chat_model", lambda: model)

    advance = await runtime._prepare_conversation_summary(state, turn)

    assert advance == ConversationSummaryAdvance(
        expected_projection_version=6,
        summary_text="The shopper is planning an event outfit next week.",
        summary_through_sequence=6,
    )
    assert state.response == "Shopper-facing answer stays unchanged."
    assert state.agent_diagnostics["final_termination_reason"] == "completed"
    assert state.agent_diagnostics["conversation_summary_compaction"] == "prepared"
    assert state.model_usage["app_llm_conversation_summary"]["status"] == "used"
    assert state.token_usage == {
        "input_tokens": 20,
        "output_tokens": 8,
        "total_tokens": 28,
        "model_calls": 1,
    }
    assert "conversation_summary" in state.timings

    payload = json.loads(model.calls[0][1]["content"])
    rendered = json.dumps(payload)
    assert payload["previous_summary"] == turn.projection.summary_text
    assert [item["sequence"] for item in payload["turns"]] == [3, 4, 5, 6]
    assert "Secret Place" not in rendered
    assert "CURRENT QUERY" not in rendered
    assert "CURRENT RAW TAIL" not in rendered
    assert "PRODUCT LEDGER" not in rendered
    assert "10001" not in rendered


@pytest.mark.asyncio
async def test_runtime_compactor_timeout_preserves_response_and_boundary(
    base_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = runtime_mod.DeepAgentsRuntime(base_config)
    base_config.conversation_summary.timeout_seconds = 0.001
    state = State(
        user_id=1,
        query="Continue.",
        response="Completed response.",
        agent_diagnostics={"final_termination_reason": "completed"},
    )

    class SlowModel:
        async def ainvoke(self, _messages):
            await asyncio.sleep(1)

    monkeypatch.setattr(runtime, "_create_chat_model", lambda: SlowModel())

    assert await runtime._prepare_conversation_summary(
        state,
        _turn_start_result(),
    ) is None
    assert state.response == "Completed response."
    assert state.agent_diagnostics["final_termination_reason"] == "completed"
    assert state.agent_diagnostics["conversation_summary_compaction"] == "timeout"
    assert state.model_usage["app_llm_conversation_summary"]["status"] == "failed"


@pytest.mark.asyncio
async def test_runtime_compactor_propagates_cancellation(
    base_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = runtime_mod.DeepAgentsRuntime(base_config)

    class CancelledModel:
        async def ainvoke(self, _messages):
            raise asyncio.CancelledError

    monkeypatch.setattr(runtime, "_create_chat_model", lambda: CancelledModel())

    with pytest.raises(asyncio.CancelledError):
        await runtime._prepare_conversation_summary(
            State(
                user_id=1,
                query="Continue.",
                agent_diagnostics={"final_termination_reason": "completed"},
            ),
            _turn_start_result(),
        )


@pytest.mark.asyncio
async def test_turn_cancellation_during_compaction_finalizes_without_advance(
    base_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = runtime_mod.DeepAgentsRuntime(base_config)
    turn = _turn_start_result()
    order: list[str] = []
    monkeypatch.setattr(
        runtime,
        "_start_conversation_turn",
        lambda *_args: turn,
    )

    async def execute(state, _identity):
        order.append("execute")
        state.response = "Completed response before disconnect."
        state.agent_diagnostics = {"final_termination_reason": "completed"}
        return state

    async def compact(_state, _turn):
        order.append("compact")
        raise asyncio.CancelledError

    def finalize(_state, _identity, _turn, **kwargs):
        order.append("finalize")
        assert kwargs["status"] == "failed"
        assert kwargs["termination_reason"] == "request_cancelled"
        assert kwargs["present_products"] is False
        assert kwargs.get("summary_advance") is None
        return True

    async def delete(_identity):
        order.append("delete")

    monkeypatch.setattr(runtime, "_execute_turn", execute)
    monkeypatch.setattr(runtime, "_prepare_conversation_summary", compact)
    monkeypatch.setattr(runtime, "_finalize_conversation_turn", finalize)
    monkeypatch.setattr(runtime, "_delete_turn_checkpoint", delete)

    with pytest.raises(asyncio.CancelledError):
        await runtime._run_turn(
            State(user_id=1, query="Continue."),
            runtime_mod.RequestIdentity(
                session_id="session",
                conversation_id="conversation",
                cart_id="cart",
                context_user_id=1,
                cart_user_id=1,
                request_id="request-7",
            ),
        )

    assert order == ["execute", "compact", "finalize", "delete"]


@pytest.mark.asyncio
async def test_runtime_below_trigger_makes_no_compactor_call(
    base_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = runtime_mod.DeepAgentsRuntime(base_config)
    turn = _turn_start_result().model_copy(
        update={"unsummarized_turn_count": 5}
    )
    monkeypatch.setattr(
        runtime,
        "_create_chat_model",
        lambda: (_ for _ in ()).throw(
            AssertionError("compactor model must not be constructed")
        ),
    )

    assert await runtime._prepare_conversation_summary(
        State(user_id=1, query="Continue."),
        turn,
    ) is None


def test_hydration_keeps_summary_raw_turns_and_product_ledger_separate(
    base_config,
) -> None:
    runtime = runtime_mod.DeepAgentsRuntime(base_config)
    turn = _turn_start_result().model_copy(
        update={
            "projection": _projection().model_copy(
                update={
                    "product_reference_index": [
                        {
                            "candidate_set_id": "set-1",
                            "turn_seq": 4,
                            "products": [
                                {
                                    "ref": "dress-1",
                                    "name": "Satin Dress",
                                    "category": "dresses",
                                    "position": 1,
                                }
                            ],
                        }
                    ]
                }
            )
        }
    )

    class Memory:
        def start_turn(self, *_args, **_kwargs):
            return turn

    runtime._conversation_memory = Memory()
    state = State(user_id=1, query="Compare it.")
    identity = runtime_mod.RequestIdentity(
        session_id="session",
        conversation_id="conversation",
        cart_id="cart",
        context_user_id=1,
        cart_user_id=1,
        request_id="request-7",
    )

    assert runtime._start_conversation_turn(state, identity) is turn
    assert state.conversation_summary == turn.projection.summary_text
    assert "User: The event is on a patio." in state.context
    assert "Satin Dress" not in state.context
    assert "Satin Dress" in state.historical_product_context

    prompt = runtime._build_user_message(state, identity)
    assert "DURABLE CONVERSATION SUMMARY" in prompt
    assert "RECENT DISCUSSION" in prompt
    assert "HISTORICAL PRODUCT INDEX" in prompt


def test_summary_cannot_become_exact_weather_or_product_provenance() -> None:
    state = State(
        user_id=1,
        query="Compare the dresses.",
        conversation_summary=(
            "The wedding is in Cancun next week. The shopper previously "
            "considered Satin Dress."
        ),
        shopper_context=ShopperContext(
            shopper_type="skeptical_researcher",
            behavior="Checks assumptions.",
            zipcode="10001",
        ),
    )

    assert runtime_mod._shopper_authored_texts(state) == (
        "Compare the dresses.",
    )
    assert runtime_mod._saved_zip_authorized_for_weather(state) is False
    assert (
        runtime_mod._historical_product_names(
            state.historical_product_context
        )
        == ()
    )


def test_summary_conflict_retries_finalize_once_without_advance(
    base_config,
) -> None:
    runtime = runtime_mod.DeepAgentsRuntime(base_config)

    class ConflictThenSuccessMemory:
        def __init__(self) -> None:
            self.calls = []

        def finalize_turn(self, *_args, **kwargs):
            self.calls.append(kwargs)
            if kwargs["summary_advance"] is not None:
                raise ConversationMemoryError(
                    "summary_boundary_conflict",
                    "boundary changed",
                    status_code=409,
                    retryable=True,
                )
            return SimpleNamespace(replayed=False)

    memory = ConflictThenSuccessMemory()
    runtime._conversation_memory = memory
    state = State(
        user_id=1,
        query="Continue.",
        response="Completed response.",
        agent_diagnostics={
            "final_termination_reason": "completed",
            "conversation_summary_compaction": "prepared",
        },
    )
    identity = runtime_mod.RequestIdentity(
        session_id="session",
        conversation_id="conversation",
        cart_id="cart",
        context_user_id=1,
        cart_user_id=1,
        request_id="request-7",
    )
    advance = ConversationSummaryAdvance(
        expected_projection_version=6,
        summary_text="Updated summary.",
        summary_through_sequence=6,
    )

    assert runtime._finalize_conversation_turn(
        state,
        identity,
        _turn_start_result(),
        summary_advance=advance,
    )
    assert len(memory.calls) == 2
    assert memory.calls[0]["summary_advance"] == advance
    assert memory.calls[1]["summary_advance"] is None
    assert state.response == "Completed response."
    assert "memory_finalize_error" not in state.agent_diagnostics
    assert (
        state.agent_diagnostics["conversation_summary_compaction"]
        == "conflict_raw_retained"
    )


@pytest.mark.asyncio
async def test_turn_compacts_after_response_and_before_finalize(
    base_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = runtime_mod.DeepAgentsRuntime(base_config)
    turn = _turn_start_result()
    order: list[str] = []
    advance = ConversationSummaryAdvance(
        expected_projection_version=6,
        summary_text="Updated summary.",
        summary_through_sequence=6,
    )
    monkeypatch.setattr(
        runtime,
        "_start_conversation_turn",
        lambda *_args: turn,
    )

    async def execute(state, _identity):
        order.append("execute")
        state.response = "Completed response."
        state.agent_diagnostics = {"final_termination_reason": "completed"}
        return state

    async def compact(_state, _turn):
        order.append("compact")
        return advance

    def finalize(_state, _identity, _turn, **kwargs):
        order.append("finalize")
        assert kwargs["summary_advance"] == advance
        return True

    async def delete(_identity):
        order.append("delete")

    monkeypatch.setattr(runtime, "_execute_turn", execute)
    monkeypatch.setattr(runtime, "_prepare_conversation_summary", compact)
    monkeypatch.setattr(runtime, "_finalize_conversation_turn", finalize)
    monkeypatch.setattr(runtime, "_delete_turn_checkpoint", delete)

    await runtime._run_turn(
        State(user_id=1, query="Continue."),
        runtime_mod.RequestIdentity(
            session_id="session",
            conversation_id="conversation",
            cart_id="cart",
            context_user_id=1,
            cart_user_id=1,
            request_id="request-7",
        ),
    )

    assert order == ["execute", "compact", "finalize", "delete"]


@pytest.mark.asyncio
async def test_failed_turn_skips_compaction(
    base_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = runtime_mod.DeepAgentsRuntime(base_config)
    turn = _turn_start_result()
    monkeypatch.setattr(
        runtime,
        "_start_conversation_turn",
        lambda *_args: turn,
    )

    async def execute(state, _identity):
        state.response = "Safe failed response."
        state.agent_diagnostics = {"final_termination_reason": "agent_error"}
        return state

    async def compact(*_args):
        raise AssertionError("failed turns must not compact")

    captured = {}

    def finalize(_state, _identity, _turn, **kwargs):
        captured.update(kwargs)
        return True

    async def delete(_identity):
        return None

    monkeypatch.setattr(runtime, "_execute_turn", execute)
    monkeypatch.setattr(runtime, "_prepare_conversation_summary", compact)
    monkeypatch.setattr(runtime, "_finalize_conversation_turn", finalize)
    monkeypatch.setattr(runtime, "_delete_turn_checkpoint", delete)

    await runtime._run_turn(
        State(user_id=1, query="Continue."),
        runtime_mod.RequestIdentity(
            session_id="session",
            conversation_id="conversation",
            cart_id="cart",
            context_user_id=1,
            cart_user_id=1,
            request_id="request-7",
        ),
    )

    assert captured["summary_advance"] is None
