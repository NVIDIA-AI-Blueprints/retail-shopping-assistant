# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What guardrails do to the shape of a turn.

The guardrail client itself is covered in `test_guardrails.py`. This module
covers the runtime around it: what runs before the input decision, what is
allowed to overlap it, what a block or a provider error costs, and what a turn
with guardrails off is spared.

Every test here drives `_execute_turn` with the catalog and the perception
client stubbed, so the only live behaviour is the ordering the guardrails
impose.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from chain_server.src.agenttypes import DialogueTurn, State
from chain_server.src.catalog_capabilities import CatalogCapabilities
from chain_server.src.guardrails import GuardrailDecision
from chain_server.src.runtime.identity import RequestIdentity
from chain_server.src.runtime.runtime import DeepAgentsRuntime


async def _no_media(_state) -> str:
    return ""


def _capabilities() -> CatalogCapabilities:
    return CatalogCapabilities(
        catalog_id="test-catalog",
        retrieval_modes=["text"],
        filters={},
    )


def always(agent):
    """Build the same agent whatever the turn asks for."""

    return lambda _state, _identity, _turn_capabilities=None: agent


@pytest.fixture
def turn_identity() -> RequestIdentity:
    return RequestIdentity(
        session_id="session-a",
        conversation_id="conversation-a",
        cart_id="cart-a",
        context_user_id=111,
        cart_user_id=222,
        request_id="request-a",
    )


@pytest.fixture
def build_runtime(base_config, monkeypatch: pytest.MonkeyPatch):
    """A runtime with everything but the guardrails stubbed out.

    Config is set before construction because the runtime reads the guardrail
    timeout and URL in `__init__`.
    """

    def build(guardrails, create_agent, *, analyze=_no_media, **config_overrides):
        for name, value in config_overrides.items():
            setattr(base_config, name, value)
        runtime = DeepAgentsRuntime(base_config)
        runtime._guardrails = guardrails
        monkeypatch.setattr(runtime._media_perception, "analyze", analyze)
        monkeypatch.setattr(runtime._catalog_capabilities, "get", _capabilities)
        monkeypatch.setattr(runtime, "_create_agent", create_agent)
        return runtime

    return build


@pytest.mark.asyncio
async def test_speculative_main_model_overlaps_input_guardrail_but_tools_wait(
    base_config,
    build_runtime,
    turn_identity,
) -> None:
    guardrail_started = asyncio.Event()
    release_guardrail = asyncio.Event()
    agent_started = asyncio.Event()
    tool_executed = asyncio.Event()

    class DelayedGuardrails:
        async def check_input(self, **_kwargs):
            guardrail_started.set()
            await release_guardrail.wait()
            return GuardrailDecision(status="allow", stage="input")

        async def check_output(self, **_kwargs):
            return GuardrailDecision(status="allow", stage="output")

    class FakeAgent:
        def __init__(self, tool_gate):
            self.tool_gate = tool_gate

        async def ainvoke(self, _payload, config):
            assert config["configurable"]["thread_id"] == (
                turn_identity.checkpoint_thread_id
            )
            agent_started.set()
            await self.tool_gate.wait()
            tool_executed.set()
            return {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "I can help with apparel shopping.",
                        "usage_metadata": {
                            "input_tokens": 10,
                            "output_tokens": 5,
                            "total_tokens": 15,
                        },
                    }
                ]
            }

    def fake_create_agent(
        _state,
        _identity,
        _turn_capabilities=None,
        *,
        input_guardrail_tool_gate=None,
    ):
        assert input_guardrail_tool_gate is not None
        return FakeAgent(input_guardrail_tool_gate)

    runtime = build_runtime(
        DelayedGuardrails(),
        fake_create_agent,
        guardrails_speculative_main_model_enabled=True,
        grounding_rewrite_enabled=False,
    )

    pending = asyncio.create_task(
        runtime._execute_turn(
            State(user_id=111, query="help me shop", guardrails=True),
            turn_identity,
        )
    )
    await asyncio.wait_for(guardrail_started.wait(), timeout=1)
    await asyncio.wait_for(agent_started.wait(), timeout=1)

    assert not tool_executed.is_set()

    release_guardrail.set()
    output = await asyncio.wait_for(pending, timeout=1)

    assert tool_executed.is_set()
    assert output.response == "I can help with apparel shopping."
    assert output.timings["input_guardrail_model_overlap"] > 0
    assert output.model_usage["app_llm"]["calls"] == 1


@pytest.mark.asyncio
async def test_speculative_input_block_cancels_model_and_executes_no_tool(
    base_config,
    build_runtime,
    turn_identity,
) -> None:
    release_guardrail = asyncio.Event()
    agent_started = asyncio.Event()
    agent_cancelled = asyncio.Event()
    tool_executed = asyncio.Event()

    class BlockingGuardrails:
        async def check_input(self, **_kwargs):
            await release_guardrail.wait()
            return GuardrailDecision(status="block", stage="input")

    class FakeAgent:
        def __init__(self, tool_gate):
            self.tool_gate = tool_gate

        async def ainvoke(self, _payload, config):
            agent_started.set()
            try:
                await self.tool_gate.wait()
                tool_executed.set()
            finally:
                agent_cancelled.set()

    def fake_create_agent(
        _state,
        _identity,
        _turn_capabilities=None,
        *,
        input_guardrail_tool_gate=None,
    ):
        return FakeAgent(input_guardrail_tool_gate)

    runtime = build_runtime(
        BlockingGuardrails(),
        fake_create_agent,
        guardrails_speculative_main_model_enabled=True,
    )

    pending = asyncio.create_task(
        runtime._execute_turn(
            State(user_id=111, query="blocked", guardrails=True),
            turn_identity,
        )
    )
    await asyncio.wait_for(agent_started.wait(), timeout=1)
    release_guardrail.set()
    output = await asyncio.wait_for(pending, timeout=1)

    assert output.response == base_config.unsafe_message
    assert agent_cancelled.is_set()
    assert not tool_executed.is_set()
    assert output.model_usage["app_llm"]["status"] == "failed"
    assert output.model_usage["app_llm"]["calls"] == 1
    assert "may still be billed" in output.model_usage["app_llm"]["detail"]


@pytest.mark.asyncio
async def test_default_mode_waits_for_input_guardrail_before_model(
    build_runtime,
    turn_identity,
) -> None:
    guardrail_started = asyncio.Event()
    release_guardrail = asyncio.Event()
    agent_started = asyncio.Event()
    input_kwargs = {}

    class DelayedGuardrails:
        async def check_input(self, **kwargs):
            input_kwargs.update(kwargs)
            guardrail_started.set()
            await release_guardrail.wait()
            return GuardrailDecision(status="allow", stage="input")

        async def check_output(self, **_kwargs):
            return GuardrailDecision(status="allow", stage="output")

    class FakeAgent:
        async def ainvoke(self, _payload, config):
            agent_started.set()
            return {
                "messages": [{"role": "assistant", "content": "Ready to shop."}]
            }

    runtime = build_runtime(
        DelayedGuardrails(),
        always(FakeAgent()),
        guardrails_speculative_main_model_enabled=False,
        grounding_rewrite_enabled=False,
    )

    pending = asyncio.create_task(
        runtime._execute_turn(
            State(
                user_id=111,
                query="what about the first one?",
                guardrails=True,
                dialogue=[
                    DialogueTurn(
                        sequence=1,
                        shopper_text="Show me dresses",
                        assistant_text="Here are two dresses.",
                    )
                ],
            ),
            turn_identity,
        )
    )
    await asyncio.wait_for(guardrail_started.wait(), timeout=1)
    await asyncio.sleep(0)

    assert not agent_started.is_set()
    assert input_kwargs["conversation"] == [
        {"role": "user", "content": "Show me dresses"},
        {"role": "assistant", "content": "Here are two dresses."},
    ]

    release_guardrail.set()
    output = await asyncio.wait_for(pending, timeout=1)

    assert agent_started.is_set()
    assert output.response == "Ready to shop."
    assert "input_guardrail_model_overlap" not in output.timings


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_mode", ["open", "closed"])
async def test_input_guardrail_error_obeys_failure_mode(
    base_config,
    build_runtime,
    turn_identity,
    failure_mode: str,
) -> None:
    agent_called = False

    class ErroringInputGuardrails:
        async def check_input(self, **_kwargs):
            return GuardrailDecision(
                status="error",
                stage="input",
                diagnostic_code="test_input_failure",
            )

        async def check_output(self, **_kwargs):
            return GuardrailDecision(status="allow", stage="output")

    class FakeAgent:
        async def ainvoke(self, _payload, config):
            nonlocal agent_called
            agent_called = True
            return {
                "messages": [{"role": "assistant", "content": "Ready to shop."}]
            }

    runtime = build_runtime(
        ErroringInputGuardrails(),
        always(FakeAgent()),
        guardrails_failure_mode=failure_mode,
        grounding_rewrite_enabled=False,
    )

    output = await runtime._execute_turn(
        State(user_id=111, query="help me shop", guardrails=True),
        turn_identity,
    )

    if failure_mode == "open":
        assert agent_called
        assert output.response == "Ready to shop."
        assert output.agent_diagnostics["final_termination_reason"] == "completed"
    else:
        assert not agent_called
        assert output.response == base_config.guardrails_unavailable_message
        assert output.agent_diagnostics["final_termination_reason"] == (
            "input_guardrail_error"
        )
    assert output.guardrail_results[0] == {
        "stage": "input",
        "status": "error",
        "violated_categories": [],
        "latency_ms": 0.0,
        "model_calls": {},
    }
    assert [result["stage"] for result in output.guardrail_results] == (
        ["input", "output"] if failure_mode == "open" else ["input"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_mode", ["open", "closed"])
async def test_output_guardrail_error_obeys_failure_mode(
    base_config,
    build_runtime,
    turn_identity,
    failure_mode: str,
) -> None:
    class ErroringOutputGuardrails:
        async def check_input(self, **_kwargs):
            return GuardrailDecision(status="allow", stage="input")

        async def check_output(self, **_kwargs):
            return GuardrailDecision(
                status="error",
                stage="output",
                diagnostic_code="test_output_failure",
            )

    class FakeAgent:
        async def ainvoke(self, _payload, config):
            return {
                "messages": [
                    {"role": "assistant", "content": "Here is a safe answer."}
                ]
            }

    runtime = build_runtime(
        ErroringOutputGuardrails(),
        always(FakeAgent()),
        guardrails_failure_mode=failure_mode,
        grounding_rewrite_enabled=False,
    )
    state = State(user_id=111, query="help me shop", guardrails=True)
    state.product_results = [{"product_id": "product-a"}]
    state.retrieved = {"Product A": "/images/product-a.jpg"}

    output = await runtime._execute_turn(state, turn_identity)

    if failure_mode == "open":
        assert output.response == "Here is a safe answer."
        assert output.product_results == [{"product_id": "product-a"}]
        assert output.retrieved == {"Product A": "/images/product-a.jpg"}
        assert output.agent_diagnostics["final_termination_reason"] == "completed"
    else:
        assert output.response.startswith(base_config.guardrails_unavailable_message)
        assert output.product_results == []
        assert output.retrieved == {}
        assert output.agent_diagnostics["final_termination_reason"] == (
            "output_guardrail_error"
        )


@pytest.mark.asyncio
async def test_a_turn_with_guardrails_off_reaches_no_guardrail(
    build_runtime,
    turn_identity,
) -> None:
    """Off means absent, not allowed: no call, no timing, no reported model."""

    class RefusingGuardrails:
        async def check_input(self, **_kwargs):
            raise AssertionError("input guardrail ran with guardrails off")

        async def check_output(self, **_kwargs):
            raise AssertionError("output guardrail ran with guardrails off")

    class FakeAgent:
        async def ainvoke(self, _payload, config):
            return {
                "messages": [{"role": "assistant", "content": "Ready to shop."}]
            }

    runtime = build_runtime(
        RefusingGuardrails(),
        always(FakeAgent()),
        grounding_rewrite_enabled=False,
    )

    output = await runtime._execute_turn(
        State(user_id=111, query="help me shop", guardrails=False),
        turn_identity,
    )

    assert output.response == "Ready to shop."
    assert output.guardrail_results == []
    assert "safety_input" not in output.timings
    assert "safety_output" not in output.timings
    assert "input_guardrail_model_overlap" not in output.timings
    assert "content_safety" not in output.model_usage
    assert "topic_control" not in output.model_usage


@pytest.mark.asyncio
@pytest.mark.parametrize("guardrails", [False, True])
async def test_media_card_is_withheld_only_from_a_guarded_turn(
    build_runtime,
    turn_identity,
    guardrails: bool,
) -> None:
    """An unguarded turn keeps the card; nothing would vet it later anyway."""

    analysis = json.dumps(
        {"summary": "a cream cable-knit sweater", "fashion_items": []}
    )

    class AllowingGuardrails:
        async def check_input(self, **_kwargs):
            return GuardrailDecision(status="allow", stage="input")

        async def check_output(self, **_kwargs):
            return GuardrailDecision(status="allow", stage="output")

    class FakeAgent:
        async def ainvoke(self, _payload, config):
            return {
                "messages": [{"role": "assistant", "content": "Ready to shop."}]
            }

    async def fake_analyze(state):
        state.media_analysis = analysis
        return analysis

    emitted: list[dict] = []
    runtime = build_runtime(
        AllowingGuardrails(),
        always(FakeAgent()),
        analyze=fake_analyze,
        grounding_rewrite_enabled=False,
    )

    await runtime._execute_turn(
        State(
            user_id=111,
            query="what goes with this?",
            guardrails=guardrails,
            media=[{"type": "image", "data": "data:image/png;base64,AAAA"}],
        ),
        turn_identity,
        on_progress=lambda chunk: emitted.append(json.loads(chunk)),
    )

    kinds = [message["type"] for message in emitted]
    if guardrails:
        assert "media_analysis" not in kinds
        assert {"stage": "media_perception", "status": "completed"} in [
            message["payload"] for message in emitted
        ]
    else:
        assert "progress" not in kinds
        card = next(
            message for message in emitted if message["type"] == "media_analysis"
        )
        assert card["payload"]["summary"] == "a cream cable-knit sweater"
