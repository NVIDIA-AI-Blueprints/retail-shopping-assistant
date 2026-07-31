# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused proof for the typed weather-scope semantic control contract."""

from __future__ import annotations

from datetime import date
import json
import time

from langchain_core.messages import AIMessage
import pytest

from chain_server.src.agenttypes import State
from chain_server.src.deepagents_runtime import (
    DeepAgentsRuntime,
    _format_current_weather_scope,
)
from chain_server.src.weather_scope_resolver import (
    WEATHER_SCOPE_RESOLVER_SYSTEM_PROMPT,
    WeatherScopeResolverDecision,
    build_weather_scope_resolver_prompt,
    parse_weather_scope_resolver_tool_call,
)
from shared.weather_scope import CurrentWeatherScope


def _resolver_call(
    arguments: dict[str, object],
    *,
    name: str = "WeatherScopeResolverDecision",
) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "id": "resolver-decision",
                "name": name,
                "args": arguments,
            }
        ],
    )


@pytest.mark.parametrize(
    "decision",
    ["same_subject", "new_subject", "answers_pending", "unchanged", "unclear"],
)
def test_contract_exposes_each_semantic_decision(decision: str) -> None:
    schema = WeatherScopeResolverDecision.model_json_schema()

    assert decision in schema["properties"]["decision"]["enum"]


@pytest.mark.parametrize(
    "decision",
    ["same_subject", "new_subject", "unchanged", "unclear"],
)
def test_non_pending_decision_is_relation_only(decision: str) -> None:
    parsed = parse_weather_scope_resolver_tool_call(
        _resolver_call({"decision": decision})
    )

    assert parsed is not None
    assert parsed.decision == decision
    assert parsed.pending_source_turn_id is None


def test_resolver_rejects_scope_extraction_fields() -> None:
    parsed = parse_weather_scope_resolver_tool_call(
        _resolver_call(
            {
                "decision": "new_subject",
                "resolution": {
                    "scope_revision": 4,
                    "location_action": "clear",
                    "window_action": "set",
                    "date": "2026-08-15",
                },
            }
        )
    )

    assert parsed is None


def test_pending_answer_requires_top_level_exact_handle_shape() -> None:
    parsed = parse_weather_scope_resolver_tool_call(
        _resolver_call(
            {
                "decision": "answers_pending",
                "pending_source_turn_id": "pending-turn",
            }
        )
    )

    assert parsed is not None
    assert parsed.decision == "answers_pending"
    assert parsed.pending_source_turn_id == "pending-turn"


@pytest.mark.parametrize(
    "decision",
    ["same_subject", "new_subject", "unchanged", "unclear"],
)
def test_non_pending_resolver_decision_must_omit_pending_handle(
    decision: str,
) -> None:
    parsed = parse_weather_scope_resolver_tool_call(
        _resolver_call(
            {
                "decision": decision,
                "pending_source_turn_id": "pending-turn",
            }
        )
    )

    assert parsed is None


def test_answers_pending_must_include_pending_handle() -> None:
    assert (
        parse_weather_scope_resolver_tool_call(
            _resolver_call({"decision": "answers_pending"})
        )
        is None
    )


@pytest.mark.parametrize(
    "pending_source_turn_id",
    [" pending-turn", "pending-turn ", "pending\nturn", "pending\u007fturn"],
)
def test_pending_source_turn_handle_must_be_trimmed_and_single_line(
    pending_source_turn_id: str,
) -> None:
    assert (
        parse_weather_scope_resolver_tool_call(
            _resolver_call(
                {
                    "decision": "answers_pending",
                    "pending_source_turn_id": pending_source_turn_id,
                }
            )
        )
        is None
    )


def test_parser_fails_closed_without_exactly_one_control_call() -> None:
    assert parse_weather_scope_resolver_tool_call(AIMessage(content="prose")) is None
    assert (
        parse_weather_scope_resolver_tool_call(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "one",
                        "name": "WeatherScopeResolverDecision",
                        "args": {
                            "decision": "unchanged",
                        },
                    },
                    {
                        "id": "two",
                        "name": "WeatherScopeResolverDecision",
                        "args": {
                            "decision": "unchanged",
                        },
                    },
                ],
            )
        )
        is None
    )


def test_parser_fails_closed_on_wrong_control_name_or_arguments() -> None:
    assert (
        parse_weather_scope_resolver_tool_call(
            _resolver_call(
                {
                    "decision": "unchanged",
                },
                name="another_tool",
            )
        )
        is None
    )
    assert (
        parse_weather_scope_resolver_tool_call(
            _resolver_call(
                {
                    "decision": "unchanged",
                    "extra": True,
                }
            )
        )
        is None
    )


def test_prompt_builder_separates_non_authoritative_semantic_context() -> None:
    prompt = build_weather_scope_resolver_prompt(
        current_query="I also have a conference on August 15.",
        current_scope_json={
            "revision": 4,
            "location": {"value": {"kind": "shopper_provided_location"}},
            "window": None,
        },
        current_utc_date=date(2026, 7, 30),
        rolling_summary="The shopper was planning a wedding in NYC.",
        scope_source_turns=[
            {
                "sequence": 1,
                "shopper_text": "I need a wedding outfit.",
                "assistant_text": "What date is the wedding?",
            }
        ],
        recent_turns=[
            {
                "sequence": 1,
                "shopper_text": "I need a wedding outfit.",
                "assistant_text": "What date is the wedding?",
            }
        ],
    )

    payload = json.loads(prompt)
    assert payload["current_query"].startswith("I also have a conference")
    assert payload["current_scope"]["revision"] == 4
    assert payload["current_utc_date"] == "2026-07-30"
    assert payload["scope_subject_context"] == {
        "authority": "source_sequence_bound_semantic_identity_only",
        "turns": [
            {
                "sequence": 1,
                "shopper_text": "I need a wedding outfit.",
                "assistant_text": "What date is the wedding?",
            }
        ],
    }
    assert payload["semantic_context"] == {
        "authoritative": False,
        "rolling_summary": "The shopper was planning a wedding in NYC.",
        "recent_turns": [
            {
                "sequence": 1,
                "shopper_text": "I need a wedding outfit.",
                "assistant_text": "What date is the wedding?",
            }
        ],
    }
    assert "output_schema" not in payload
    assert "\n" not in prompt


def test_prompt_keeps_semantics_with_the_model_and_facts_outside_it() -> None:
    normalized = " ".join(
        WEATHER_SCOPE_RESOLVER_SYSTEM_PROMPT.lower().split()
    )

    assert "business-tool-disabled semantic decision" in normalized
    assert "do not establish location, date, forecast" in normalized
    assert "source_sequence" in normalized
    assert "semantic identity of the subject" in normalized
    assert "return only that semantic relation" in normalized
    assert "do not extract, normalize, copy, or author location" in normalized
    assert "top-level pending_source_turn_id" in normalized
    assert "answers only the supplied pending question" in normalized
    assert "changes or withdraws the opposite component" in normalized
    assert "choose same_subject instead" in normalized
    assert "exactly one required weatherscoperesolverdecision" in normalized


class _ResolverModel:
    def __init__(self, result: AIMessage) -> None:
        self.result = result
        self.calls: list[object] = []
        self.bindings: list[dict[str, object]] = []

    def bind_tools(self, tools: object, **kwargs: object) -> "_ResolverModel":
        self.bindings.append({"tools": tools, **kwargs})
        return self

    async def ainvoke(self, messages: object) -> AIMessage:
        self.calls.append(messages)
        return self.result


def _current_scope() -> CurrentWeatherScope:
    return CurrentWeatherScope.model_validate(
        {
            "revision": 1,
            "location": {
                "value": {
                    "kind": "shopper_provided_location",
                    "location": "NYC",
                    "location_query": "NYC, NY",
                },
                "source_turn_id": "wedding-turn",
                "source_sequence": 1,
            },
            "window": {
                "value": {
                    "start_date": "2026-08-08",
                    "end_date": "2026-08-08",
                },
                "source_turn_id": "wedding-turn",
                "source_sequence": 1,
            },
        }
    )


@pytest.mark.asyncio
async def test_runtime_resolver_isolates_a_new_subject_before_activation(
    base_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WEATHER_API_KEY", "test-weather-key")
    base_config.weather.enabled = True
    model = _ResolverModel(
        _resolver_call(
            {
                "decision": "new_subject",
            }
        )
    )
    runtime = DeepAgentsRuntime(base_config)
    monkeypatch.setattr(runtime, "_create_chat_model", lambda: model)
    state = State(
        user_id=1,
        query="I also have a conference on August 15.",
        conversation_projection_version=4,
        conversation_memory_contract_version=4,
        current_weather_scope=_current_scope(),
        conversation_summary="The shopper has an NYC wedding on August 8.",
        recent_conversation_turns=[
            {
                "sequence": 1,
                "shopper_text": (
                    "Give me weather styling for my NYC wedding on August 8."
                ),
                "assistant_text": "Here is the wedding forecast guidance.",
            }
        ],
    )

    decision = await runtime._resolve_existing_weather_scope(
        state,
        current_utc_date=date(2026, 7, 30),
        execution_deadline=time.monotonic() + 10,
    )

    assert decision is not None
    assert decision.decision == "new_subject"
    assert decision.pending_source_turn_id is None
    assert len(model.calls) == 1
    assert model.bindings[0]["tool_choice"] == "WeatherScopeResolverDecision"
    assert model.bindings[0]["parallel_tool_calls"] is False
    assert state.model_usage["app_llm_weather_scope_resolver"]["status"] == "used"


@pytest.mark.asyncio
async def test_runtime_resolver_rejects_answers_pending_without_handle(
    base_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WEATHER_API_KEY", "test-weather-key")
    base_config.weather.enabled = True
    model = _ResolverModel(
        _resolver_call({"decision": "answers_pending"})
    )
    runtime = DeepAgentsRuntime(base_config)
    monkeypatch.setattr(runtime, "_create_chat_model", lambda: model)
    state = State(
        user_id=1,
        query="August 15.",
        conversation_projection_version=4,
        conversation_memory_contract_version=4,
        current_weather_scope=CurrentWeatherScope.model_validate(
            {
                "revision": 1,
                "pending_question": "event_date",
                "pending_source_turn_id": "wedding-turn",
                "pending_source_sequence": 1,
                "location": {
                    "value": {
                        "kind": "shopper_provided_location",
                        "location": "NYC",
                        "location_query": "NYC, NY",
                    },
                    "source_turn_id": "wedding-turn",
                    "source_sequence": 1,
                },
            }
        ),
        recent_conversation_turns=[
            {
                "sequence": 1,
                "shopper_text": (
                    "Give me weather styling for my NYC wedding on August 8."
                ),
                "assistant_text": "Here is the wedding forecast guidance.",
            }
        ],
    )

    decision = await runtime._resolve_existing_weather_scope(
        state,
        current_utc_date=date(2026, 7, 30),
        execution_deadline=time.monotonic() + 10,
    )

    assert decision is not None
    assert decision.decision == "unclear"
    assert state.model_usage["app_llm_weather_scope_resolver"]["status"] == (
        "failed"
    )


@pytest.mark.asyncio
async def test_runtime_resolver_rejects_answers_pending_with_wrong_handle(
    base_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WEATHER_API_KEY", "test-weather-key")
    base_config.weather.enabled = True
    model = _ResolverModel(
        _resolver_call(
            {
                "decision": "answers_pending",
                "pending_source_turn_id": "different-turn",
            }
        )
    )
    runtime = DeepAgentsRuntime(base_config)
    monkeypatch.setattr(runtime, "_create_chat_model", lambda: model)
    state = State(
        user_id=1,
        query="August 15.",
        conversation_projection_version=4,
        conversation_memory_contract_version=4,
        current_weather_scope=CurrentWeatherScope.model_validate(
            {
                "revision": 1,
                "pending_question": "event_date",
                "pending_source_turn_id": "wedding-turn",
                "pending_source_sequence": 1,
                "location": {
                    "value": {
                        "kind": "shopper_provided_location",
                        "location": "NYC",
                        "location_query": "NYC, NY",
                    },
                    "source_turn_id": "wedding-turn",
                    "source_sequence": 1,
                },
            }
        ),
        recent_conversation_turns=[
            {
                "sequence": 1,
                "shopper_text": "I need wedding styling for NYC.",
                "assistant_text": "What date is the wedding?",
            }
        ],
    )

    decision = await runtime._resolve_existing_weather_scope(
        state,
        current_utc_date=date(2026, 7, 30),
        execution_deadline=time.monotonic() + 10,
    )

    assert decision is not None
    assert decision.decision == "unclear"
    assert state.model_usage["app_llm_weather_scope_resolver"]["status"] == (
        "failed"
    )


@pytest.mark.asyncio
async def test_runtime_resolver_accepts_exact_pending_handle(
    base_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WEATHER_API_KEY", "test-weather-key")
    base_config.weather.enabled = True
    model = _ResolverModel(
        _resolver_call(
            {
                "decision": "answers_pending",
                "pending_source_turn_id": "wedding-turn",
            }
        )
    )
    runtime = DeepAgentsRuntime(base_config)
    monkeypatch.setattr(runtime, "_create_chat_model", lambda: model)
    current_scope = CurrentWeatherScope.model_validate(
        {
            "revision": 1,
            "pending_question": "event_date",
            "pending_source_turn_id": "wedding-turn",
            "pending_source_sequence": 1,
            "location": {
                "value": {
                    "kind": "shopper_provided_location",
                    "location": "NYC",
                    "location_query": "NYC, NY",
                },
                "source_turn_id": "wedding-turn",
                "source_sequence": 1,
            },
        }
    )
    state = State(
        user_id=1,
        query="August 15.",
        conversation_projection_version=4,
        conversation_memory_contract_version=4,
        current_weather_scope=current_scope,
        recent_conversation_turns=[
            {
                "sequence": 1,
                "shopper_text": "I need wedding styling for NYC.",
                "assistant_text": "What date is the wedding?",
            }
        ],
    )

    decision = await runtime._resolve_existing_weather_scope(
        state,
        current_utc_date=date(2026, 7, 30),
        execution_deadline=time.monotonic() + 10,
    )

    assert decision is not None
    assert decision.decision == "answers_pending"
    assert decision.pending_source_turn_id == "wedding-turn"
    assert state.model_usage["app_llm_weather_scope_resolver"]["status"] == (
        "used"
    )


def test_current_scope_formatter_hides_opaque_pending_handle() -> None:
    formatted = _format_current_weather_scope(
        CurrentWeatherScope.model_validate(
            {
                "revision": 1,
                "pending_question": "event_date",
                "pending_source_turn_id": "pending-turn",
                "pending_source_sequence": 2,
                "location": {
                    "value": {
                        "kind": "shopper_provided_location",
                        "location": "Seattle",
                    },
                    "source_turn_id": "location-turn",
                    "source_sequence": 1,
                },
            }
        )
    )

    assert "pending-turn" not in formatted
    assert "resolver-only" in formatted


@pytest.mark.asyncio
async def test_runtime_resolver_binds_an_empty_pending_scope_to_its_source_turn(
    base_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WEATHER_API_KEY", "test-weather-key")
    base_config.weather.enabled = True
    model = _ResolverModel(
        _resolver_call({"decision": "unchanged"})
    )
    runtime = DeepAgentsRuntime(base_config)
    monkeypatch.setattr(runtime, "_create_chat_model", lambda: model)
    state = State(
        user_id=1,
        query="Show me shoes for the look.",
        conversation_projection_version=4,
        conversation_memory_contract_version=4,
        current_weather_scope=CurrentWeatherScope.model_validate(
            {
                "revision": 1,
                "pending_question": "event_location",
                "pending_source_turn_id": "wedding-turn",
                "pending_source_sequence": 1,
            }
        ),
        recent_conversation_turns=[
            {
                "sequence": 1,
                "shopper_text": "I need an outfit for a wedding.",
                "assistant_text": "Where is the wedding?",
            }
        ],
    )

    decision = await runtime._resolve_existing_weather_scope(
        state,
        current_utc_date=date(2026, 7, 30),
        execution_deadline=time.monotonic() + 10,
    )

    assert decision is not None
    assert decision.decision == "unchanged"
    resolver_payload = json.loads(model.calls[0][1]["content"])
    assert resolver_payload["scope_subject_context"]["turns"] == [
        state.recent_conversation_turns[0]
    ]


@pytest.mark.asyncio
async def test_runtime_resolver_fails_closed_on_invalid_model_output(
    base_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WEATHER_API_KEY", "test-weather-key")
    base_config.weather.enabled = True
    model = _ResolverModel(AIMessage(content="not-a-control-call"))
    runtime = DeepAgentsRuntime(base_config)
    monkeypatch.setattr(runtime, "_create_chat_model", lambda: model)
    state = State(
        user_id=1,
        query="Maybe the other event.",
        conversation_projection_version=4,
        conversation_memory_contract_version=4,
        current_weather_scope=_current_scope(),
        recent_conversation_turns=[
            {
                "sequence": 1,
                "shopper_text": "I need wedding styling.",
                "assistant_text": "What should I help with next?",
            }
        ],
    )

    decision = await runtime._resolve_existing_weather_scope(
        state,
        current_utc_date=date(2026, 7, 30),
        execution_deadline=time.monotonic() + 10,
    )

    assert decision is not None
    assert decision.decision == "unclear"
    assert state.model_usage["app_llm_weather_scope_resolver"]["status"] == (
        "failed"
    )


@pytest.mark.asyncio
async def test_runtime_resolver_does_not_model_guess_without_source_turn(
    base_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WEATHER_API_KEY", "test-weather-key")
    base_config.weather.enabled = True
    model = _ResolverModel(AIMessage(content="must not be called"))
    runtime = DeepAgentsRuntime(base_config)
    monkeypatch.setattr(runtime, "_create_chat_model", lambda: model)
    state = State(
        user_id=1,
        query="What about the other event?",
        conversation_projection_version=4,
        conversation_memory_contract_version=4,
        current_weather_scope=_current_scope(),
    )

    decision = await runtime._resolve_existing_weather_scope(
        state,
        current_utc_date=date(2026, 7, 30),
        execution_deadline=time.monotonic() + 10,
    )

    assert decision is not None
    assert decision.decision == "unclear"
    assert model.calls == []
    assert model.bindings == []
    assert state.model_usage["app_llm_weather_scope_resolver"]["status"] == (
        "not_used"
    )
