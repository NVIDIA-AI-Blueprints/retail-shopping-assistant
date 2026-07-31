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
    ("subject_relation", "pending_disposition"),
    [
        ("same_subject", "not_addressed"),
        ("same_subject", "answered"),
        ("new_subject", "not_addressed"),
        ("unchanged", "not_addressed"),
        ("unchanged", "resume_requested"),
        ("unclear", "not_addressed"),
    ],
)
def test_contract_exposes_each_canonical_outcome(
    subject_relation: str,
    pending_disposition: str,
) -> None:
    schema = WeatherScopeResolverDecision.model_json_schema()

    assert subject_relation in schema["properties"]["subject_relation"]["enum"]
    assert pending_disposition in schema["properties"]["pending_disposition"][
        "enum"
    ]


@pytest.mark.parametrize(
    "subject_relation",
    ["same_subject", "new_subject", "unchanged", "unclear"],
)
def test_not_addressed_outcome_is_relation_only(
    subject_relation: str,
) -> None:
    parsed = parse_weather_scope_resolver_tool_call(
        _resolver_call(
            {
                "subject_relation": subject_relation,
                "pending_disposition": "not_addressed",
            }
        )
    )

    assert parsed is not None
    assert parsed.subject_relation == subject_relation
    assert parsed.pending_disposition == "not_addressed"
    assert parsed.pending_source_turn_id is None


def test_resolver_rejects_scope_extraction_fields() -> None:
    parsed = parse_weather_scope_resolver_tool_call(
        _resolver_call(
            {
                "subject_relation": "new_subject",
                "pending_disposition": "not_addressed",
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
                "subject_relation": "same_subject",
                "pending_disposition": "answered",
                "pending_source_turn_id": "pending-turn",
            }
        )
    )

    assert parsed is not None
    assert parsed.subject_relation == "same_subject"
    assert parsed.pending_disposition == "answered"
    assert parsed.pending_source_turn_id == "pending-turn"


@pytest.mark.parametrize(
    ("subject_relation", "pending_disposition"),
    [
        ("same_subject", "not_addressed"),
        ("new_subject", "not_addressed"),
        ("unchanged", "not_addressed"),
        ("unclear", "not_addressed"),
    ],
)
def test_not_addressed_outcome_must_omit_pending_handle(
    subject_relation: str,
    pending_disposition: str,
) -> None:
    parsed = parse_weather_scope_resolver_tool_call(
        _resolver_call(
            {
                "subject_relation": subject_relation,
                "pending_disposition": pending_disposition,
                "pending_source_turn_id": "pending-turn",
            }
        )
    )

    assert parsed is None


@pytest.mark.parametrize("pending_disposition", ["answered", "resume_requested"])
def test_pending_control_must_include_pending_handle(
    pending_disposition: str,
) -> None:
    subject_relation = (
        "same_subject"
        if pending_disposition == "answered"
        else "unchanged"
    )

    assert parse_weather_scope_resolver_tool_call(
        _resolver_call(
            {
                "subject_relation": subject_relation,
                "pending_disposition": pending_disposition,
            }
        )
    ) is None


@pytest.mark.parametrize(
    ("subject_relation", "pending_disposition"),
    [
        ("same_subject", "resume_requested"),
        ("new_subject", "answered"),
        ("new_subject", "resume_requested"),
        ("unchanged", "answered"),
        ("unclear", "answered"),
        ("unclear", "resume_requested"),
    ],
)
def test_contract_rejects_noncanonical_axis_combinations(
    subject_relation: str,
    pending_disposition: str,
) -> None:
    assert parse_weather_scope_resolver_tool_call(
        _resolver_call(
            {
                "subject_relation": subject_relation,
                "pending_disposition": pending_disposition,
                "pending_source_turn_id": "pending-turn",
            }
        )
    ) is None


def test_contract_rejects_legacy_mixed_decision() -> None:
    assert parse_weather_scope_resolver_tool_call(
        _resolver_call({"decision": "unchanged"})
    ) is None


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
                    "subject_relation": "same_subject",
                    "pending_disposition": "answered",
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
                            "subject_relation": "unchanged",
                            "pending_disposition": "not_addressed",
                        },
                    },
                    {
                        "id": "two",
                        "name": "WeatherScopeResolverDecision",
                        "args": {
                            "subject_relation": "unchanged",
                            "pending_disposition": "not_addressed",
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
                    "subject_relation": "unchanged",
                    "pending_disposition": "not_addressed",
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
                    "subject_relation": "unchanged",
                    "pending_disposition": "not_addressed",
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
    assert "choose exactly one subject_relation" in normalized
    assert "separately choose exactly one pending_disposition" in normalized
    assert "return only these semantic axes" in normalized
    assert "do not extract, normalize, copy, or author location" in normalized
    assert "top-level pending_source_turn_id" in normalized
    assert "shopper answers only the supplied pending question" in normalized
    assert "explicitly asks what information is still needed" in normalized
    assert "should be asked again without changing its scope" in normalized
    assert "changes or withdraws the stored opposite component" in normalized
    assert "use same_subject/not_addressed" in normalized
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
                "subject_relation": "new_subject",
                "pending_disposition": "not_addressed",
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
    assert decision.subject_relation == "new_subject"
    assert decision.pending_disposition == "not_addressed"
    assert decision.pending_source_turn_id is None
    assert len(model.calls) == 1
    assert model.bindings[0]["tool_choice"] == "WeatherScopeResolverDecision"
    assert model.bindings[0]["parallel_tool_calls"] is False
    assert state.model_usage["app_llm_weather_scope_resolver"]["status"] == "used"


@pytest.mark.asyncio
async def test_v5_resolver_uses_exact_scope_sources_outside_recent_context(
    base_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WEATHER_API_KEY", "test-weather-key")
    base_config.weather.enabled = True
    model = _ResolverModel(
        _resolver_call(
            {
                "subject_relation": "same_subject",
                "pending_disposition": "not_addressed",
            }
        )
    )
    runtime = DeepAgentsRuntime(base_config)
    monkeypatch.setattr(runtime, "_create_chat_model", lambda: model)
    state = State(
        user_id=1,
        query="Refresh that wedding forecast.",
        conversation_projection_version=4,
        conversation_memory_contract_version=5,
        current_weather_scope=_current_scope(),
        current_weather_scope_source_turns=[
            {
                "turn_id": "wedding-turn",
                "sequence": 1,
                "shopper_text": (
                    "Give me weather styling for my NYC wedding on August 8."
                ),
                "assistant_text": "Here is the wedding forecast guidance.",
                "status": "completed",
            }
        ],
        recent_conversation_turns=[
            {
                "sequence": 8,
                "shopper_text": "Show me black flats.",
                "assistant_text": "Here are the current catalog candidates.",
            }
        ],
    )

    decision = await runtime._resolve_existing_weather_scope(
        state,
        current_utc_date=date(2026, 7, 30),
        execution_deadline=time.monotonic() + 10,
    )

    assert decision is not None
    assert decision.subject_relation == "same_subject"
    assert decision.pending_disposition == "not_addressed"
    resolver_payload = json.loads(model.calls[0][1]["content"])
    assert resolver_payload["scope_subject_context"]["turns"] == [
        {
            "sequence": 1,
            "shopper_text": (
                "Give me weather styling for my NYC wedding on August 8."
            ),
            "assistant_text": "Here is the wedding forecast guidance.",
        }
    ]
    assert resolver_payload["semantic_context"]["recent_turns"] == (
        state.recent_conversation_turns
    )


@pytest.mark.asyncio
async def test_v5_resolver_answers_pending_from_source_outside_recent_context(
    base_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WEATHER_API_KEY", "test-weather-key")
    base_config.weather.enabled = True
    model = _ResolverModel(
        _resolver_call(
            {
                "subject_relation": "same_subject",
                "pending_disposition": "answered",
                "pending_source_turn_id": "conference-turn",
            }
        )
    )
    runtime = DeepAgentsRuntime(base_config)
    monkeypatch.setattr(runtime, "_create_chat_model", lambda: model)
    state = State(
        user_id=1,
        query="Seattle.",
        conversation_projection_version=4,
        conversation_memory_contract_version=5,
        current_weather_scope=CurrentWeatherScope.model_validate(
            {
                "revision": 2,
                "pending_question": "event_location",
                "pending_source_turn_id": "conference-turn",
                "pending_source_sequence": 2,
                "window": {
                    "value": {
                        "start_date": "2026-08-12",
                        "end_date": "2026-08-12",
                    },
                    "source_turn_id": "conference-turn",
                    "source_sequence": 2,
                },
            }
        ),
        current_weather_scope_source_turns=[
            {
                "turn_id": "conference-turn",
                "sequence": 2,
                "shopper_text": (
                    "My outdoor conference is August 12. "
                    "I still need to give you the city."
                ),
                "assistant_text": "What location should I plan around?",
                "status": "completed",
            }
        ],
        recent_conversation_turns=[
            {
                "sequence": 8,
                "shopper_text": "Show me black flats.",
                "assistant_text": "Here are the current catalog candidates.",
            }
        ],
    )

    decision = await runtime._resolve_existing_weather_scope(
        state,
        current_utc_date=date(2026, 7, 30),
        execution_deadline=time.monotonic() + 10,
    )

    assert decision is not None
    assert decision.subject_relation == "same_subject"
    assert decision.pending_disposition == "answered"
    assert decision.pending_source_turn_id == "conference-turn"
    resolver_payload = json.loads(model.calls[0][1]["content"])
    assert resolver_payload["scope_subject_context"]["turns"][0][
        "shopper_text"
    ].startswith("My outdoor conference")
    assert resolver_payload["semantic_context"]["recent_turns"] == (
        state.recent_conversation_turns
    )


@pytest.mark.asyncio
async def test_v5_resolver_rejects_same_sequence_with_wrong_source_turn(
    base_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WEATHER_API_KEY", "test-weather-key")
    base_config.weather.enabled = True
    model = _ResolverModel(
        _resolver_call(
            {
                "subject_relation": "same_subject",
                "pending_disposition": "not_addressed",
            }
        )
    )
    runtime = DeepAgentsRuntime(base_config)
    monkeypatch.setattr(runtime, "_create_chat_model", lambda: model)
    state = State(
        user_id=1,
        query="Refresh that event forecast.",
        conversation_memory_contract_version=5,
        current_weather_scope=_current_scope(),
        current_weather_scope_source_turns=[
            {
                "turn_id": "different-turn",
                "sequence": 1,
                "shopper_text": "A different event on the same sequence.",
                "assistant_text": "Different event guidance.",
                "status": "completed",
            }
        ],
    )

    decision = await runtime._resolve_existing_weather_scope(
        state,
        current_utc_date=date(2026, 7, 30),
        execution_deadline=time.monotonic() + 10,
    )

    assert decision is not None
    assert decision.subject_relation == "unclear"
    assert decision.pending_disposition == "not_addressed"
    assert model.calls == []
    assert state.model_usage["app_llm_weather_scope_resolver"]["status"] == (
        "not_used"
    )


@pytest.mark.asyncio
async def test_runtime_resolver_rejects_answers_pending_without_handle(
    base_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WEATHER_API_KEY", "test-weather-key")
    base_config.weather.enabled = True
    model = _ResolverModel(
        _resolver_call(
            {
                "subject_relation": "same_subject",
                "pending_disposition": "answered",
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
    assert decision.subject_relation == "unclear"
    assert decision.pending_disposition == "not_addressed"
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
                "subject_relation": "same_subject",
                "pending_disposition": "answered",
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
    assert decision.subject_relation == "unclear"
    assert decision.pending_disposition == "not_addressed"
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
                "subject_relation": "same_subject",
                "pending_disposition": "answered",
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
    assert decision.subject_relation == "same_subject"
    assert decision.pending_disposition == "answered"
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
        _resolver_call(
            {
                "subject_relation": "unchanged",
                "pending_disposition": "not_addressed",
            }
        )
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
    assert decision.subject_relation == "unchanged"
    assert decision.pending_disposition == "not_addressed"
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
    assert decision.subject_relation == "unclear"
    assert decision.pending_disposition == "not_addressed"
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
    assert decision.subject_relation == "unclear"
    assert decision.pending_disposition == "not_addressed"
    assert model.calls == []
    assert model.bindings == []
    assert state.model_usage["app_llm_weather_scope_resolver"]["status"] == (
        "not_used"
    )
