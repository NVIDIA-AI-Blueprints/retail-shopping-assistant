# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused runtime proof for bounded durable weather-receipt reuse."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from langchain_core.messages import AIMessage

from chain_server.src.agenttypes import State
from chain_server.src.conversation_memory import (
    ConversationMemoryClient,
    ConversationMemoryError,
    ConversationProjection,
    RecentConversationTurn,
    TurnStartResult,
)
from chain_server.src import deepagents_runtime as runtime_mod
from chain_server.src.weather import WeatherFailure
from chain_server.src.weather_tool import WEATHER_FORECAST_EVIDENCE_PREFIX
from shared.weather_receipts import (
    SavedAreaWeatherScope,
    ShopperLocationWeatherScope,
    WeatherForecastReceipt,
    WeatherReceiptAttribution,
    WeatherReceiptDay,
    WeatherReceiptEvidence,
    WeatherReceiptPromotion,
    WeatherReceiptWindow,
    weather_receipt_id,
    weather_scope_key,
)
from shared.weather_scope import (
    CurrentWeatherScope,
    CurrentWeatherScopeResolution,
    CurrentWeatherScopeTransition,
    WeatherScopeLocationAuthority,
    WeatherScopeWindowAuthority,
)


FETCHED_AT = datetime(2026, 7, 30, 12, 0, tzinfo=timezone.utc)
RECEIPT_TTL_SECONDS = 3_600
REQUEST_ID = "request-current"


class _HttpResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def json(self) -> dict:
        return self.payload


class _ScriptedSession:
    def __init__(self, *responses: _HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def post(self, url: str, *, json: dict, timeout: float) -> _HttpResponse:
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        return self.responses.pop(0)


def _evidence(
    *,
    forecast_date: date = date(2026, 8, 3),
    fetched_at: datetime = FETCHED_AT,
    resolved_location: str | None = "New York, New York, United States",
) -> WeatherReceiptEvidence:
    return WeatherReceiptEvidence(
        provider="visual_crossing",
        fetched_at=fetched_at,
        requested_window=WeatherReceiptWindow(
            start_date=forecast_date,
            end_date=forecast_date,
        ),
        resolved_location=resolved_location,
        days=[
            WeatherReceiptDay(
                date=forecast_date,
                condition="rain",
                precipitation_probability_pct=70.0,
                precipitation_types=["rain"],
                temperature_low_f=57.0,
                temperature_high_f=66.0,
            )
        ],
        attribution=WeatherReceiptAttribution(),
    )


def _receipt(
    *,
    source_turn_id: str = "turn-weather",
    source_tool_call_id: str = "weather-call",
    forecast_date: date = date(2026, 8, 3),
    fetched_at: datetime = FETCHED_AT,
    saved_area: bool = False,
) -> WeatherForecastReceipt:
    scope = (
        SavedAreaWeatherScope()
        if saved_area
        else ShopperLocationWeatherScope(
            location="NYC",
            location_query="NYC, NY",
        )
    )
    evidence = _evidence(
        forecast_date=forecast_date,
        fetched_at=fetched_at,
        resolved_location=(
            None if saved_area else "New York, New York, United States"
        ),
    )
    scope_key = weather_scope_key(scope, evidence)
    return WeatherForecastReceipt(
        receipt_id=weather_receipt_id(
            source_turn_id=source_turn_id,
            source_tool_call_id=source_tool_call_id,
            scope_key=scope_key,
            fetched_at=evidence.fetched_at,
        ),
        scope_key=scope_key,
        source_turn_id=source_turn_id,
        source_sequence=2,
        source_tool_call_id=source_tool_call_id,
        location_scope=scope,
        evidence=evidence,
        valid_until=evidence.fetched_at
        + timedelta(seconds=RECEIPT_TTL_SECONDS),
    )


def _promotion(*, expected_projection_version: int = 4) -> WeatherReceiptPromotion:
    return WeatherReceiptPromotion(
        expected_projection_version=expected_projection_version,
        source_tool_call_id="weather-call",
        location_scope=ShopperLocationWeatherScope(
            location="NYC",
            location_query="NYC, NY",
        ),
        evidence=_evidence(),
        ttl_seconds=RECEIPT_TTL_SECONDS,
    )


def _scope_resolution() -> CurrentWeatherScopeResolution:
    return CurrentWeatherScopeResolution(
        expected_projection_version=4,
        expected_scope_revision=0,
        location_action="set",
        window_action="set",
        location_scope=ShopperLocationWeatherScope(
            location="NYC",
            location_query="NYC, NY",
        ),
        requested_window=WeatherReceiptWindow(
            start_date=date(2026, 8, 3),
            end_date=date(2026, 8, 3),
        ),
    )


def _current_scope(
    receipt: WeatherForecastReceipt | None = None,
) -> CurrentWeatherScope:
    scoped_receipt = receipt or _receipt()
    return CurrentWeatherScope(
        revision=1,
        location=WeatherScopeLocationAuthority(
            value=scoped_receipt.location_scope,
            source_turn_id=scoped_receipt.source_turn_id,
            source_sequence=scoped_receipt.source_sequence,
        ),
        window=WeatherScopeWindowAuthority(
            value=scoped_receipt.evidence.requested_window,
            source_turn_id=scoped_receipt.source_turn_id,
            source_sequence=scoped_receipt.source_sequence,
        ),
    )


def _success_content(
    *,
    forecast_date: date = date(2026, 8, 3),
) -> str:
    return (
        f"{WEATHER_FORECAST_EVIDENCE_PREFIX} "
        + _evidence(forecast_date=forecast_date).model_dump_json()
    )


def _current_weather_result(
    *,
    content: str | None = None,
    include_call: bool = True,
    tool_call_id: str = "weather-call",
) -> dict:
    messages: list[dict] = [
        {"role": "user", "content": f"REQUEST ID: {REQUEST_ID}"},
    ]
    if include_call:
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "name": "get_weather_forecast_tool",
                        "args": {},
                    }
                ],
            }
        )
    if content is not None:
        messages.append(
            {
                "role": "tool",
                "name": "get_weather_forecast_tool",
                "tool_call_id": tool_call_id,
                "content": content,
            }
        )
    return {"messages": messages}


def _identity() -> runtime_mod.RequestIdentity:
    return runtime_mod.RequestIdentity(
        session_id="session",
        conversation_id="conversation",
        cart_id="cart",
        context_user_id=1,
        cart_user_id=1,
        request_id=REQUEST_ID,
    )


def _turn(
    *,
    projection: ConversationProjection | None = None,
) -> TurnStartResult:
    return TurnStartResult(
        turn_id="turn-current",
        contract_version=2,
        attempt_id="attempt-current",
        sequence=3,
        shopper_context=None,
        projection=projection or ConversationProjection(version=4),
        cart=[],
    )


def test_receipt_prompt_projection_exposes_scope_without_forecast_evidence() -> None:
    receipt = _receipt(
        source_turn_id="private-source-turn",
        source_tool_call_id="private-source-call",
        saved_area=True,
    )

    rendered = runtime_mod._format_active_weather_receipts([receipt])

    assert receipt.receipt_id in rendered
    assert '"receipt_type":"weather_forecast.v1"' in rendered
    assert '"kind":"confirmed_saved_zip"' in rendered
    assert '"requested_window":' in rendered
    assert '"valid_until":' in rendered
    assert '"evidence":' not in rendered
    assert '"days":' not in rendered
    assert '"condition":' not in rendered
    assert '"resolved_location":' not in rendered
    assert "Weather Data Provided by Visual Crossing" not in rendered
    assert receipt.scope_key not in rendered
    assert "private-source-turn" not in rendered
    assert "private-source-call" not in rendered
    assert "source_sequence" not in rendered
    assert "source_tool" not in rendered
    assert "98101" not in rendered


def test_main_model_message_never_exposes_unbound_forecast_facts(
    base_config,
) -> None:
    receipt = _receipt()
    runtime = runtime_mod.DeepAgentsRuntime(base_config)
    state = State(
        user_id=1,
        query="Compare the dresses for the same event.",
        active_weather_receipts=[receipt],
    )

    message = runtime._build_user_message(state, _identity())

    assert receipt.receipt_id in message
    assert '"requested_window":' in message
    assert "NYC" in message
    assert "New York, New York, United States" not in message
    assert '"condition":' not in message
    assert '"precipitation_probability_pct":' not in message
    assert '"temperature_low_f":' not in message
    assert '"temperature_high_f":' not in message
    assert "Weather Data Provided by Visual Crossing" not in message


def test_dynamic_activation_accepts_only_listed_exact_scope_receipt() -> None:
    receipt = _receipt()
    activation = runtime_mod._skill_activation_input_model(
        ("event-context", "outfit-styling"),
        (receipt.receipt_id,),
    )

    selected = activation(
        skill_names=["outfit-styling", "event-context"],
        event_context_next_question="none",
        weather_receipt_id=receipt.receipt_id,
    )

    assert selected.weather_receipt_id == receipt.receipt_id
    with pytest.raises(ValidationError):
        activation(
            skill_names=["outfit-styling", "event-context"],
            event_context_next_question="none",
            weather_receipt_id="sha256:" + ("0" * 64),
        )


def test_receipt_identity_is_redacted_from_normal_and_failed_turn_diagnostics() -> None:
    receipt = _receipt()
    messages = [
        {
            "role": "user",
            "content": f"REQUEST ID: {REQUEST_ID}",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "activation-call",
                    "name": "activate_shopper_skills_tool",
                    "args": {
                        "skill_names": [
                            "outfit-styling",
                            "event-context",
                        ],
                        "event_context_next_question": "none",
                        "weather_receipt_id": receipt.receipt_id,
                    },
                }
            ],
        },
        {
            "role": "tool",
            "name": "activate_shopper_skills_tool",
            "tool_call_id": "activation-call",
            "content": (
                "SHOPPER_SKILL_ACTIVATION_COMPLETE: "
                "/shopper/outfit-styling/SKILL.md, "
                "/shopper/event-context/SKILL.md"
            ),
        },
    ]

    diagnostics = runtime_mod._collect_agent_diagnostics(
        messages,
        request_id=REQUEST_ID,
        final_termination_reason="agent_error",
        preserve_partial_messages=True,
    )
    diagnostics["weather_receipt_status"] = "bound"

    assert diagnostics["tool_calls"][0]["arguments"] == {
        "skill_names": ["outfit-styling", "event-context"],
        "event_context_next_question": "none",
    }
    serialized = json.dumps(diagnostics, sort_keys=True)
    assert receipt.receipt_id not in serialized
    assert "weather_receipt_id" not in serialized
    assert '"weather_receipt_status": "bound"' in serialized


def test_unselected_receipt_binding_is_inert() -> None:
    receipt = _receipt()
    activation = runtime_mod._skill_activation_input_model(
        ("event-context", "outfit-styling"),
        (receipt.receipt_id,),
    )

    accepted = activation(
        skill_names=["outfit-styling"],
        weather_receipt_id=receipt.receipt_id,
    )

    assert accepted.weather_receipt_id is None


def test_selected_receipt_binding_requires_no_question() -> None:
    receipt = _receipt()
    activation = runtime_mod._skill_activation_input_model(
        ("event-context", "outfit-styling"),
        (receipt.receipt_id,),
    )

    with pytest.raises(
        ValidationError,
        match="weather receipt binding requires event-context with no",
    ):
        activation(
            skill_names=["outfit-styling", "event-context"],
            event_context_next_question="event_date",
            weather_receipt_id=receipt.receipt_id,
        )


def test_paired_current_success_prepares_typed_promotion() -> None:
    state = State(
        user_id=1,
        query="Compare them for the NYC wedding.",
        conversation_projection_version=4,
        current_weather_scope=_current_scope(),
    )

    promotion = runtime_mod._current_weather_receipt_promotion(
        state,
        _current_weather_result(content=_success_content()),
        request_id=REQUEST_ID,
        ttl_seconds=RECEIPT_TTL_SECONDS,
    )

    assert promotion is not None
    assert promotion.expected_projection_version == 4
    assert promotion.source_tool_call_id == "weather-call"
    assert promotion.location_scope == ShopperLocationWeatherScope(
        location="NYC",
        location_query="NYC, NY",
    )
    assert promotion.evidence.requested_window.start_date == date(2026, 8, 3)
    assert promotion.ttl_seconds == RECEIPT_TTL_SECONDS


def test_failures_unpaired_results_and_prior_turn_results_never_promote() -> None:
    state = State(
        user_id=1,
        query="Continue.",
        conversation_projection_version=4,
    )
    failure_content = (
        "WEATHER_FORECAST_FAILURE: "
        '{"ok":false,"code":"weather_unavailable","retryable":true}'
    )
    failure = _current_weather_result(content=failure_content)
    unpaired = _current_weather_result(
        content=_success_content(),
        include_call=False,
    )
    prior = _current_weather_result(content=None)
    prior["messages"] = [
        {
            "role": "user",
            "content": "REQUEST ID: request-prior",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "weather-prior",
                    "name": "get_weather_forecast_tool",
                    "args": {},
                }
            ],
        },
        {
            "role": "tool",
            "name": "get_weather_forecast_tool",
            "tool_call_id": "weather-prior",
            "content": _success_content(),
        },
        {
            "role": "user",
            "content": f"REQUEST ID: {REQUEST_ID}",
        },
    ]

    for result in (failure, unpaired, prior):
        assert (
            runtime_mod._current_weather_receipt_promotion(
                state,
                result,
                request_id=REQUEST_ID,
                ttl_seconds=RECEIPT_TTL_SECONDS,
            )
            is None
        )


def test_current_weather_outcome_wins_over_selected_receipt() -> None:
    receipt = _receipt()
    state = State(
        user_id=1,
        query="Refresh it.",
        active_weather_receipts=[receipt],
        selected_weather_receipt_id=receipt.receipt_id,
    )
    result = _current_weather_result(
        content=(
            "WEATHER_FORECAST_FAILURE: "
            '{"ok":false,"code":"weather_unavailable","retryable":true}'
        ),
    )

    outcome, reused_receipt = runtime_mod._effective_weather_outcome(
        state,
        result,
        request_id=REQUEST_ID,
    )

    assert isinstance(outcome, WeatherFailure)
    assert outcome.code == "weather_unavailable"
    assert reused_receipt is False


def test_only_explicitly_selected_receipt_becomes_effective() -> None:
    first = _receipt(
        source_turn_id="turn-first",
        source_tool_call_id="call-first",
        forecast_date=date(2026, 8, 3),
    )
    second = _receipt(
        source_turn_id="turn-second",
        source_tool_call_id="call-second",
        forecast_date=date(2026, 8, 4),
    )
    state = State(
        user_id=1,
        query="Compare them for the event.",
        active_weather_receipts=[first, second],
        selected_weather_receipt_id=second.receipt_id,
    )

    outcome, reused_receipt = runtime_mod._effective_weather_outcome(
        state,
        {"messages": []},
        request_id=REQUEST_ID,
    )

    assert outcome is not None
    assert outcome.requested_window.start_date == date(2026, 8, 4)
    assert reused_receipt is True


@pytest.mark.asyncio
async def test_bound_receipt_guides_comparison_without_repeating_forecast(
    base_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = runtime_mod.DeepAgentsRuntime(base_config)
    receipt = _receipt()
    captured: dict[str, object] = {}

    class ComparisonEditor:
        async def ainvoke(self, messages):
            captured["messages"] = messages
            return AIMessage(
                content=(
                    "Intricate Lace Gown is the more formal styling choice; "
                    "Wavy Hem Satin Dress is the lower-price option.\n\n"
                    "Live forecast for Aug 3, 2026: rain, 57–66°F, with a 70% "
                    "precipitation chance. Weather Data Provided by Visual "
                    "Crossing."
                )
            )

    monkeypatch.setattr(
        runtime,
        "_create_chat_model",
        lambda: ComparisonEditor(),
    )
    state = State(
        user_id=1,
        query="Compare the lacy gown and hem satin dress for the event.",
        active_weather_receipts=[receipt],
        selected_weather_receipt_id=receipt.receipt_id,
        agent_diagnostics={
            "skill_files_read": [
                "/shopper/outfit-styling/SKILL.md",
                "/shopper/event-context/SKILL.md",
            ]
        },
    )
    result = {
        "messages": [
            {"role": "user", "content": f"REQUEST ID: {REQUEST_ID}"},
            {
                "role": "tool",
                "name": "resolve_conversation_products_tool",
                "content": (
                    "Resolved Intricate Lace Gown and Wavy Hem Satin Dress."
                ),
            },
            {
                "role": "tool",
                "name": "get_product_details_tool",
                "content": (
                    f"{runtime_mod._PRODUCT_DETAIL_GROUNDING_NOTE}\n"
                    "PRODUCT_REF: dress-1\n"
                    "NAME: Intricate Lace Gown\n"
                    "PRICE: $139.99 USD"
                ),
            },
            {
                "role": "tool",
                "name": "get_product_details_tool",
                "content": (
                    f"{runtime_mod._PRODUCT_DETAIL_GROUNDING_NOTE}\n"
                    "PRODUCT_REF: dress-2\n"
                    "NAME: Wavy Hem Satin Dress\n"
                    "PRICE: $89.99 USD"
                ),
            },
        ]
    }

    response = await runtime._rewrite_response_for_grounding(
        state,
        result,
        "Compare the two dresses for the event.",
        request_id=REQUEST_ID,
    )

    assert "Intricate Lace Gown" in response
    assert "Wavy Hem Satin Dress" in response
    assert "Live forecast" not in response
    assert "Aug 3, 2026" not in response
    assert "57–66°F" not in response
    assert "Weather Data Provided by Visual Crossing" not in response
    editor_prompt = captured["messages"][1]["content"]
    assert "SERVER-BOUND DURABLE WEATHER STYLING DIRECTION" in editor_prompt
    assert "New York, New York, United States" not in editor_prompt
    assert "2026-08-03" not in editor_prompt
    assert receipt.receipt_id not in editor_prompt


@pytest.mark.parametrize(
    "selected_receipt_id",
    [None, "sha256:" + ("0" * 64)],
)
def test_unbound_or_unknown_receipt_provides_no_weather_evidence(
    selected_receipt_id: str | None,
) -> None:
    receipt = _receipt()
    state = State(
        user_id=1,
        query="Compare them.",
        active_weather_receipts=[receipt],
        selected_weather_receipt_id=selected_receipt_id,
    )

    assert runtime_mod._effective_weather_outcome(
        state,
        {"messages": []},
        request_id=REQUEST_ID,
    ) == (None, False)


def test_bound_hydrated_receipt_uses_turn_start_validity_snapshot() -> None:
    receipt = _receipt(
        fetched_at=datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc),
    )
    state = State(
        user_id=1,
        query="Compare them.",
        active_weather_receipts=[receipt],
        selected_weather_receipt_id=receipt.receipt_id,
    )

    assert runtime_mod._bound_weather_receipt(state) == receipt
    assert runtime_mod._effective_weather_outcome(
        state,
        {"messages": []},
        request_id=REQUEST_ID,
    ) == (runtime_mod._weather_forecast_from_receipt(receipt), True)


def test_completed_finalize_carries_promotion_but_failed_finalize_drops_it(
    base_config,
) -> None:
    runtime = runtime_mod.DeepAgentsRuntime(base_config)

    class Memory:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def finalize_turn(self, *_args, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(replayed=False)

    memory = Memory()
    runtime._conversation_memory = memory
    promotion = _promotion()

    completed = State(
        user_id=1,
        query="Continue.",
        response="Completed response.",
        weather_receipt_promotion=promotion,
        agent_diagnostics={"final_termination_reason": "completed"},
    )
    failed = State(
        user_id=1,
        query="Continue.",
        response="Timed out.",
        weather_receipt_promotion=promotion,
        agent_diagnostics={"final_termination_reason": "agent_timeout"},
    )

    assert runtime._finalize_conversation_turn(
        completed,
        _identity(),
        _turn(),
    )
    assert runtime._finalize_conversation_turn(
        failed,
        _identity(),
        _turn(),
        status="failed",
        termination_reason="agent_timeout",
    )
    assert memory.calls[0]["weather_receipt_promotion"] == promotion
    assert memory.calls[1]["weather_receipt_promotion"] is None


def test_v3_completed_finalize_carries_current_weather_scope_transition(
    base_config,
) -> None:
    runtime = runtime_mod.DeepAgentsRuntime(base_config)

    class Memory:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def finalize_turn(self, *_args, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(replayed=False)

    memory = Memory()
    runtime._conversation_memory = memory
    transition = CurrentWeatherScopeTransition(
        expected_projection_version=4,
        action="replace",
        location_scope=ShopperLocationWeatherScope(location="Seattle"),
    )
    completed = State(
        user_id=1,
        query="A different wedding in Seattle.",
        response="What date should I plan around?",
        current_weather_scope_transition=transition,
        agent_diagnostics={"final_termination_reason": "completed"},
    )
    failed = completed.model_copy(
        update={
            "response": "Timed out.",
            "agent_diagnostics": {
                "final_termination_reason": "agent_timeout"
            },
        }
    )
    turn = _turn().model_copy(update={"contract_version": 3})

    assert runtime._finalize_conversation_turn(
        completed,
        _identity(),
        turn,
    )
    assert runtime._finalize_conversation_turn(
        failed,
        _identity(),
        turn,
        status="failed",
        termination_reason="agent_timeout",
    )

    assert memory.calls[0]["current_weather_scope_transition"] == transition
    assert memory.calls[1]["current_weather_scope_transition"] is None


def test_v1_turn_drops_optional_weather_promotion_before_finalize(
    base_config,
) -> None:
    runtime = runtime_mod.DeepAgentsRuntime(base_config)

    class Memory:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def finalize_turn(self, *_args, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(replayed=False)

    memory = Memory()
    runtime._conversation_memory = memory
    state = State(
        user_id=1,
        query="Continue.",
        response="Completed response.",
        weather_receipt_promotion=_promotion(),
        agent_diagnostics={
            "final_termination_reason": "completed",
            "weather_receipt_status": "promotion_prepared",
        },
    )
    turn = _turn().model_copy(update={"contract_version": 1})

    assert runtime._finalize_conversation_turn(
        state,
        _identity(),
        turn,
    )
    assert memory.calls[0]["weather_receipt_promotion"] is None
    assert state.agent_diagnostics["weather_receipt_status"] == (
        "promotion_dropped_contract_v1"
    )


def test_projection_conflict_retries_finalize_without_optional_promotion(
    base_config,
) -> None:
    runtime = runtime_mod.DeepAgentsRuntime(base_config)

    class ConflictThenSuccessMemory:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def finalize_turn(self, *_args, **kwargs):
            self.calls.append(kwargs)
            if kwargs["weather_receipt_promotion"] is not None:
                raise ConversationMemoryError(
                    "projection_version_conflict",
                    "projection changed",
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
        weather_receipt_promotion=_promotion(),
        agent_diagnostics={
            "final_termination_reason": "completed",
            "weather_receipt_status": "promotion_prepared",
        },
    )

    assert runtime._finalize_conversation_turn(
        state,
        _identity(),
        _turn(),
    )
    assert len(memory.calls) == 2
    assert memory.calls[0]["weather_receipt_promotion"] is not None
    assert memory.calls[1]["weather_receipt_promotion"] is None
    assert state.agent_diagnostics["weather_receipt_status"] == (
        "promotion_dropped"
    )
    assert "memory_finalize_error" not in state.agent_diagnostics


@pytest.mark.parametrize(
    "detail",
    [
        "weather_receipt_stale",
        "weather_receipt_status_conflict",
        "weather_receipt_scope_conflict",
    ],
)
def test_http_receipt_conflict_retries_without_promotion_but_keeps_scope(
    base_config,
    detail: str,
) -> None:
    runtime = runtime_mod.DeepAgentsRuntime(base_config)
    session = _ScriptedSession(
        _HttpResponse({"detail": detail}, status_code=409),
        _HttpResponse(
            {
                "turn_id": "turn-current",
                "attempt_id": "attempt-current",
                "sequence": 3,
                "replayed": False,
                "status": "completed",
                "assistant_text": "Completed response.",
                "termination_reason": "completed",
            }
        ),
    )
    runtime._conversation_memory = ConversationMemoryClient(
        "http://memory",
        session=session,
    )
    state = State(
        user_id=1,
        query="Continue.",
        response="Completed response.",
        weather_receipt_promotion=_promotion(),
        current_weather_scope_resolution=_scope_resolution(),
        agent_diagnostics={
            "final_termination_reason": "completed",
            "weather_receipt_status": "promotion_prepared",
        },
    )

    assert runtime._finalize_conversation_turn(
        state,
        _identity(),
        _turn().model_copy(update={"contract_version": 4}),
    )

    assert len(session.calls) == 2
    first = session.calls[0]["json"]
    second = session.calls[1]["json"]
    assert "weather_receipt_promotion" in first
    assert "weather_receipt_promotion" not in second
    assert "current_weather_scope_resolution" in first
    assert (
        second["current_weather_scope_resolution"]
        == first["current_weather_scope_resolution"]
    )
    assert state.agent_diagnostics["weather_receipt_status"] == (
        "promotion_dropped"
    )


@pytest.mark.parametrize(
    "detail",
    [
        "projection_version_conflict",
        "current_weather_scope_revision_conflict",
        "current_weather_scope_resolution_conflict",
        "current_weather_scope_status_conflict",
        "current_weather_scope_saved_area_unavailable",
    ],
)
def test_http_scope_conflict_terminalizes_failed_without_scope_or_products(
    base_config,
    detail: str,
) -> None:
    runtime = runtime_mod.DeepAgentsRuntime(base_config)
    session = _ScriptedSession(
        _HttpResponse({"detail": detail}, status_code=409),
        _HttpResponse(
            {
                "turn_id": "turn-current",
                "attempt_id": "attempt-current",
                "sequence": 3,
                "replayed": False,
                "status": "failed",
                "assistant_text": runtime_mod._GROUNDING_FAILURE_RESPONSE,
                "termination_reason": detail,
            }
        ),
    )
    runtime._conversation_memory = ConversationMemoryClient(
        "http://memory",
        session=session,
    )
    state = State(
        user_id=1,
        query="Continue.",
        response="Unpersisted weather response.",
        product_results=[
            {
                "product_id": "stale-product",
                "display_name": "Stale product",
            }
        ],
        retrieved={"Stale product": "/images/stale.png"},
        current_weather_scope_resolution=_scope_resolution(),
        agent_diagnostics={"final_termination_reason": "completed"},
    )

    assert runtime._finalize_conversation_turn(
        state,
        _identity(),
        _turn().model_copy(update={"contract_version": 4}),
    )

    assert len(session.calls) == 2
    first = session.calls[0]["json"]
    failed = session.calls[1]["json"]
    assert first["status"] == "completed"
    assert "current_weather_scope_resolution" in first
    assert failed["status"] == "failed"
    assert failed["termination_reason"] == detail
    assert "current_weather_scope_resolution" not in failed
    assert "weather_receipt_promotion" not in failed
    assert failed["output"]["product_results"] == []
    assert failed["output"]["retrieved"] == {}
    assert state.response == runtime_mod._GROUNDING_FAILURE_RESPONSE
    assert state.product_results == []
    assert state.retrieved == {}
    assert state.agent_diagnostics["memory_finalize_error"] == detail


def test_hydration_keeps_receipts_out_of_summary_raw_and_product_lanes(
    base_config,
) -> None:
    runtime = runtime_mod.DeepAgentsRuntime(base_config)
    receipt = _receipt()
    projection = ConversationProjection(
        version=4,
        summary_text="The shopper is planning a wedding outfit.",
        summary_through_sequence=1,
        product_reference_index=[
            {
                "candidate_set_id": "set-1",
                "turn_seq": 1,
                "products": [
                    {
                        "ref": "dress-1",
                        "name": "Satin Dress",
                        "category": "dresses",
                        "position": 1,
                    }
                ],
            }
        ],
        active_receipts=[receipt],
        current_weather_scope=CurrentWeatherScope(
            revision=1,
            location=WeatherScopeLocationAuthority(
                value=receipt.location_scope,
                source_turn_id=receipt.source_turn_id,
                source_sequence=receipt.source_sequence,
            ),
            window=WeatherScopeWindowAuthority(
                value=receipt.evidence.requested_window,
                source_turn_id=receipt.source_turn_id,
                source_sequence=receipt.source_sequence,
            ),
        ),
    )
    turn = _turn(projection=projection).model_copy(
        update={
            "contract_version": 3,
            "recent_turns": [
                RecentConversationTurn(
                    sequence=2,
                    shopper_text="Compare those dresses.",
                    assistant_text="I’ll compare the established options.",
                    status="completed",
                )
            ],
            "unsummarized_turn_count": 1,
        }
    )

    class Memory:
        def start_turn(self, *_args, **_kwargs):
            return turn

    runtime._conversation_memory = Memory()
    state = State(user_id=1, query="Which works better for the event?")

    assert runtime._start_conversation_turn(state, _identity()) is turn
    assert state.conversation_projection_version == 4
    assert state.active_weather_receipts == [receipt]
    assert state.current_weather_scope == projection.current_weather_scope
    assert receipt.receipt_id not in state.conversation_summary
    assert receipt.receipt_id not in state.context
    assert receipt.receipt_id not in state.historical_product_context
    assert "Satin Dress" in state.historical_product_context

    prompt = runtime._build_user_message(state, _identity())
    assert "DURABLE CONVERSATION SUMMARY" in prompt
    assert "RECENT DISCUSSION" in prompt
    assert "HISTORICAL PRODUCT INDEX" in prompt
    assert "VALID DURABLE WEATHER RECEIPTS" in prompt
    assert prompt.count(receipt.receipt_id) == 1
    assert json.dumps(receipt.model_dump(mode="json")) not in prompt
