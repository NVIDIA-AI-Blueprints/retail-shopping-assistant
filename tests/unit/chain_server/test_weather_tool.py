# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for direct and event-scoped weather wrappers."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
from typing import Any

import pytest

from chain_server.src.tool_policy import SHOPPING_TOOL_POLICIES
from chain_server.src.weather import (
    VISUAL_CROSSING_ATTRIBUTION_LABEL,
    VISUAL_CROSSING_ATTRIBUTION_URL,
    WeatherAttribution,
    WeatherDay,
    WeatherRequest,
    WeatherRequestedWindow,
    WeatherResult,
    weather_failure,
)
from chain_server.src.weather_tool import (
    WEATHER_FORECAST_EVIDENCE_PREFIX,
    WEATHER_FORECAST_FAILURE_PREFIX,
    WeatherScopeForecastBinding,
    WeatherScopeSelection,
    WeatherForecastEvidence,
    compile_weather_scope_resolution,
    get_event_weather_forecast_tool,
    get_scoped_weather_forecast_tool,
    get_weather_forecast_tool,
    parse_weather_tool_evidence,
    weather_date_context_available,
)
from shared.weather_receipts import (
    ShopperLocationWeatherScope,
    WeatherReceiptWindow,
)
from shared.weather_scope import (
    CurrentWeatherScope,
    WeatherScopeLocationAuthority,
    WeatherScopeUnavailableAuthority,
    WeatherScopeWindowAuthority,
    apply_current_weather_scope_resolution,
    effective_resolved_weather_scope_unavailability,
    effective_resolved_weather_scope_values,
)


FORECAST_DATE = date(2026, 7, 29)
SAVED_ZIPCODE = "98101"
EXPLICIT_LOCATION = "NYC"
CURRENT_DATE = date(2026, 7, 28)
SENSITIVE_EXCEPTION = "provider secret at Seattle 98101"


class RecordingClient:
    def __init__(
        self,
        outcome: Any | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.requests: list[WeatherRequest] = []
        self.outcome = outcome or weather_failure("weather_disabled")
        self.error = error

    def get_forecast(self, request: WeatherRequest):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.outcome


def _success_result(
    *,
    resolved_location: str = "Seattle, Washington, United States",
    start_date: date = FORECAST_DATE,
    end_date: date = FORECAST_DATE,
) -> WeatherResult:
    days = [
        start_date + timedelta(days=offset)
        for offset in range((end_date - start_date).days + 1)
    ]
    return WeatherResult(
        fetched_at=datetime(2026, 7, 28, 18, 0, tzinfo=timezone.utc),
        requested_window=WeatherRequestedWindow(
            start_date=start_date,
            end_date=end_date,
        ),
        resolved_location=resolved_location,
        timezone="America/Los_Angeles",
        days=[
            WeatherDay(
                date=day,
                condition="rain",
                precipitation_probability_pct=70,
                precipitation_types=["rain"],
                temperature_low_f=57,
                temperature_high_f=66,
            )
            for day in days
        ],
        attribution=WeatherAttribution(
            label=VISUAL_CROSSING_ATTRIBUTION_LABEL,
            url=VISUAL_CROSSING_ATTRIBUTION_URL,
        ),
    )


def _event_tool(
    client: RecordingClient,
    *,
    saved_zipcode: str | None = SAVED_ZIPCODE,
    saved_zip_authorized: bool = True,
    shopper_provided_texts: set[str] | None = None,
    current_date: date = CURRENT_DATE,
):
    return get_event_weather_forecast_tool(
        client,
        saved_zipcode=saved_zipcode,
        saved_zip_authorized=saved_zip_authorized,
        shopper_provided_texts=(
            {"NYC, on an outdoor patio next week."}
            if shopper_provided_texts is None
            else shopper_provided_texts
        ),
        current_date=current_date,
    )


def _existing_nyc_scope() -> CurrentWeatherScope:
    return CurrentWeatherScope(
        revision=1,
        location=WeatherScopeLocationAuthority(
            value=ShopperLocationWeatherScope(
                location="NYC",
                location_query="NYC, NY",
            ),
            source_turn_id="turn-nyc",
            source_sequence=1,
        ),
        window=WeatherScopeWindowAuthority(
            value=WeatherReceiptWindow(
                start_date=date(2026, 8, 7),
                end_date=date(2026, 8, 7),
            ),
            source_turn_id="turn-nyc",
            source_sequence=1,
        ),
    )


def _existing_nyc_location_only_scope() -> CurrentWeatherScope:
    return CurrentWeatherScope(
        revision=1,
        location=WeatherScopeLocationAuthority(
            value=ShopperLocationWeatherScope(
                location="NYC",
                location_query="NYC, NY",
            ),
            source_turn_id="turn-nyc",
            source_sequence=1,
        ),
    )


def test_new_subject_sets_location_and_explicitly_clears_prior_window() -> None:
    current_scope = _existing_nyc_scope()
    resolution = compile_weather_scope_resolution(
        WeatherScopeSelection(
            scope_revision=1,
            location_action="set",
            window_action="clear",
            location_source="shopper_provided_location",
            location="Seattle",
        ),
        current_scope=current_scope,
        current_shopper_text="Help with a different wedding in Seattle.",
        saved_zip_authorized=False,
        expected_projection_version=4,
        current_date=CURRENT_DATE,
    )

    location, window = effective_resolved_weather_scope_values(
        current_scope,
        resolution,
    )

    assert resolution.location_action == "set"
    assert resolution.window_action == "clear"
    assert location == ShopperLocationWeatherScope(location="Seattle")
    assert window is None


@pytest.mark.parametrize(
    ("shopper_location", "location_query"),
    [
        ("Seattle", None),
        ("NYC", "NYC, NY"),
    ],
)
def test_atomic_location_set_can_explicitly_clear_prior_window(
    shopper_location: str,
    location_query: str | None,
) -> None:
    current_scope = _existing_nyc_scope()
    resolution = compile_weather_scope_resolution(
        WeatherScopeSelection(
            scope_revision=1,
            location_action="set",
            window_action="clear",
            location_source="shopper_provided_location",
            location=shopper_location,
            location_query=location_query,
        ),
        current_scope=current_scope,
        current_shopper_text=f"The same wedding moved to {shopper_location}.",
        saved_zip_authorized=False,
        expected_projection_version=4,
        current_date=CURRENT_DATE,
    )

    effective_location, window = effective_resolved_weather_scope_values(
        current_scope,
        resolution,
    )

    assert effective_location == ShopperLocationWeatherScope(
        location=shopper_location,
        location_query=location_query,
    )
    assert window is None
    persisted = apply_current_weather_scope_resolution(
        current_scope,
        resolution,
        source_turn_id="turn-seattle",
        source_sequence=2,
    )
    assert persisted.location is not None
    assert persisted.location.value == effective_location
    assert persisted.location.source_turn_id == "turn-seattle"
    assert persisted.window is None


def test_same_pending_question_preserves_its_durable_source_binding() -> None:
    current_scope = CurrentWeatherScope.model_validate(
        {
            "revision": 2,
            "pending_question": "event_location",
            "pending_source_turn_id": "conference-turn",
            "pending_source_sequence": 2,
            "window": {
                "value": {
                    "start_date": "2026-08-15",
                    "end_date": "2026-08-15",
                },
                "source_turn_id": "conference-turn",
                "source_sequence": 2,
            },
        }
    )
    resolution = compile_weather_scope_resolution(
        WeatherScopeSelection(
            scope_revision=2,
            location_action="clear",
            window_action="set",
            date="2026-08-16",
        ),
        current_scope=current_scope,
        current_shopper_text="The same conference is August 16.",
        saved_zip_authorized=False,
        expected_projection_version=5,
        current_date=CURRENT_DATE,
    ).model_copy(
        update={
            "pending_question": "event_location",
            "preserve_pending_source_turn_id": "conference-turn",
        }
    )

    persisted = apply_current_weather_scope_resolution(
        current_scope,
        resolution,
        source_turn_id="date-correction-turn",
        source_sequence=3,
    )

    assert persisted.pending_question == "event_location"
    assert persisted.pending_source_turn_id == "conference-turn"
    assert persisted.pending_source_sequence == 2
    assert persisted.window is not None
    assert persisted.window.source_turn_id == "date-correction-turn"


def test_same_pending_question_without_preserve_handle_stamps_new_source() -> None:
    current_scope = CurrentWeatherScope.model_validate(
        {
            "revision": 2,
            "pending_question": "event_location",
            "pending_source_turn_id": "older-subject-turn",
            "pending_source_sequence": 2,
            "window": {
                "value": {
                    "start_date": "2026-08-15",
                    "end_date": "2026-08-15",
                },
                "source_turn_id": "older-subject-turn",
                "source_sequence": 2,
            },
        }
    )
    resolution = compile_weather_scope_resolution(
        WeatherScopeSelection(
            scope_revision=2,
            location_action="clear",
            window_action="set",
            date="2026-08-22",
        ),
        current_scope=current_scope,
        current_shopper_text="The new conference is August 22.",
        saved_zip_authorized=False,
        expected_projection_version=5,
        current_date=CURRENT_DATE,
    ).model_copy(update={"pending_question": "event_location"})

    persisted = apply_current_weather_scope_resolution(
        current_scope,
        resolution,
        source_turn_id="new-subject-turn",
        source_sequence=3,
    )

    assert persisted.pending_question == "event_location"
    assert persisted.pending_source_turn_id == "new-subject-turn"
    assert persisted.pending_source_sequence == 3


def test_wrong_pending_preserve_handle_fails_shared_validation() -> None:
    current_scope = CurrentWeatherScope.model_validate(
        {
            "revision": 2,
            "pending_question": "event_location",
            "pending_source_turn_id": "bound-turn",
            "pending_source_sequence": 2,
            "window": {
                "value": {
                    "start_date": "2026-08-15",
                    "end_date": "2026-08-15",
                },
                "source_turn_id": "bound-turn",
                "source_sequence": 2,
            },
        }
    )
    resolution = compile_weather_scope_resolution(
        WeatherScopeSelection(
            scope_revision=2,
            location_action="clear",
            window_action="set",
            date="2026-08-16",
        ),
        current_scope=current_scope,
        current_shopper_text="The same conference is August 16.",
        saved_zip_authorized=False,
        expected_projection_version=5,
        current_date=CURRENT_DATE,
    ).model_copy(
        update={
            "pending_question": "event_location",
            "preserve_pending_source_turn_id": "wrong-turn",
        }
    )

    with pytest.raises(
        ValueError,
        match="pending source preservation conflict",
    ):
        apply_current_weather_scope_resolution(
            current_scope,
            resolution,
            source_turn_id="date-correction-turn",
            source_sequence=3,
        )


def test_cross_subject_date_resolution_clears_nyc_and_cannot_call_weather() -> None:
    current_scope = _existing_nyc_location_only_scope()
    resolution = compile_weather_scope_resolution(
        WeatherScopeSelection(
            scope_revision=1,
            location_action="clear",
            window_action="set",
            date="2026-08-15",
        ),
        current_scope=current_scope,
        current_shopper_text="I also have a conference on August 15.",
        saved_zip_authorized=False,
        expected_projection_version=4,
        current_date=CURRENT_DATE,
    )
    location, window = effective_resolved_weather_scope_values(
        current_scope,
        resolution,
    )

    assert resolution.location_action == "clear"
    assert resolution.window_action == "set"
    assert location is None
    assert window == WeatherReceiptWindow(
        start_date=date(2026, 8, 15),
        end_date=date(2026, 8, 15),
    )

    client = RecordingClient()
    binding = WeatherScopeForecastBinding(
        current_scope,
        saved_zipcode=None,
    )
    binding.bind(resolution, relative_date=None, weekday=None)
    result = parse_weather_tool_evidence(
        get_scoped_weather_forecast_tool(client, binding).invoke({})
    )

    assert result == weather_failure("weather_request_invalid")
    assert client.requests == []


def test_bare_pending_same_subject_date_retains_nyc_and_sets_window() -> None:
    current_scope = _existing_nyc_location_only_scope()
    resolution = compile_weather_scope_resolution(
        WeatherScopeSelection(
            scope_revision=1,
            location_action="retain",
            window_action="set",
            relative_date="next_week",
        ),
        current_scope=current_scope,
        current_shopper_text="It will be next week.",
        saved_zip_authorized=False,
        expected_projection_version=4,
        current_date=CURRENT_DATE,
    )

    location, window = effective_resolved_weather_scope_values(
        current_scope,
        resolution,
    )

    assert resolution.location_action == "retain"
    assert resolution.window_action == "set"
    assert location == ShopperLocationWeatherScope(
        location="NYC",
        location_query="NYC, NY",
    )
    assert window == WeatherReceiptWindow(
        start_date=date(2026, 8, 3),
        end_date=date(2026, 8, 9),
    )


def test_stale_weather_scope_revision_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="weather scope revision does not match current scope",
    ):
        compile_weather_scope_resolution(
            WeatherScopeSelection(
                scope_revision=0,
                location_action="retain",
                window_action="set",
                relative_date="next_week",
            ),
            current_scope=_existing_nyc_location_only_scope(),
            current_shopper_text="It will be next week.",
            saved_zip_authorized=False,
            expected_projection_version=4,
            current_date=CURRENT_DATE,
        )


@pytest.mark.parametrize(
    "payload",
    [
        {
            "scope_revision": 1,
            "location_action": "retain",
            "window_action": "clear",
            "location_source": "shopper_provided_location",
            "location": "Seattle",
        },
        {
            "scope_revision": 1,
            "location_action": "clear",
            "window_action": "set",
        },
        {
            "scope_revision": 1,
            "location_action": "clear",
            "window_action": "retain",
            "date": "2026-08-15",
        },
    ],
)
def test_scope_selection_actions_own_their_component_payloads(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        WeatherScopeSelection.model_validate(payload)


def test_initial_scope_cannot_retain_a_missing_component() -> None:
    with pytest.raises(
        ValueError,
        match="weather scope cannot retain a missing location",
    ):
        compile_weather_scope_resolution(
            WeatherScopeSelection(
                scope_revision=0,
                location_action="retain",
                window_action="set",
                relative_date="next_week",
            ),
            current_scope=CurrentWeatherScope(),
            current_shopper_text="It will be next week.",
            saved_zip_authorized=False,
            expected_projection_version=0,
            current_date=CURRENT_DATE,
        )


def test_scope_selection_compiles_model_owned_unavailability_without_text_rules(
) -> None:
    current_scope = _existing_nyc_location_only_scope()
    resolution = compile_weather_scope_resolution(
        WeatherScopeSelection(
            scope_revision=1,
            location_action="retain",
            window_action="unavailable",
        ),
        current_scope=current_scope,
        current_shopper_text="Continue with the same trip.",
        saved_zip_authorized=False,
        expected_projection_version=4,
        current_date=CURRENT_DATE,
    )

    assert resolution.location_action == "retain"
    assert resolution.window_action == "unavailable"
    assert resolution.requested_window is None
    assert effective_resolved_weather_scope_values(
        current_scope,
        resolution,
    ) == (
        ShopperLocationWeatherScope(
            location="NYC",
            location_query="NYC, NY",
        ),
        None,
    )
    assert effective_resolved_weather_scope_unavailability(
        current_scope,
        resolution,
    ) == (False, True)


def test_scope_selection_can_retain_a_source_bound_unavailable_component() -> None:
    current_scope = CurrentWeatherScope(
        revision=2,
        location=WeatherScopeLocationAuthority(
            value=ShopperLocationWeatherScope(location="Seattle"),
            source_turn_id="location-turn",
            source_sequence=1,
        ),
        window_unavailable=WeatherScopeUnavailableAuthority(
            source_turn_id="date-unavailable-turn",
            source_sequence=2,
        ),
    )
    resolution = compile_weather_scope_resolution(
        WeatherScopeSelection(
            scope_revision=2,
            location_action="set",
            window_action="retain",
            location_source="shopper_provided_location",
            location="Portland",
        ),
        current_scope=current_scope,
        current_shopper_text="The same trip moved to Portland.",
        saved_zip_authorized=False,
        expected_projection_version=5,
        current_date=CURRENT_DATE,
    )

    persisted = apply_current_weather_scope_resolution(
        current_scope,
        resolution,
        source_turn_id="location-update-turn",
        source_sequence=3,
    )

    assert persisted.location is not None
    assert persisted.location.value == ShopperLocationWeatherScope(
        location="Portland"
    )
    assert persisted.window is None
    assert persisted.window_unavailable == current_scope.window_unavailable


def test_scope_selection_schema_explains_atomic_component_resolution() -> None:
    properties = WeatherScopeSelection.model_json_schema()["properties"]

    assert "Exact revision from CURRENT WEATHER SCOPE" in properties[
        "scope_revision"
    ]["description"]
    assert "Never rely on an omitted field to inherit location" in properties[
        "location_action"
    ]["description"]
    assert "Never rely on an omitted field to inherit a date" in properties[
        "window_action"
    ]["description"]
    assert "unavailable" in properties["location_action"]["enum"]
    assert "unavailable" in properties["window_action"]["enum"]
    assert "ordinary missing, askable location" in properties[
        "location_action"
    ]["description"]
    assert "ordinary missing, askable date" in properties[
        "window_action"
    ]["description"]
    assert "pending_source_turn_id" not in properties
    assert "shortest exact current-turn shopper phrase" in properties[
        "location"
    ]["description"]
    assert "location='NYC', location_query='NYC, NY'" in properties[
        "location_query"
    ]["description"]
    assert "full Monday-Sunday window" in properties["relative_date"][
        "description"
    ]
    assert "exactly equals the normalized current-turn" in properties["date"][
        "description"
    ]


@pytest.mark.parametrize(
    ("selection", "shopper_text", "expected_window"),
    [
        (
            WeatherScopeSelection(
                scope_revision=0,
                location_action="clear",
                window_action="set",
                date="2026-08-07",
            ),
            "Plan for 2026-08-07.",
            (date(2026, 8, 7), date(2026, 8, 7)),
        ),
        (
            WeatherScopeSelection(
                scope_revision=0,
                location_action="clear",
                window_action="set",
                start_date="2026-08-07",
                end_date="2026-08-09",
            ),
            "Plan for August 7 through August 9.",
            (date(2026, 8, 7), date(2026, 8, 9)),
        ),
        (
            WeatherScopeSelection(
                scope_revision=0,
                location_action="clear",
                window_action="set",
                date="2026-07-29",
            ),
            "Plan for tomorrow.",
            (date(2026, 7, 29), date(2026, 7, 29)),
        ),
    ],
)
def test_explicit_scope_window_must_equal_current_turn_authority(
    selection: WeatherScopeSelection,
    shopper_text: str,
    expected_window: tuple[date, date],
) -> None:
    resolution = compile_weather_scope_resolution(
        selection,
        current_scope=CurrentWeatherScope(),
        current_shopper_text=shopper_text,
        saved_zip_authorized=False,
        expected_projection_version=0,
        current_date=CURRENT_DATE,
    )

    assert resolution.requested_window == WeatherReceiptWindow(
        start_date=expected_window[0],
        end_date=expected_window[1],
    )


@pytest.mark.parametrize(
    ("selection", "shopper_text"),
    [
        (
            WeatherScopeSelection(
                scope_revision=0,
                location_action="clear",
                window_action="set",
                date="2026-08-08",
            ),
            "Plan for 2026-08-07.",
        ),
        (
            WeatherScopeSelection(
                scope_revision=0,
                location_action="clear",
                window_action="set",
                start_date="2026-08-07",
                end_date="2026-08-10",
            ),
            "Plan for August 7 through August 9.",
        ),
    ],
)
def test_explicit_scope_window_rejects_model_selected_date_drift(
    selection: WeatherScopeSelection,
    shopper_text: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="does not match current-turn authority",
    ):
        compile_weather_scope_resolution(
            selection,
            current_scope=CurrentWeatherScope(),
            current_shopper_text=shopper_text,
            saved_zip_authorized=False,
            expected_projection_version=0,
            current_date=CURRENT_DATE,
        )


def test_non_event_weather_request_compiles_and_calls_bound_tool() -> None:
    start_date = date(2026, 8, 3)
    end_date = date(2026, 8, 9)
    client = RecordingClient(
        _success_result(
            resolved_location="Denver, Colorado, United States",
            start_date=start_date,
            end_date=end_date,
        )
    )
    resolution = compile_weather_scope_resolution(
        WeatherScopeSelection(
            scope_revision=0,
            location_action="set",
            window_action="set",
            location_source="shopper_provided_location",
            location="Denver",
            relative_date="next_week",
        ),
        current_scope=CurrentWeatherScope(),
        current_shopper_text="What should I wear in Denver next week?",
        saved_zip_authorized=False,
        expected_projection_version=0,
        current_date=CURRENT_DATE,
    )
    binding = WeatherScopeForecastBinding(
        CurrentWeatherScope(),
        saved_zipcode=None,
    )
    binding.bind(
        resolution,
        relative_date="next_week",
        weekday=None,
    )
    tool = get_scoped_weather_forecast_tool(client, binding)

    result = parse_weather_tool_evidence(tool.invoke({}))

    assert tool.args == {}
    assert isinstance(result, WeatherForecastEvidence)
    assert result.relative_date == "next_week"
    assert client.requests == [
        WeatherRequest(
            location="Denver",
            start_date=start_date,
            end_date=end_date,
        )
    ]


def test_qualified_nyc_friday_scope_reaches_zero_argument_adapter() -> None:
    friday = date(2026, 8, 7)
    client = RecordingClient(
        _success_result(
            resolved_location="New York, NY, United States",
            start_date=friday,
            end_date=friday,
        )
    )
    resolution = compile_weather_scope_resolution(
        WeatherScopeSelection(
            scope_revision=0,
            location_action="set",
            window_action="set",
            location_source="shopper_provided_location",
            location="NYC",
            location_query="NYC, NY",
            relative_date="next_week",
            weekday="friday",
        ),
        current_scope=CurrentWeatherScope(),
        current_shopper_text="What should I wear in NYC Friday next week?",
        saved_zip_authorized=False,
        expected_projection_version=0,
        current_date=CURRENT_DATE,
    )
    binding = WeatherScopeForecastBinding(
        CurrentWeatherScope(),
        saved_zipcode=None,
    )
    binding.bind(
        resolution,
        relative_date="next_week",
        weekday="friday",
    )

    evidence = parse_weather_tool_evidence(
        get_scoped_weather_forecast_tool(client, binding).invoke({})
    )

    assert client.requests == [
        WeatherRequest(
            location="NYC, NY",
            start_date=friday,
            end_date=friday,
        )
    ]
    assert isinstance(evidence, WeatherForecastEvidence)
    assert evidence.relative_date == "next_week"
    assert evidence.weekday == "friday"


def test_incomplete_resolution_never_calls_weather_provider() -> None:
    client = RecordingClient()
    current_scope = _existing_nyc_scope()
    resolution = compile_weather_scope_resolution(
        WeatherScopeSelection(
            scope_revision=1,
            location_action="set",
            window_action="clear",
            location_source="shopper_provided_location",
            location="Seattle",
        ),
        current_scope=current_scope,
        current_shopper_text="A different wedding in Seattle.",
        saved_zip_authorized=False,
        expected_projection_version=4,
        current_date=CURRENT_DATE,
    )
    binding = WeatherScopeForecastBinding(
        current_scope,
        saved_zipcode=None,
    )
    binding.bind(resolution, relative_date=None, weekday=None)

    result = parse_weather_tool_evidence(
        get_scoped_weather_forecast_tool(client, binding).invoke({})
    )

    assert result == weather_failure("weather_request_invalid")
    assert client.requests == []


def test_tool_has_the_closed_name_schema_and_direct_result() -> None:
    client = RecordingClient()
    tool = get_weather_forecast_tool(client)

    result = tool.invoke(
        {
            "location": "Denver",
            "start_date": "2026-07-28",
            "end_date": "2026-07-29",
        }
    )

    assert tool.name == "get_weather_forecast_tool"
    assert tool.return_direct is False
    assert set(tool.args) == {"location", "date", "start_date", "end_date"}
    assert result["code"] == "weather_disabled"
    assert client.requests == [
        WeatherRequest(
            location="Denver",
            start_date="2026-07-28",
            end_date="2026-07-29",
        )
    ]


def test_tool_validation_returns_only_the_sanitized_typed_failure() -> None:
    client = RecordingClient()
    tool = get_weather_forecast_tool(client)

    result = tool.invoke(
        {
            "location": "Seattle",
            "date": "next week",
            "extra": "must not enter the contract",
        }
    )

    parsed = json.loads(result)
    assert parsed == {
        "ok": False,
        "code": "weather_request_invalid",
        "message": "The weather request is invalid.",
        "retryable": False,
    }
    assert client.requests == []
    assert "Seattle" not in result
    assert "next week" not in result


def test_event_tool_has_closed_authority_and_date_schema() -> None:
    tool = _event_tool(RecordingClient())

    assert tool.name == "get_weather_forecast_tool"
    assert tool.return_direct is False
    assert set(tool.args) == {
        "location_source",
        "location",
        "location_query",
        "relative_date",
        "weekday",
        "date",
        "start_date",
        "end_date",
    }
    query_schema = tool.args_schema.model_json_schema()["properties"][
        "location_query"
    ]
    assert "Required when location is an abbreviation" in query_schema[
        "description"
    ]
    assert "location='NYC', location_query='NYC, NY'" in query_schema[
        "description"
    ]
    relative_date_schema = tool.args_schema.model_json_schema()["properties"][
        "relative_date"
    ]
    assert "Never use this as a placeholder" in relative_date_schema[
        "description"
    ]
    assert "For an abbreviation or geographically ambiguous name" in (
        tool.description
    )
    assert "exact '<weekday> next week' phrase" in tool.description
    assert "matching lowercase weekday" in tool.description
    assert "do not call this tool; apply the event-context" in (
        tool.description
    )
    assert SHOPPING_TOOL_POLICIES[tool.name].allowed_skills_any_of == frozenset(
        {"event-context"}
    )
    assert SHOPPING_TOOL_POLICIES[tool.name].risk == "read"


def test_schema_invalid_attempt_is_consumed_and_cannot_retry() -> None:
    client = RecordingClient(_success_result())
    tool = _event_tool(client)

    invalid = tool.invoke(
        {
            "location_source": "confirmed_saved_zip",
        }
    )
    retry = tool.invoke(
        {
            "location_source": "confirmed_saved_zip",
            "date": FORECAST_DATE.isoformat(),
        }
    )

    assert parse_weather_tool_evidence(invalid) == weather_failure(
        "weather_request_invalid"
    )
    assert parse_weather_tool_evidence(retry) == weather_failure(
        "weather_request_invalid"
    )
    assert client.requests == []


def test_confirmed_saved_zip_is_server_supplied_and_not_exposed() -> None:
    client = RecordingClient(_success_result())
    tool = _event_tool(client)

    result = tool.invoke(
        {
            "location_source": "confirmed_saved_zip",
            "date": FORECAST_DATE.isoformat(),
        }
    )

    assert client.requests == [
        WeatherRequest(
            location=SAVED_ZIPCODE,
            date=FORECAST_DATE,
        )
    ]
    assert result.startswith(WEATHER_FORECAST_EVIDENCE_PREFIX)
    parsed = parse_weather_tool_evidence(result)
    assert isinstance(parsed, WeatherForecastEvidence)
    assert parsed.requested_window.start_date == FORECAST_DATE
    assert parsed.days[0].condition == "rain"
    assert parsed.attribution.label == VISUAL_CROSSING_ATTRIBUTION_LABEL
    assert parsed.attribution.url == VISUAL_CROSSING_ATTRIBUTION_URL
    for omitted in (
        SAVED_ZIPCODE,
        "Seattle",
        "resolved_location",
        "America/Los_Angeles",
        "timezone",
    ):
        assert omitted not in result


def test_unconfirmed_saved_zip_never_reaches_the_client() -> None:
    client = RecordingClient(_success_result())
    tool = _event_tool(client, saved_zip_authorized=False)

    result = tool.invoke(
        {
            "location_source": "confirmed_saved_zip",
            "date": FORECAST_DATE.isoformat(),
        }
    )

    assert parse_weather_tool_evidence(result) == weather_failure(
        "weather_request_invalid"
    )
    assert client.requests == []
    assert SAVED_ZIPCODE not in result


def test_shopper_place_is_verbatim_grounded_and_resolution_is_visible() -> None:
    explicit_client = RecordingClient(
        _success_result(
            resolved_location="New York, New York, United States"
        )
    )
    explicit_tool = _event_tool(explicit_client)

    success = explicit_tool.invoke(
        {
            "location_source": "shopper_provided_location",
            "location": EXPLICIT_LOCATION,
            "location_query": "NYC, NY",
            "date": FORECAST_DATE.isoformat(),
        }
    )

    assert success.startswith(WEATHER_FORECAST_EVIDENCE_PREFIX)
    assert explicit_client.requests == [
        WeatherRequest(
            location="NYC, NY",
            date=FORECAST_DATE,
        )
    ]
    parsed = parse_weather_tool_evidence(success)
    assert isinstance(parsed, WeatherForecastEvidence)
    assert (
        parsed.resolved_location
        == "New York, New York, United States"
    )
    assert SAVED_ZIPCODE not in success

    ungrounded_client = RecordingClient(_success_result())
    ungrounded_tool = _event_tool(ungrounded_client)
    failure = ungrounded_tool.invoke(
        {
            "location_source": "shopper_provided_location",
            "location": "New York, New York",
            "location_query": "New York, NY",
            "date": FORECAST_DATE.isoformat(),
        }
    )

    parsed = parse_weather_tool_evidence(failure)
    assert parsed == weather_failure("weather_request_invalid")
    assert failure.startswith(WEATHER_FORECAST_FAILURE_PREFIX)
    assert ungrounded_client.requests == []
    assert SAVED_ZIPCODE not in failure
    assert "New York" not in failure


@pytest.mark.parametrize(
    ("shopper_text", "location"),
    [
        ("The wedding is in Denver next Saturday.", "Denver"),
        ("It is in Cancun next week.", "Cancun"),
        ("Paris, France on August 6.", "Paris, France"),
        ("The venue is in Cancún, Mexico.", "Cancún, Mexico"),
        ("Use postal code 10011.", "10011"),
    ],
)
def test_shopper_named_locations_reach_the_client_verbatim(
    shopper_text: str,
    location: str,
) -> None:
    client = RecordingClient(_success_result())
    tool = _event_tool(
        client,
        shopper_provided_texts={shopper_text},
    )

    result = tool.invoke(
        {
            "location_source": "shopper_provided_location",
            "location": location,
            "date": FORECAST_DATE.isoformat(),
        }
    )

    assert result.startswith(WEATHER_FORECAST_EVIDENCE_PREFIX)
    assert client.requests[0].location == location


def test_location_authority_forwards_the_exact_shopper_text_span() -> None:
    client = RecordingClient(_success_result())
    tool = _event_tool(
        client,
        shopper_provided_texts={"The wedding is in NYC next week."},
    )

    result = tool.invoke(
        {
            "location_source": "shopper_provided_location",
            "location": "nyc",
            "date": FORECAST_DATE.isoformat(),
        }
    )

    assert result.startswith(WEATHER_FORECAST_EVIDENCE_PREFIX)
    assert client.requests == [
        WeatherRequest(location="NYC", date=FORECAST_DATE)
    ]


def test_location_authority_rejects_a_rewritten_place_phrase() -> None:
    client = RecordingClient(_success_result())
    tool = _event_tool(
        client,
        shopper_provided_texts={"The wedding is in New   York next week."},
    )

    result = tool.invoke(
        {
            "location_source": "shopper_provided_location",
            "location": "New York",
            "date": FORECAST_DATE.isoformat(),
        }
    )

    assert parse_weather_tool_evidence(result) == weather_failure(
        "weather_request_invalid"
    )
    assert client.requests == []


def test_location_query_may_make_an_ambiguous_place_assumption() -> None:
    client = RecordingClient(
        _success_result(resolved_location="Springfield, TX, United States")
    )
    tool = _event_tool(
        client,
        shopper_provided_texts={
            "The reception is in Springfield next week."
        },
    )

    result = tool.invoke(
        {
            "location_source": "shopper_provided_location",
            "location": "Springfield",
            "location_query": "Springfield, TX",
            "date": FORECAST_DATE.isoformat(),
        }
    )

    assert client.requests == [
        WeatherRequest(location="Springfield, TX", date=FORECAST_DATE)
    ]
    parsed = parse_weather_tool_evidence(result)
    assert isinstance(parsed, WeatherForecastEvidence)
    assert parsed.resolved_location == "Springfield, TX, United States"


def test_location_query_cannot_add_an_unstated_representative_zip() -> None:
    client = RecordingClient(_success_result())
    tool = _event_tool(
        client,
        shopper_provided_texts={"The wedding is in Denver next week."},
    )

    result = tool.invoke(
        {
            "location_source": "shopper_provided_location",
            "location": "Denver",
            "location_query": "Denver, CO 80202",
            "date": FORECAST_DATE.isoformat(),
        }
    )

    assert parse_weather_tool_evidence(result) == weather_failure(
        "weather_request_invalid"
    )
    assert client.requests == []


@pytest.mark.parametrize(
    ("shopper_location", "location_query"),
    [
        ("NYC", "New York, NY"),
        ("NYC", "Nairobi, Kenya"),
        ("LA", "London, UK"),
        ("Springfield", "Springfield Gardens, NY"),
        ("Denver", "Denver Colorado"),
        ("Paris", "Paris, France, Europe, Earth"),
    ],
)
def test_location_query_cannot_rewrite_the_authoritative_place(
    shopper_location: str,
    location_query: str,
) -> None:
    client = RecordingClient(_success_result())
    tool = _event_tool(
        client,
        shopper_provided_texts={f"The wedding is in {shopper_location}."},
    )

    result = tool.invoke(
        {
            "location_source": "shopper_provided_location",
            "location": shopper_location,
            "location_query": location_query,
            "date": FORECAST_DATE.isoformat(),
        }
    )

    assert parse_weather_tool_evidence(result) == weather_failure(
        "weather_request_invalid"
    )
    assert client.requests == []


@pytest.mark.parametrize(
    "arguments",
    [
        {
            "location_source": "confirmed_saved_zip",
            "location": SAVED_ZIPCODE,
            "date": FORECAST_DATE.isoformat(),
        },
        {
            "location_source": "confirmed_saved_zip",
            "location_query": "Seattle, WA",
            "date": FORECAST_DATE.isoformat(),
        },
        {
            "location_source": "shopper_provided_location",
            "date": FORECAST_DATE.isoformat(),
        },
        {
            "location_source": "shopper_provided_location",
            "location": EXPLICIT_LOCATION,
        },
    ],
)
def test_event_tool_rejects_ambiguous_authority_or_missing_date(
    arguments: dict[str, str],
) -> None:
    client = RecordingClient(_success_result())
    tool = _event_tool(client)

    result = tool.invoke(arguments)

    assert parse_weather_tool_evidence(result) == weather_failure(
        "weather_request_invalid"
    )
    assert client.requests == []


@pytest.mark.parametrize(
    ("anchor", "expected_start"),
    [
        (date(2026, 7, 27), date(2026, 8, 3)),
        (date(2026, 7, 28), date(2026, 8, 3)),
        (date(2026, 8, 2), date(2026, 8, 3)),
    ],
)
def test_next_week_is_server_resolved_to_next_monday_through_sunday(
    anchor: date,
    expected_start: date,
) -> None:
    client = RecordingClient()
    tool = _event_tool(client, current_date=anchor)

    result = tool.invoke(
        {
            "location_source": "shopper_provided_location",
            "location": EXPLICIT_LOCATION,
            "relative_date": "next_week",
        }
    )

    assert parse_weather_tool_evidence(result) == weather_failure(
        "weather_disabled"
    )
    assert client.requests == [
        WeatherRequest(
            location=EXPLICIT_LOCATION,
            start_date=expected_start,
            end_date=expected_start + timedelta(days=6),
        )
    ]


def test_next_week_success_preserves_server_owned_relative_provenance() -> None:
    expected_start = date(2026, 8, 3)
    client = RecordingClient(
        _success_result(
            resolved_location="New York, New York, United States",
            start_date=expected_start,
            end_date=expected_start + timedelta(days=6),
        )
    )
    tool = _event_tool(client)

    result = tool.invoke(
        {
            "location_source": "shopper_provided_location",
            "location": EXPLICIT_LOCATION,
            "location_query": "NYC, NY",
            "relative_date": "next_week",
        }
    )

    parsed = parse_weather_tool_evidence(result)
    assert isinstance(parsed, WeatherForecastEvidence)
    assert parsed.relative_date == "next_week"
    assert parsed.requested_window == WeatherRequestedWindow(
        start_date=expected_start,
        end_date=expected_start + timedelta(days=6),
    )
    assert client.requests == [
        WeatherRequest(
            location="NYC, NY",
            start_date=expected_start,
            end_date=expected_start + timedelta(days=6),
        )
    ]


@pytest.mark.parametrize(
    ("weekday", "offset"),
    [
        ("monday", 0),
        ("tuesday", 1),
        ("wednesday", 2),
        ("thursday", 3),
        ("friday", 4),
        ("saturday", 5),
        ("sunday", 6),
    ],
)
def test_weekday_next_week_is_server_resolved_to_one_exact_date(
    weekday: str,
    offset: int,
) -> None:
    next_monday = date(2026, 8, 3)
    event_date = next_monday + timedelta(days=offset)
    client = RecordingClient(
        _success_result(
            resolved_location="New York, New York, United States",
            start_date=event_date,
            end_date=event_date,
        )
    )
    tool = _event_tool(
        client,
        shopper_provided_texts={
            f"The wedding is in NYC on {weekday.title()} next week."
        },
    )

    result = tool.invoke(
        {
            "location_source": "shopper_provided_location",
            "location": "NYC",
            "location_query": "NYC, NY",
            "relative_date": "next_week",
            "weekday": weekday,
        }
    )

    parsed = parse_weather_tool_evidence(result)
    assert isinstance(parsed, WeatherForecastEvidence)
    assert parsed.relative_date == "next_week"
    assert parsed.weekday == weekday
    assert parsed.requested_window == WeatherRequestedWindow(
        start_date=event_date,
        end_date=event_date,
    )
    assert client.requests == [
        WeatherRequest(location="NYC, NY", date=event_date)
    ]


@pytest.mark.parametrize(
    "shopper_text",
    [
        "The wedding is in NYC Friday next week, maybe outdoors.",
        "The venue is possibly outside; the NYC wedding is Friday next week.",
        "I did not say blue. The NYC wedding is Friday next week.",
    ],
)
def test_relative_weekday_ignores_unrelated_uncertainty(
    shopper_text: str,
) -> None:
    event_date = date(2026, 8, 7)
    client = RecordingClient(
        _success_result(start_date=event_date, end_date=event_date)
    )
    tool = _event_tool(
        client,
        shopper_provided_texts={shopper_text},
    )

    result = tool.invoke(
        {
            "location_source": "shopper_provided_location",
            "location": "NYC",
            "relative_date": "next_week",
            "weekday": "friday",
        }
    )

    parsed = parse_weather_tool_evidence(result)
    assert isinstance(parsed, WeatherForecastEvidence)
    assert parsed.weekday == "friday"
    assert client.requests == [WeatherRequest(location="NYC", date=event_date)]


def test_relative_weekday_evidence_rejects_a_different_calendar_weekday() -> None:
    thursday = date(2026, 8, 6)
    result = _success_result(start_date=thursday, end_date=thursday)

    with pytest.raises(ValueError, match="relative weekday"):
        WeatherForecastEvidence(
            provider=result.provider,
            fetched_at=result.fetched_at,
            requested_window=result.requested_window,
            relative_date="next_week",
            weekday="friday",
            days=result.days,
            attribution=result.attribution,
        )


@pytest.mark.parametrize("supplied_weekday", [None, "thursday"])
def test_exact_weekday_next_week_rejects_omission_or_mismatch(
    supplied_weekday: str | None,
) -> None:
    client = RecordingClient(_success_result())
    tool = _event_tool(
        client,
        shopper_provided_texts={
            "The wedding is in NYC on Friday next week."
        },
    )
    arguments = {
        "location_source": "shopper_provided_location",
        "location": "NYC",
        "relative_date": "next_week",
    }
    if supplied_weekday is not None:
        arguments["weekday"] = supplied_weekday

    result = tool.invoke(arguments)

    assert parse_weather_tool_evidence(result) == weather_failure(
        "weather_request_invalid"
    )
    assert client.requests == []


def test_bare_next_week_rejects_an_invented_weekday() -> None:
    client = RecordingClient(_success_result())
    tool = _event_tool(
        client,
        shopper_provided_texts={"The wedding is in NYC next week."},
    )

    result = tool.invoke(
        {
            "location_source": "shopper_provided_location",
            "location": "NYC",
            "relative_date": "next_week",
            "weekday": "friday",
        }
    )

    assert parse_weather_tool_evidence(result) == weather_failure(
        "weather_request_invalid"
    )
    assert client.requests == []


def test_mixed_weekdays_next_week_fail_closed() -> None:
    client = RecordingClient(_success_result())
    tool = _event_tool(
        client,
        shopper_provided_texts={
            "The wedding could be Friday next week or Saturday next week."
        },
    )

    result = tool.invoke(
        {
            "location_source": "shopper_provided_location",
            "location": "NYC",
            "relative_date": "next_week",
            "weekday": "friday",
        }
    )

    assert parse_weather_tool_evidence(result) == weather_failure(
        "weather_request_invalid"
    )
    assert client.requests == []


@pytest.mark.parametrize(
    "shopper_text",
    [
        "The wedding is Friday next week or Saturday.",
        "The wedding is Friday or Saturday next week.",
        "Maybe Friday next week.",
        "I didn't say Friday next week.",
    ],
)
def test_ambiguous_or_retracted_relative_weekday_fails_closed(
    shopper_text: str,
) -> None:
    client = RecordingClient(_success_result())
    tool = _event_tool(
        client,
        shopper_provided_texts={shopper_text},
    )

    result = tool.invoke(
        {
            "location_source": "shopper_provided_location",
            "location": "NYC",
            "relative_date": "next_week",
            "weekday": "friday",
        }
    )

    assert parse_weather_tool_evidence(result) == weather_failure(
        "weather_request_invalid"
    )
    assert client.requests == []


def test_weekday_requires_next_week_mode() -> None:
    client = RecordingClient(_success_result())
    tool = _event_tool(client)

    result = tool.invoke(
        {
            "location_source": "shopper_provided_location",
            "location": "NYC",
            "weekday": "friday",
            "date": FORECAST_DATE.isoformat(),
        }
    )

    assert parse_weather_tool_evidence(result) == weather_failure(
        "weather_request_invalid"
    )
    assert client.requests == []


def test_next_week_requires_shopper_authored_relative_language() -> None:
    client = RecordingClient(_success_result())
    tool = _event_tool(
        client,
        shopper_provided_texts={"The wedding is in NYC in August."},
    )

    result = tool.invoke(
        {
            "location_source": "shopper_provided_location",
            "location": EXPLICIT_LOCATION,
            "relative_date": "next_week",
        }
    )

    assert parse_weather_tool_evidence(result) == weather_failure(
        "weather_request_invalid"
    )
    assert client.requests == []


@pytest.mark.parametrize(
    "current_text",
    [
        "Actually, not next week; use September 10.",
        "Forget next week; use September 10.",
        "Next week was wrong; use September 10.",
        "Next week, actually September 10.",
        "Use September 10 instead.",
        "The wedding is tomorrow.",
        "Don't use next week.",
        "I can't do next week.",
        "Next week has been canceled.",
        "No next week.",
        "Next week is off.",
        "Next week is no longer the date.",
        "I changed it from next week.",
        "The date changed; I don't know the new date.",
        "That date is no longer right.",
        "That date won't work.",
        "We don't have a date yet.",
        "We haven't set the date.",
        "We need a new date.",
        "I'll send the new date later.",
        "That isn't the date anymore.",
        "The date no longer works.",
        "The date is up in the air.",
    ],
)
def test_current_date_override_cannot_reuse_prior_next_week(
    current_text: str,
) -> None:
    client = RecordingClient(_success_result())
    tool = _event_tool(
        client,
        shopper_provided_texts=(
            current_text,
            "The wedding was originally in NYC next week.",
        ),
    )

    result = tool.invoke(
        {
            "location_source": "shopper_provided_location",
            "location": "NYC",
            "relative_date": "next_week",
        }
    )

    assert parse_weather_tool_evidence(result) == weather_failure(
        "weather_request_invalid"
    )
    assert client.requests == []


def test_latest_prior_negation_supersedes_older_next_week() -> None:
    client = RecordingClient(_success_result())
    tool = _event_tool(
        client,
        shopper_provided_texts=(
            "NYC.",
            "The wedding was going to be next week.",
            "Actually, not next week.",
        ),
    )

    result = tool.invoke(
        {
            "location_source": "shopper_provided_location",
            "location": "NYC",
            "relative_date": "next_week",
        }
    )

    assert parse_weather_tool_evidence(result) == weather_failure(
        "weather_request_invalid"
    )
    assert client.requests == []


def test_standalone_weekday_correction_cannot_reuse_prior_relative_weekday() -> None:
    client = RecordingClient(_success_result())
    tool = _event_tool(
        client,
        shopper_provided_texts=(
            "Saturday instead.",
            "The wedding was originally Friday next week.",
        ),
    )

    result = tool.invoke(
        {
            "location_source": "shopper_provided_location",
            "location": "NYC",
            "relative_date": "next_week",
            "weekday": "friday",
        }
    )

    assert parse_weather_tool_evidence(result) == weather_failure(
        "weather_request_invalid"
    )
    assert client.requests == []


def test_current_relative_weekday_supersedes_prior_relative_weekday() -> None:
    event_date = date(2026, 8, 8)
    client = RecordingClient(
        _success_result(start_date=event_date, end_date=event_date)
    )
    tool = _event_tool(
        client,
        shopper_provided_texts=(
            "Actually, Saturday next week.",
            "The wedding in NYC was originally Friday next week.",
        ),
    )

    result = tool.invoke(
        {
            "location_source": "shopper_provided_location",
            "location": "NYC",
            "relative_date": "next_week",
            "weekday": "saturday",
        }
    )

    parsed = parse_weather_tool_evidence(result)
    assert isinstance(parsed, WeatherForecastEvidence)
    assert parsed.weekday == "saturday"
    assert client.requests == [WeatherRequest(location="NYC", date=event_date)]


@pytest.mark.parametrize(
    ("shopper_texts", "expected"),
    [
        ((), False),
        (("NYC, on an outdoor patio.",), False),
        (("The wedding is next week.",), True),
        (("The wedding is Friday next week.",), True),
        (("The wedding is tomorrow.",), True),
        (("The wedding is 2026-09-10.",), True),
        (("The wedding is September 10.",), True),
        (("Use September 10 instead.",), True),
        (("September 10 through September 12.",), True),
        (("September 10, 2026 through September 12, 2026.",), True),
        (("2026-09-10 to 2026-09-12.",), True),
        (("September 10 or September 12.",), False),
        (("The wedding is not September 10.",), False),
        (("I need a wedding dress in NYC, size 8/10.",), False),
        (("Friday next week, maybe outdoors.",), True),
        (
            ("The venue is possibly outside; wedding Friday next week.",),
            True,
        ),
        (
            ("I did not say blue. The wedding is Friday next week.",),
            True,
        ),
        (("September 10, maybe outdoors.",), True),
        (("Tomorrow, maybe on a patio.",), True),
        (("Friday next week, maybe.",), False),
        (("September 10, maybe.",), False),
        (("The wedding is Friday next week or Saturday.",), False),
        (("The wedding is Friday or Saturday next week.",), False),
        (("Maybe Friday next week.",), False),
        (("I didn't say Friday next week.",), False),
        (
            (
                "NYC, on an outdoor patio.",
                "The wedding is Friday next week.",
            ),
            True,
        ),
        (
            (
                "Saturday instead.",
                "The wedding was Friday next week.",
            ),
            False,
        ),
        (
            (
                "Don't use next week.",
                "The wedding was Friday next week.",
            ),
            False,
        ),
        (
            (
                "The date changed; I don't know the new date.",
                "The wedding was Friday next week.",
            ),
            False,
        ),
    ],
)
def test_weather_date_context_visibility_uses_latest_shopper_authority(
    shopper_texts: tuple[str, ...],
    expected: bool,
) -> None:
    assert weather_date_context_available(shopper_texts) is expected


def test_location_followup_may_use_prior_unsuperseded_next_week() -> None:
    expected_start = date(2026, 8, 3)
    client = RecordingClient(
        _success_result(
            start_date=expected_start,
            end_date=expected_start + timedelta(days=6),
        )
    )
    tool = _event_tool(
        client,
        shopper_provided_texts=(
            "NYC.",
            "The wedding is next week.",
        ),
    )

    result = tool.invoke(
        {
            "location_source": "shopper_provided_location",
            "location": "NYC",
            "relative_date": "next_week",
        }
    )

    parsed = parse_weather_tool_evidence(result)
    assert isinstance(parsed, WeatherForecastEvidence)
    assert parsed.relative_date == "next_week"


def test_relative_and_explicit_date_modes_are_mutually_exclusive() -> None:
    client = RecordingClient(_success_result())
    tool = _event_tool(client)

    result = tool.invoke(
        {
            "location_source": "shopper_provided_location",
            "location": EXPLICIT_LOCATION,
            "relative_date": "next_week",
            "date": FORECAST_DATE.isoformat(),
        }
    )

    assert parse_weather_tool_evidence(result) == weather_failure(
        "weather_request_invalid"
    )
    assert client.requests == []


def test_event_tool_allows_only_one_attempt_per_turn() -> None:
    client = RecordingClient(_success_result())
    tool = _event_tool(client)
    arguments = {
        "location_source": "confirmed_saved_zip",
        "date": FORECAST_DATE.isoformat(),
    }

    first = tool.invoke(arguments)
    second = tool.invoke(arguments)

    assert first.startswith(WEATHER_FORECAST_EVIDENCE_PREFIX)
    assert parse_weather_tool_evidence(second) == weather_failure(
        "weather_request_invalid"
    )
    assert len(client.requests) == 1


def test_schema_invalid_event_call_consumes_the_turn_attempt() -> None:
    client = RecordingClient(_success_result())
    tool = _event_tool(client)

    invalid = tool.invoke(
        {
            "location_source": "confirmed_saved_zip",
        }
    )
    retry = tool.invoke(
        {
            "location_source": "confirmed_saved_zip",
            "date": FORECAST_DATE.isoformat(),
        }
    )

    assert parse_weather_tool_evidence(invalid) == weather_failure(
        "weather_request_invalid"
    )
    assert parse_weather_tool_evidence(retry) == weather_failure(
        "weather_request_invalid"
    )
    assert client.requests == []


def test_direct_and_event_wrappers_sanitize_client_exceptions() -> None:
    direct_client = RecordingClient(
        error=RuntimeError(SENSITIVE_EXCEPTION)
    )
    direct_tool = get_weather_forecast_tool(direct_client)

    direct_result = direct_tool.invoke(
        {
            "location": SAVED_ZIPCODE,
            "date": FORECAST_DATE.isoformat(),
        }
    )

    assert direct_result == weather_failure(
        "weather_unavailable"
    ).model_dump(mode="json")
    serialized_direct = json.dumps(direct_result)
    assert SENSITIVE_EXCEPTION not in serialized_direct
    assert SAVED_ZIPCODE not in serialized_direct

    event_client = RecordingClient(
        error=RuntimeError(SENSITIVE_EXCEPTION)
    )
    event_tool = _event_tool(event_client)
    event_result = event_tool.invoke(
        {
            "location_source": "confirmed_saved_zip",
            "date": FORECAST_DATE.isoformat(),
        }
    )

    assert parse_weather_tool_evidence(event_result) == weather_failure(
        "weather_unavailable"
    )
    assert SENSITIVE_EXCEPTION not in event_result
    assert SAVED_ZIPCODE not in event_result
    assert "Seattle" not in event_result
