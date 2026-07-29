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
    WeatherForecastEvidence,
    get_event_weather_forecast_tool,
    get_weather_forecast_tool,
    parse_weather_tool_evidence,
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
    prior_candidates_available: bool = False,
    on_reuse_prior_candidates=None,
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
        prior_candidates_available=prior_candidates_available,
        on_reuse_prior_candidates=on_reuse_prior_candidates,
    )


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
        "candidate_action",
        "location_source",
        "location",
        "location_query",
        "relative_date",
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
    assert "For an abbreviation or geographically ambiguous name" in (
        tool.description
    )
    assert SHOPPING_TOOL_POLICIES[tool.name].allowed_skills_any_of == frozenset(
        {"event-context"}
    )
    assert SHOPPING_TOOL_POLICIES[tool.name].risk == "read"


def test_reusing_prior_candidates_requires_evidence_and_closes_search() -> None:
    rejected_client = RecordingClient()
    rejected_callbacks: list[str] = []
    rejected_tool = _event_tool(
        rejected_client,
        on_reuse_prior_candidates=lambda: rejected_callbacks.append("closed"),
    )

    rejected = rejected_tool.invoke(
        {
            "candidate_action": "reuse_prior_candidates",
            "location_source": "confirmed_saved_zip",
            "date": FORECAST_DATE.isoformat(),
        }
    )

    assert parse_weather_tool_evidence(rejected) == weather_failure(
        "weather_request_invalid"
    )
    assert rejected_client.requests == []
    assert rejected_callbacks == []

    accepted_client = RecordingClient()
    accepted_callbacks: list[str] = []
    accepted_tool = _event_tool(
        accepted_client,
        prior_candidates_available=True,
        on_reuse_prior_candidates=lambda: accepted_callbacks.append("closed"),
    )

    accepted = accepted_tool.invoke(
        {
            "candidate_action": "reuse_prior_candidates",
            "location_source": "confirmed_saved_zip",
            "date": FORECAST_DATE.isoformat(),
        }
    )

    assert parse_weather_tool_evidence(accepted) == weather_failure(
        "weather_disabled"
    )
    assert accepted_client.requests == [
        WeatherRequest(location=SAVED_ZIPCODE, date=FORECAST_DATE)
    ]
    assert accepted_callbacks == ["closed"]


def test_new_candidate_request_keeps_catalog_search_open() -> None:
    client = RecordingClient()
    callbacks: list[str] = []
    tool = _event_tool(
        client,
        prior_candidates_available=True,
        on_reuse_prior_candidates=lambda: callbacks.append("closed"),
    )

    result = tool.invoke(
        {
            "candidate_action": "search_new_candidates",
            "location_source": "confirmed_saved_zip",
            "date": FORECAST_DATE.isoformat(),
        }
    )

    assert parse_weather_tool_evidence(result) == weather_failure(
        "weather_disabled"
    )
    assert callbacks == []


def test_confirmed_saved_zip_is_server_supplied_and_not_exposed() -> None:
    client = RecordingClient(_success_result())
    tool = _event_tool(client)

    result = tool.invoke(
        {
            "candidate_action": "search_new_candidates",
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
            "candidate_action": "search_new_candidates",
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
            "candidate_action": "search_new_candidates",
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
            "candidate_action": "search_new_candidates",
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
            "candidate_action": "search_new_candidates",
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
            "candidate_action": "search_new_candidates",
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
            "candidate_action": "search_new_candidates",
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
            "candidate_action": "search_new_candidates",
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
            "candidate_action": "search_new_candidates",
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
            "candidate_action": "search_new_candidates",
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
            "date": FORECAST_DATE.isoformat(),
        },
        {
            "candidate_action": "search_new_candidates",
            "location_source": "confirmed_saved_zip",
            "location": SAVED_ZIPCODE,
            "date": FORECAST_DATE.isoformat(),
        },
        {
            "candidate_action": "search_new_candidates",
            "location_source": "confirmed_saved_zip",
            "location_query": "Seattle, WA",
            "date": FORECAST_DATE.isoformat(),
        },
        {
            "candidate_action": "search_new_candidates",
            "location_source": "shopper_provided_location",
            "date": FORECAST_DATE.isoformat(),
        },
        {
            "candidate_action": "search_new_candidates",
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
            "candidate_action": "search_new_candidates",
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
            "candidate_action": "search_new_candidates",
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


def test_next_week_requires_shopper_authored_relative_language() -> None:
    client = RecordingClient(_success_result())
    tool = _event_tool(
        client,
        shopper_provided_texts={"The wedding is in NYC in August."},
    )

    result = tool.invoke(
        {
            "candidate_action": "search_new_candidates",
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
            "candidate_action": "search_new_candidates",
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
            "candidate_action": "search_new_candidates",
            "location_source": "shopper_provided_location",
            "location": "NYC",
            "relative_date": "next_week",
        }
    )

    assert parse_weather_tool_evidence(result) == weather_failure(
        "weather_request_invalid"
    )
    assert client.requests == []


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
            "candidate_action": "search_new_candidates",
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
            "candidate_action": "search_new_candidates",
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
        "candidate_action": "search_new_candidates",
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
            "candidate_action": "search_new_candidates",
            "location_source": "confirmed_saved_zip",
        }
    )
    retry = tool.invoke(
        {
            "candidate_action": "search_new_candidates",
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
            "candidate_action": "search_new_candidates",
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
