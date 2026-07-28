# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline contract tests for the dormant weather client and provider adapter."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pytest
import requests
from pydantic import ValidationError

from chain_server.src.weather import (
    MAX_PROVIDER_RESPONSE_BYTES,
    VISUAL_CROSSING_ATTRIBUTION_URL,
    VisualCrossingWeatherClient,
    WeatherConfig,
    WeatherFailure,
    WeatherRequest,
    WeatherResult,
    build_weather_client,
)


NOW = datetime(2026, 7, 27, 18, 0, tzinfo=timezone.utc)
TODAY = date(2026, 7, 27)
ZIPCODE = "98101"
SECRET = "weather-provider-secret"


class FakeResponse:
    def __init__(
        self,
        payload: Any = None,
        *,
        status_code: int = 200,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
        chunk_error: Exception | None = None,
    ) -> None:
        self.status_code = status_code
        self.headers = headers or {}
        self.body = (
            json.dumps(payload).encode("utf-8") if body is None else body
        )
        self.chunk_error = chunk_error
        self.closed = False

    def iter_content(self, chunk_size: int):
        if self.chunk_error is not None:
            raise self.chunk_error
        for offset in range(0, len(self.body), chunk_size):
            yield self.body[offset : offset + chunk_size]

    def close(self) -> None:
        self.closed = True


class FakeSession:
    def __init__(
        self,
        response: FakeResponse | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response


def raw_day(
    day: date,
    *,
    source: str = "fcst",
    icon: str = "clear-day",
    precipprob: float = 0,
    preciptype: list[str] | None = None,
    tempmin: float | None = 50,
    tempmax: float | None = 70,
) -> dict[str, Any]:
    return {
        "datetime": day.isoformat(),
        "source": source,
        "icon": icon,
        "precipprob": precipprob,
        "preciptype": preciptype,
        "tempmin": tempmin,
        "tempmax": tempmax,
    }


def payload(days: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "resolvedAddress": "Seattle, Washington, United States",
        "timezone": "America/Los_Angeles",
        "days": days,
        "latitude": 47.6,
        "longitude": -122.3,
        "description": "raw provider prose must not survive",
    }


def enabled_config(**overrides: Any) -> WeatherConfig:
    return WeatherConfig(enabled=True, **overrides)


def client_for(
    response: FakeResponse,
    *,
    config: WeatherConfig | None = None,
) -> tuple[VisualCrossingWeatherClient, FakeSession]:
    session = FakeSession(response)
    client = VisualCrossingWeatherClient(
        config or enabled_config(),
        SECRET,
        session=session,
        clock=lambda: NOW,
    )
    return client, session


def assert_failure(outcome: Any, code: str, retryable: bool) -> WeatherFailure:
    assert isinstance(outcome, WeatherFailure)
    assert outcome.ok is False
    assert outcome.code == code
    assert outcome.retryable is retryable
    return outcome


class TestWeatherConfig:
    def test_defaults_are_disabled_and_pinned(self) -> None:
        config = WeatherConfig()

        assert config.enabled is False
        assert config.provider == "visual_crossing"
        assert config.api_key_env == "WEATHER_API_KEY"
        assert config.timeout_seconds == 3.0
        assert config.max_forecast_horizon_days == 15
        assert config.max_range_days == 15
        assert config.model_dump().get("api_key") is None

    @pytest.mark.parametrize(
        "field,value",
        [
            ("base_url", "http://weather.visualcrossing.com/timeline"),
            ("base_url", "https://example.com/timeline"),
            (
                "base_url",
                "https://weather.visualcrossing.com/"
                "VisualCrossingWebServices/rest/services/timeline?x=1",
            ),
            ("api_key_env", "WEATHER-KEY"),
            ("api_key_env", "1WEATHER_KEY"),
            ("timeout_seconds", 0),
            ("timeout_seconds", True),
            ("timeout_seconds", "3.0"),
            ("timeout_seconds", float("inf")),
            ("timeout_seconds", float("nan")),
            ("max_forecast_horizon_days", 0),
            ("max_forecast_horizon_days", True),
            ("max_forecast_horizon_days", "15"),
            ("max_forecast_horizon_days", 16),
            ("max_range_days", 0),
            ("max_range_days", True),
            ("max_range_days", "15"),
            ("max_range_days", 16),
        ],
    )
    def test_invalid_config_fails_closed(self, field: str, value: Any) -> None:
        with pytest.raises(ValidationError):
            WeatherConfig(**{field: value})

    def test_disabled_and_missing_key_clients_return_typed_failures(self) -> None:
        disabled_session = FakeSession(error=AssertionError("must not call"))
        disabled = build_weather_client(
            WeatherConfig(),
            environ={},
            session=disabled_session,
        )
        assert_failure(
            disabled.get_forecast(WeatherRequest(location=ZIPCODE)),
            "weather_disabled",
            False,
        )
        assert disabled_session.calls == []

        missing_session = FakeSession(error=AssertionError("must not call"))
        missing = build_weather_client(
            enabled_config(),
            environ={},
            session=missing_session,
        )
        assert_failure(
            missing.get_forecast(WeatherRequest(location=ZIPCODE)),
            "weather_config_invalid",
            False,
        )
        assert missing_session.calls == []

    def test_enabled_builder_reads_only_the_named_environment_key(self) -> None:
        response = FakeResponse(payload([raw_day(TODAY, source="comb")]))
        session = FakeSession(response)
        client = build_weather_client(
            enabled_config(api_key_env="PRIVATE_WEATHER_KEY"),
            environ={
                "WEATHER_API_KEY": "wrong-key",
                "PRIVATE_WEATHER_KEY": SECRET,
            },
            session=session,
            clock=lambda: NOW,
        )

        outcome = client.get_forecast(WeatherRequest(location=ZIPCODE))

        assert isinstance(outcome, WeatherResult)
        assert session.calls[0]["params"]["key"] == SECRET


class TestWeatherRequest:
    @pytest.mark.parametrize(
        "location",
        [
            "98101",
            "Denver",
            "Cancún, Mexico",
            "Paris, France",
            "St. John's, Newfoundland and Labrador",
        ],
    )
    def test_accepts_named_places_and_postal_codes(self, location: str) -> None:
        assert WeatherRequest(location=location).location == location

    @pytest.mark.parametrize(
        "location",
        [
            "",
            " Denver",
            "Denver ",
            "Denver\nColorado",
            "Paris\x7fFrance",
            "x" * 257,
            98101,
        ],
    )
    def test_rejects_unsafe_or_unbounded_locations(
        self,
        location: Any,
    ) -> None:
        with pytest.raises(ValidationError):
            WeatherRequest(location=location)

    @pytest.mark.parametrize(
        "values",
        [
            {"date": "2026/07/27"},
            {"date": "2026-02-30"},
            {"date": "2026-07-27T00:00:00"},
            {"date": datetime(2026, 7, 27)},
            {"date": "2026-07-27", "start_date": "2026-07-27", "end_date": "2026-07-28"},
            {"start_date": "2026-07-27"},
            {"end_date": "2026-07-27"},
            {"start_date": "2026-07-28", "end_date": "2026-07-27"},
            {"start_date": "2026-07-27", "end_date": "2026-08-11"},
            {"city": "Seattle"},
        ],
    )
    def test_rejects_invalid_or_open_date_modes(self, values: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            WeatherRequest(location=ZIPCODE, **values)

    def test_accepts_today_single_and_inclusive_range_modes(self) -> None:
        assert WeatherRequest(location=ZIPCODE).explicit_window() is None
        assert WeatherRequest(
            location=ZIPCODE,
            date="2026-07-28",
        ).explicit_window() == (date(2026, 7, 28), date(2026, 7, 28))
        assert WeatherRequest(
            location=ZIPCODE,
            start_date="2026-07-28",
            end_date="2026-07-29",
        ).explicit_window() == (date(2026, 7, 28), date(2026, 7, 29))


class TestVisualCrossingSuccess:
    def test_today_request_is_one_local_combined_day(self) -> None:
        response = FakeResponse(
            payload(
                [
                    raw_day(
                        TODAY,
                        source="comb",
                        icon="rain",
                        precipprob=70,
                        preciptype=["rain"],
                        tempmin=57,
                        tempmax=66,
                    )
                ]
            )
        )
        client, session = client_for(response)

        outcome = client.get_forecast(WeatherRequest(location=ZIPCODE))

        assert isinstance(outcome, WeatherResult)
        assert outcome.fetched_at == NOW
        assert outcome.requested_window.start_date == TODAY
        assert outcome.requested_window.end_date == TODAY
        assert outcome.resolved_location == "Seattle, Washington, United States"
        assert outcome.timezone == "America/Los_Angeles"
        assert outcome.days[0].condition == "rain"
        assert outcome.days[0].precipitation_probability_pct == 70
        assert outcome.days[0].precipitation_types == ["rain"]
        assert outcome.attribution.url == VISUAL_CROSSING_ATTRIBUTION_URL
        assert session.calls == [
            {
                "url": (
                    "https://weather.visualcrossing.com/"
                    "VisualCrossingWebServices/rest/services/timeline/98101/today"
                ),
                "params": {
                    "unitGroup": "us",
                    "include": "days",
                    "elements": (
                        "datetime,tempmax,tempmin,precipprob,"
                        "preciptype,icon,source"
                    ),
                    "iconSet": "icons2",
                    "contentType": "json",
                    "key": SECRET,
                },
                "timeout": 3.0,
                "allow_redirects": False,
                "stream": True,
            }
        ]
        assert response.closed is True
        serialized = outcome.model_dump_json()
        for omitted in ("latitude", "longitude", "description", SECRET, ZIPCODE):
            assert omitted not in serialized

    def test_single_date_request_returns_exact_date(self) -> None:
        requested = TODAY + timedelta(days=3)
        client, _ = client_for(FakeResponse(payload([raw_day(requested)])))

        outcome = client.get_forecast(
            WeatherRequest(location=ZIPCODE, date=requested)
        )

        assert isinstance(outcome, WeatherResult)
        assert [item.date for item in outcome.days] == [requested]
        assert outcome.requested_window.start_date == requested

    def test_named_location_is_one_percent_encoded_path_segment(self) -> None:
        requested = TODAY + timedelta(days=1)
        client, session = client_for(
            FakeResponse(payload([raw_day(requested)]))
        )

        outcome = client.get_forecast(
            WeatherRequest(
                location="Cancún/Quintana Roo, Mexico",
                date=requested,
            )
        )

        assert isinstance(outcome, WeatherResult)
        assert session.calls[0]["url"].endswith(
            "/Canc%C3%BAn%2FQuintana%20Roo%2C%20Mexico/2026-07-28"
        )

    def test_inclusive_range_is_complete_and_ordered(self) -> None:
        requested_days = [TODAY + timedelta(days=offset) for offset in range(1, 4)]
        client, session = client_for(
            FakeResponse(payload([raw_day(day) for day in requested_days]))
        )

        outcome = client.get_forecast(
            WeatherRequest(
                location=ZIPCODE,
                start_date=requested_days[0],
                end_date=requested_days[-1],
            )
        )

        assert isinstance(outcome, WeatherResult)
        assert [item.date for item in outcome.days] == requested_days
        assert session.calls[0]["url"].endswith(
            "/98101/2026-07-28/2026-07-30"
        )

    @pytest.mark.parametrize(
        "icon,precipitation,expected",
        [
            ("clear-day", None, "clear"),
            ("partly-cloudy-day", None, "cloudy"),
            ("rain", ["rain"], "rain"),
            ("snow", ["snow"], "snow"),
            ("sleet", ["freezingrain"], "ice"),
            ("thunder-rain", ["rain"], "storm"),
            ("fog", None, "fog"),
            ("rain-and-snow", ["rain", "snow"], "mixed"),
            ("wind", None, "unknown"),
            ("unclear", None, "unknown"),
            ("brainstorm", None, "unknown"),
        ],
    )
    def test_normalizes_conditions_to_the_finite_domain(
        self,
        icon: str,
        precipitation: list[str] | None,
        expected: str,
    ) -> None:
        client, _ = client_for(
            FakeResponse(
                payload(
                    [
                        raw_day(
                            TODAY,
                            source="comb",
                            icon=icon,
                            preciptype=precipitation,
                        )
                    ]
                )
            )
        )

        outcome = client.get_forecast(WeatherRequest(location=ZIPCODE))

        assert isinstance(outcome, WeatherResult)
        assert outcome.days[0].condition == expected

    def test_normalizes_and_deduplicates_precipitation_types(self) -> None:
        client, _ = client_for(
            FakeResponse(
                payload(
                    [
                        raw_day(
                            TODAY,
                            source="comb",
                            preciptype=[
                                "ice",
                                "rain",
                                "freezingrain",
                                "rain",
                                "snow",
                            ],
                        )
                    ]
                )
            )
        )

        outcome = client.get_forecast(WeatherRequest(location=ZIPCODE))

        assert isinstance(outcome, WeatherResult)
        assert outcome.days[0].precipitation_types == [
            "rain",
            "snow",
            "freezing_rain",
            "ice",
        ]


class TestVisualCrossingFailures:
    @pytest.mark.parametrize(
        "status_code,code,retryable",
        [
            (302, "weather_response_invalid", False),
            (400, "weather_location_not_found", False),
            (401, "weather_auth_failed", False),
            (403, "weather_auth_failed", False),
            (404, "weather_response_invalid", False),
            (418, "weather_response_invalid", False),
            (429, "weather_rate_limited", True),
            (500, "weather_unavailable", True),
            (503, "weather_unavailable", True),
        ],
    )
    def test_maps_provider_status_without_reading_body(
        self,
        status_code: int,
        code: str,
        retryable: bool,
    ) -> None:
        response = FakeResponse(
            body=b"provider body with secret location detail",
            status_code=status_code,
        )
        client, _ = client_for(response)

        outcome = client.get_forecast(WeatherRequest(location=ZIPCODE))

        failure = assert_failure(outcome, code, retryable)
        assert "provider body" not in failure.model_dump_json()
        assert response.closed is True

    @pytest.mark.parametrize(
        "error,code",
        [
            (requests.Timeout("sensitive timeout details"), "weather_timeout"),
            (
                requests.ConnectionError("sensitive connection details"),
                "weather_unavailable",
            ),
        ],
    )
    def test_sanitizes_transport_failures(
        self,
        error: Exception,
        code: str,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        session = FakeSession(error=error)
        client = VisualCrossingWeatherClient(
            enabled_config(),
            SECRET,
            session=session,
            clock=lambda: NOW,
        )

        outcome = client.get_forecast(
            WeatherRequest(location=ZIPCODE, date=TODAY + timedelta(days=1))
        )

        failure = assert_failure(
            outcome,
            code,
            code in {"weather_timeout", "weather_unavailable"},
        )
        combined = failure.model_dump_json() + caplog.text
        for sensitive in (
            SECRET,
            ZIPCODE,
            "2026-07-28",
            "sensitive",
            "VisualCrossingWebServices",
        ):
            assert sensitive not in combined

    def test_stream_timeout_is_typed_and_response_is_closed(self) -> None:
        response = FakeResponse(
            payload({}),
            chunk_error=requests.Timeout("sensitive stream error"),
        )
        client, _ = client_for(response)

        outcome = client.get_forecast(WeatherRequest(location=ZIPCODE))

        assert_failure(outcome, "weather_timeout", True)
        assert response.closed is True

    @pytest.mark.parametrize(
        "response",
        [
            FakeResponse(
                body=b"{}",
                headers={"Content-Length": str(MAX_PROVIDER_RESPONSE_BYTES + 1)},
            ),
            FakeResponse(
                body=b"x" * (MAX_PROVIDER_RESPONSE_BYTES + 1),
            ),
            FakeResponse(body=b"{}", headers={"Content-Length": "not-a-number"}),
            FakeResponse(body=b""),
        ],
    )
    def test_rejects_empty_or_oversized_provider_bodies(
        self,
        response: FakeResponse,
    ) -> None:
        client, _ = client_for(response)

        outcome = client.get_forecast(WeatherRequest(location=ZIPCODE))

        assert_failure(outcome, "weather_response_invalid", False)
        assert response.closed is True

    @pytest.mark.parametrize(
        "body",
        [
            b"not-json",
            b"[]",
            json.dumps({"timezone": "America/Los_Angeles", "days": []}).encode(),
            json.dumps(
                {
                    "resolvedAddress": "Seattle",
                    "timezone": "Not/AZone",
                    "days": [],
                }
            ).encode(),
            json.dumps(
                {
                    "resolvedAddress": "Seattle",
                    "timezone": "America/Los_Angeles",
                    "days": "invalid",
                }
            ).encode(),
            json.dumps(
                {
                    "resolvedAddress": "Seattle\nWashington",
                    "timezone": "America/Los_Angeles",
                    "days": [],
                }
            ).encode(),
        ],
    )
    def test_rejects_malformed_provider_payloads(self, body: bytes) -> None:
        client, _ = client_for(FakeResponse(body=body))

        outcome = client.get_forecast(WeatherRequest(location=ZIPCODE))

        assert_failure(outcome, "weather_response_invalid", False)

    def test_rejects_pathologically_nested_json_without_raw_exception(self) -> None:
        body = (
            b'{"nested":'
            + (b"[" * 10_000)
            + b"0"
            + (b"]" * 10_000)
            + b"}"
        )
        client, _ = client_for(FakeResponse(body=body))

        outcome = client.get_forecast(WeatherRequest(location=ZIPCODE))

        assert_failure(outcome, "weather_response_invalid", False)

    def test_rejects_json_integer_over_the_interpreter_digit_limit(self) -> None:
        body = b'{"number":' + (b"9" * 10_000) + b"}"
        client, _ = client_for(FakeResponse(body=body))

        outcome = client.get_forecast(WeatherRequest(location=ZIPCODE))

        assert_failure(outcome, "weather_response_invalid", False)

    def test_rejects_provider_number_too_large_for_float_conversion(self) -> None:
        day = raw_day(TODAY, source="comb")
        day["precipprob"] = 10**4_000
        client, _ = client_for(FakeResponse(payload([day])))

        outcome = client.get_forecast(WeatherRequest(location=ZIPCODE))

        assert_failure(outcome, "weather_response_invalid", False)

    @pytest.mark.parametrize(
        "days",
        [
            [],
            [raw_day(TODAY), raw_day(TODAY + timedelta(days=1))],
            [raw_day(TODAY + timedelta(days=1))],
            [
                raw_day(TODAY + timedelta(days=1)),
                raw_day(TODAY),
            ],
        ],
    )
    def test_partial_extra_or_unexpected_days_fail_closed(
        self,
        days: list[dict[str, Any]],
    ) -> None:
        client, _ = client_for(FakeResponse(payload(days)))

        outcome = client.get_forecast(WeatherRequest(location=ZIPCODE))

        assert_failure(outcome, "weather_response_invalid", False)

    @pytest.mark.parametrize("source", ["obs", "histfcst", "stats"])
    def test_non_live_provider_sources_are_outside_horizon(
        self,
        source: str,
    ) -> None:
        client, _ = client_for(
            FakeResponse(payload([raw_day(TODAY, source=source)]))
        )

        outcome = client.get_forecast(WeatherRequest(location=ZIPCODE))

        assert_failure(outcome, "weather_outside_forecast_horizon", False)

    def test_combined_source_is_allowed_only_for_local_today(self) -> None:
        future = TODAY + timedelta(days=1)
        client, _ = client_for(
            FakeResponse(payload([raw_day(future, source="comb")]))
        )

        outcome = client.get_forecast(
            WeatherRequest(location=ZIPCODE, date=future)
        )

        assert_failure(outcome, "weather_outside_forecast_horizon", False)

    @pytest.mark.parametrize(
        "requested",
        [TODAY - timedelta(days=1), TODAY + timedelta(days=15)],
    )
    def test_dates_outside_today_through_today_plus_fourteen_fail(
        self,
        requested: date,
    ) -> None:
        client, _ = client_for(
            FakeResponse(payload([raw_day(requested, source="fcst")]))
        )

        outcome = client.get_forecast(
            WeatherRequest(location=ZIPCODE, date=requested)
        )

        assert_failure(outcome, "weather_outside_forecast_horizon", False)

    @pytest.mark.parametrize(
        "changes",
        [
            {"source": "new-source"},
            {"precipprob": -1},
            {"precipprob": 101},
            {"precipprob": float("nan")},
            {"preciptype": ["hail"]},
            {"preciptype": "rain"},
            {"tempmin": float("inf")},
            {"tempmin": -151},
            {"tempmax": 151},
            {"tempmin": 80, "tempmax": 70},
            {"icon": 4},
        ],
    )
    def test_invalid_daily_values_fail_closed(
        self,
        changes: dict[str, Any],
    ) -> None:
        day = raw_day(TODAY, source="comb")
        day.update(changes)
        client, _ = client_for(FakeResponse(payload([day])))

        outcome = client.get_forecast(WeatherRequest(location=ZIPCODE))

        assert_failure(outcome, "weather_response_invalid", False)

    def test_configured_range_cap_is_enforced(self) -> None:
        days = [TODAY + timedelta(days=offset) for offset in range(4)]
        client, session = client_for(
            FakeResponse(payload([raw_day(day, source="fcst") for day in days])),
            config=enabled_config(max_range_days=3),
        )

        outcome = client.get_forecast(
            WeatherRequest(
                location=ZIPCODE,
                start_date=days[0],
                end_date=days[-1],
            )
        )

        assert_failure(outcome, "weather_request_invalid", False)
        assert session.calls == []


def test_weather_failure_details_cannot_be_overridden() -> None:
    failure = WeatherFailure(
        code="weather_timeout",
        message=f"sensitive {SECRET} {ZIPCODE}",
        retryable=False,
    )

    assert failure.message == "The weather provider request timed out."
    assert failure.retryable is True
    assert SECRET not in failure.model_dump_json()
    assert ZIPCODE not in failure.model_dump_json()
