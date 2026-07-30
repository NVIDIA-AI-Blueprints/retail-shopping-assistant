# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from shared.weather_receipts import (
    DEFAULT_WEATHER_RECEIPT_TTL_SECONDS,
    MAX_WEATHER_RECEIPT_TTL_SECONDS,
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


FETCHED_AT = datetime(2026, 7, 30, 12, 0, tzinfo=timezone.utc)


def _evidence(
    *,
    start_date: date = date(2026, 8, 3),
    days: int = 1,
    resolved_location: str | None = "New York, NY, United States",
) -> WeatherReceiptEvidence:
    forecast_days = [
        WeatherReceiptDay(
            date=start_date + timedelta(days=offset),
            condition="rain" if offset == 0 else "cloudy",
            precipitation_probability_pct=70.0 if offset == 0 else 20.0,
            precipitation_types=["rain"] if offset == 0 else [],
            temperature_low_f=65.0,
            temperature_high_f=78.0,
        )
        for offset in range(days)
    ]
    return WeatherReceiptEvidence(
        fetched_at=FETCHED_AT,
        requested_window=WeatherReceiptWindow(
            start_date=start_date,
            end_date=start_date + timedelta(days=days - 1),
        ),
        resolved_location=resolved_location,
        days=forecast_days,
        attribution=WeatherReceiptAttribution(),
    )


def _stored_receipt(
    promotion: WeatherReceiptPromotion,
) -> WeatherForecastReceipt:
    scope_key = weather_scope_key(
        promotion.location_scope,
        promotion.evidence,
    )
    return WeatherForecastReceipt(
        receipt_id=weather_receipt_id(
            source_turn_id="turn-2",
            source_tool_call_id=promotion.source_tool_call_id,
            scope_key=scope_key,
            fetched_at=promotion.evidence.fetched_at,
        ),
        scope_key=scope_key,
        source_turn_id="turn-2",
        source_sequence=2,
        source_tool_call_id=promotion.source_tool_call_id,
        location_scope=promotion.location_scope,
        evidence=promotion.evidence,
        valid_until=(
            promotion.evidence.fetched_at
            + timedelta(seconds=promotion.ttl_seconds)
        ),
    )


def test_explicit_location_promotion_retains_exact_scope_and_default_ttl() -> None:
    promotion = WeatherReceiptPromotion(
        expected_projection_version=1,
        source_tool_call_id="weather-call-1",
        location_scope=ShopperLocationWeatherScope(
            location="NYC",
            location_query="NYC, NY",
        ),
        evidence=_evidence(),
    )
    receipt = _stored_receipt(promotion)

    assert promotion.ttl_seconds == DEFAULT_WEATHER_RECEIPT_TTL_SECONDS
    assert receipt.location_scope.model_dump(exclude_none=True) == {
        "kind": "shopper_provided_location",
        "location": "NYC",
        "location_query": "NYC, NY",
    }
    assert receipt.valid_until == FETCHED_AT + timedelta(hours=1)
    assert receipt.receipt_id.startswith("sha256:")
    assert receipt.scope_key.startswith("sha256:")


def test_saved_area_scope_contains_no_zip_or_provider_resolution() -> None:
    promotion = WeatherReceiptPromotion(
        expected_projection_version=1,
        source_tool_call_id="weather-call-1",
        location_scope=SavedAreaWeatherScope(),
        evidence=_evidence(resolved_location=None),
    )

    payload = _stored_receipt(promotion).model_dump_json()

    assert "confirmed_saved_zip" in payload
    assert "10001" not in payload
    assert "resolved_location" not in _stored_receipt(
        promotion
    ).model_dump(mode="json", exclude_none=True)["evidence"]


@pytest.mark.parametrize(
    ("scope", "resolved_location"),
    [
        (SavedAreaWeatherScope(), "Seattle, WA"),
        (ShopperLocationWeatherScope(location="NYC"), None),
    ],
)
def test_location_scope_and_provider_resolution_must_match(
    scope: SavedAreaWeatherScope | ShopperLocationWeatherScope,
    resolved_location: str | None,
) -> None:
    with pytest.raises(ValidationError):
        WeatherReceiptPromotion(
            expected_projection_version=1,
            source_tool_call_id="weather-call-1",
            location_scope=scope,
            evidence=_evidence(resolved_location=resolved_location),
        )


def test_scope_key_uses_exact_input_scope_and_dates_not_provider_assumption() -> None:
    evidence = _evidence()
    same_dates_other_resolution = evidence.model_copy(
        update={"resolved_location": "Manhattan, NY, United States"}
    )
    scope = ShopperLocationWeatherScope(
        location="NYC",
        location_query="NYC, NY",
    )

    assert weather_scope_key(scope, evidence) == weather_scope_key(
        scope,
        same_dates_other_resolution,
    )
    assert weather_scope_key(scope, evidence) != weather_scope_key(
        ShopperLocationWeatherScope(
            location="NYC",
            location_query="NYC, NJ",
        ),
        evidence,
    )
    assert weather_scope_key(scope, evidence) != weather_scope_key(
        scope,
        _evidence(start_date=date(2026, 8, 4)),
    )


def test_receipt_rejects_tampered_scope_or_identity() -> None:
    promotion = WeatherReceiptPromotion(
        expected_projection_version=1,
        source_tool_call_id="weather-call-1",
        location_scope=ShopperLocationWeatherScope(location="Cancun"),
        evidence=_evidence(resolved_location="Cancún, Quintana Roo, Mexico"),
    )
    receipt = _stored_receipt(promotion)

    with pytest.raises(ValidationError):
        WeatherForecastReceipt.model_validate_json(
            json.dumps(
                {
                **receipt.model_dump(mode="json"),
                "scope_key": "sha256:" + ("0" * 64),
                }
            )
        )
    with pytest.raises(ValidationError):
        WeatherForecastReceipt.model_validate_json(
            json.dumps(
                {
                **receipt.model_dump(mode="json"),
                "receipt_id": "sha256:" + ("0" * 64),
                }
            )
        )


def test_receipt_rejects_invalid_ttl_and_validity_boundary() -> None:
    kwargs = {
        "expected_projection_version": 1,
        "source_tool_call_id": "weather-call-1",
        "location_scope": ShopperLocationWeatherScope(location="Paris, France"),
        "evidence": _evidence(resolved_location="Paris, Île-de-France, France"),
    }

    with pytest.raises(ValidationError):
        WeatherReceiptPromotion(**kwargs, ttl_seconds=0)
    with pytest.raises(ValidationError):
        WeatherReceiptPromotion(
            **kwargs,
            ttl_seconds=MAX_WEATHER_RECEIPT_TTL_SECONDS + 1,
        )

    promotion = WeatherReceiptPromotion(**kwargs)
    receipt = _stored_receipt(promotion)
    with pytest.raises(ValidationError):
        WeatherForecastReceipt.model_validate_json(
            json.dumps(
                {
                **receipt.model_dump(mode="json"),
                "valid_until": (
                    FETCHED_AT
                    + timedelta(seconds=MAX_WEATHER_RECEIPT_TTL_SECONDS + 1)
                ).isoformat(),
                }
            )
        )


def test_evidence_rejects_non_utc_noncontiguous_or_malformed_daily_facts() -> None:
    with pytest.raises(ValidationError):
        WeatherReceiptEvidence(
            fetched_at=datetime(2026, 7, 30, 12, 0),
            requested_window=WeatherReceiptWindow(
                start_date=date(2026, 8, 3),
                end_date=date(2026, 8, 3),
            ),
            resolved_location="New York, NY",
            days=[
                WeatherReceiptDay(
                    date=date(2026, 8, 3),
                    condition="clear",
                    precipitation_probability_pct=0.0,
                )
            ],
            attribution=WeatherReceiptAttribution(),
        )

    with pytest.raises(ValidationError):
        WeatherReceiptEvidence.model_validate_json(
            json.dumps(
                {
                    **_evidence(days=2).model_dump(mode="json"),
                "days": [
                    {
                        "date": "2026-08-03",
                        "condition": "clear",
                        "precipitation_probability_pct": 0.0,
                        "precipitation_types": [],
                    },
                    {
                        "date": "2026-08-05",
                        "condition": "clear",
                        "precipitation_probability_pct": 0.0,
                        "precipitation_types": [],
                    },
                ],
                }
            )
        )

    with pytest.raises(ValidationError):
        WeatherReceiptDay(
            date=date(2026, 8, 3),
            condition="rain",
            precipitation_probability_pct=50.0,
            precipitation_types=["rain", "rain"],
        )


def test_next_week_provenance_must_match_exact_window_and_weekday() -> None:
    monday = date(2026, 8, 3)
    weekly = _evidence(start_date=monday, days=7).model_copy(
        update={"relative_date": "next_week"}
    )
    friday = _evidence(start_date=date(2026, 8, 7)).model_copy(
        update={"relative_date": "next_week", "weekday": "friday"}
    )

    assert WeatherReceiptEvidence.model_validate_json(
        weekly.model_dump_json()
    ).relative_date == "next_week"
    assert WeatherReceiptEvidence.model_validate_json(
        friday.model_dump_json()
    ).weekday == "friday"

    with pytest.raises(ValidationError):
        WeatherReceiptEvidence.model_validate_json(
            json.dumps(
                {
                    **_evidence(
                        start_date=monday,
                        days=7,
                    ).model_dump(mode="json"),
                    "relative_date": "next_week",
                    "weekday": "friday",
                }
            )
        )
