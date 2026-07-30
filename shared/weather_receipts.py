# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Closed cross-service contract for bounded durable weather evidence."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import date as CalendarDate
from datetime import datetime, timedelta
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_validator,
    model_validator,
)


WEATHER_RECEIPT_TYPE = "weather_forecast.v1"
WEATHER_TOOL_NAME = "get_weather_forecast_tool"
VISUAL_CROSSING_ATTRIBUTION_LABEL = "Weather Data Provided by Visual Crossing"
VISUAL_CROSSING_ATTRIBUTION_URL = "https://www.visualcrossing.com/"
MAX_ACTIVE_WEATHER_RECEIPTS = 4
DEFAULT_WEATHER_RECEIPT_TTL_SECONDS = 3_600
MAX_WEATHER_RECEIPT_TTL_SECONDS = 21_600
MAX_WEATHER_DAYS = 15

WeatherCondition = Literal[
    "clear",
    "cloudy",
    "rain",
    "snow",
    "ice",
    "storm",
    "fog",
    "mixed",
    "unknown",
]
PrecipitationType = Literal["rain", "snow", "freezing_rain", "ice"]
RelativeWeekday = Literal[
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
]

_CONTROL_CHARACTER_RE = re.compile(r"[\x00-\x1f\x7f]")
_ISO_DATE_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
_DIGEST_RE = r"^sha256:[0-9a-f]{64}$"
_WEEKDAY_OFFSETS: dict[RelativeWeekday, int] = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


class _ReceiptModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


def _validate_single_line(value: str, *, field_name: str) -> str:
    if (
        value != value.strip()
        or _CONTROL_CHARACTER_RE.search(value) is not None
    ):
        raise ValueError(f"{field_name} must be trimmed and single-line")
    return value


def _require_utc(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be timezone-aware UTC")
    return value


def _parse_calendar_date(value: Any) -> CalendarDate:
    if isinstance(value, datetime):
        raise ValueError("weather receipt dates must not include a time")
    if isinstance(value, CalendarDate):
        return value
    if not isinstance(value, str) or not _ISO_DATE_RE.fullmatch(value):
        raise ValueError("weather receipt dates must use YYYY-MM-DD")
    try:
        return CalendarDate.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("weather receipt dates must be valid") from exc


def _parse_datetime(value: Any, *, field_name: str) -> datetime:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be an ISO datetime")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO datetime") from exc


class SavedAreaWeatherScope(_ReceiptModel):
    """Conversation-bound saved-area scope that never stores the saved ZIP."""

    kind: Literal["confirmed_saved_zip"] = "confirmed_saved_zip"


class ShopperLocationWeatherScope(_ReceiptModel):
    """Exact shopper-authored location and optional provider qualifier."""

    kind: Literal["shopper_provided_location"] = "shopper_provided_location"
    location: str = Field(..., min_length=1, max_length=256)
    location_query: str | None = Field(default=None, min_length=1, max_length=256)

    @field_validator("location", "location_query")
    @classmethod
    def _validate_location_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_single_line(value, field_name="weather location")


WeatherLocationScope = Annotated[
    SavedAreaWeatherScope | ShopperLocationWeatherScope,
    Field(discriminator="kind"),
]


class WeatherReceiptWindow(_ReceiptModel):
    start_date: CalendarDate
    end_date: CalendarDate

    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def _parse_dates(cls, value: Any) -> CalendarDate:
        return _parse_calendar_date(value)

    @model_validator(mode="after")
    def _validate_window(self) -> "WeatherReceiptWindow":
        day_count = (self.end_date - self.start_date).days + 1
        if not 1 <= day_count <= MAX_WEATHER_DAYS:
            raise ValueError("weather receipt window must contain 1 to 15 days")
        return self


class WeatherReceiptAttribution(_ReceiptModel):
    label: Literal[
        "Weather Data Provided by Visual Crossing"
    ] = VISUAL_CROSSING_ATTRIBUTION_LABEL
    url: Literal[
        "https://www.visualcrossing.com/"
    ] = VISUAL_CROSSING_ATTRIBUTION_URL


class WeatherReceiptDay(_ReceiptModel):
    date: CalendarDate
    condition: WeatherCondition
    precipitation_probability_pct: float = Field(..., ge=0, le=100)
    precipitation_types: list[PrecipitationType] = Field(
        default_factory=list,
        max_length=4,
    )
    temperature_low_f: float | None = Field(default=None, ge=-150, le=150)
    temperature_high_f: float | None = Field(default=None, ge=-150, le=150)

    @field_validator("date", mode="before")
    @classmethod
    def _parse_date(cls, value: Any) -> CalendarDate:
        return _parse_calendar_date(value)

    @field_validator(
        "precipitation_probability_pct",
        "temperature_low_f",
        "temperature_high_f",
    )
    @classmethod
    def _require_finite_number(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("weather receipt numbers must be finite")
        return value

    @field_validator("precipitation_types")
    @classmethod
    def _require_unique_precipitation_types(
        cls,
        value: list[PrecipitationType],
    ) -> list[PrecipitationType]:
        if len(value) != len(set(value)):
            raise ValueError("weather receipt precipitation types must be unique")
        return value

    @model_validator(mode="after")
    def _validate_temperature_order(self) -> "WeatherReceiptDay":
        if (
            self.temperature_low_f is not None
            and self.temperature_high_f is not None
            and self.temperature_low_f > self.temperature_high_f
        ):
            raise ValueError("weather receipt low temperature exceeds high")
        return self


class WeatherReceiptEvidence(_ReceiptModel):
    """Successful normalized forecast facts eligible for durable promotion."""

    ok: Literal[True] = True
    provider: Literal["visual_crossing"] = "visual_crossing"
    fetched_at: datetime
    requested_window: WeatherReceiptWindow
    relative_date: Literal["next_week"] | None = None
    weekday: RelativeWeekday | None = None
    resolved_location: str | None = Field(
        default=None,
        min_length=1,
        max_length=512,
    )
    days: list[WeatherReceiptDay] = Field(
        ...,
        min_length=1,
        max_length=MAX_WEATHER_DAYS,
    )
    attribution: WeatherReceiptAttribution

    @field_validator("fetched_at", mode="before")
    @classmethod
    def _parse_fetched_at(cls, value: Any) -> datetime:
        return _parse_datetime(value, field_name="weather fetched_at")

    @field_validator("fetched_at")
    @classmethod
    def _validate_fetched_at(cls, value: datetime) -> datetime:
        return _require_utc(value, field_name="weather fetched_at")

    @field_validator("resolved_location")
    @classmethod
    def _validate_resolved_location(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_single_line(
            value,
            field_name="weather resolved_location",
        )

    @model_validator(mode="after")
    def _validate_evidence(self) -> "WeatherReceiptEvidence":
        start = self.requested_window.start_date
        end = self.requested_window.end_date
        day_count = (end - start).days + 1
        expected_dates = [
            start + timedelta(days=offset)
            for offset in range(day_count)
        ]
        if [day.date for day in self.days] != expected_dates:
            raise ValueError("weather receipt days must exactly cover the window")
        if self.weekday is not None and self.relative_date != "next_week":
            raise ValueError("weather receipt weekday requires next_week")
        if self.relative_date == "next_week":
            expected_count = 1 if self.weekday is not None else 7
            if day_count != expected_count:
                raise ValueError("weather next_week receipt has the wrong window")
            if (
                self.weekday is not None
                and start.weekday() != _WEEKDAY_OFFSETS[self.weekday]
            ):
                raise ValueError("weather receipt weekday does not match its date")
        return self


class WeatherReceiptPromotion(_ReceiptModel):
    """One current-turn success offered for atomic projection promotion."""

    expected_projection_version: int = Field(..., ge=0)
    source_tool_call_id: str = Field(..., min_length=1, max_length=256)
    location_scope: WeatherLocationScope
    evidence: WeatherReceiptEvidence
    ttl_seconds: StrictInt = Field(
        default=DEFAULT_WEATHER_RECEIPT_TTL_SECONDS,
        ge=1,
        le=MAX_WEATHER_RECEIPT_TTL_SECONDS,
    )

    @field_validator("source_tool_call_id")
    @classmethod
    def _validate_tool_call_id(cls, value: str) -> str:
        return _validate_single_line(
            value,
            field_name="weather source_tool_call_id",
        )

    @model_validator(mode="after")
    def _validate_location_projection(self) -> "WeatherReceiptPromotion":
        _validate_scope_evidence(self.location_scope, self.evidence)
        return self


class WeatherForecastReceipt(_ReceiptModel):
    """Memory-stamped active receipt returned in a conversation projection."""

    receipt_type: Literal["weather_forecast.v1"] = WEATHER_RECEIPT_TYPE
    receipt_id: str = Field(..., pattern=_DIGEST_RE)
    scope_key: str = Field(..., pattern=_DIGEST_RE)
    source_turn_id: str = Field(..., min_length=1, max_length=256)
    source_sequence: int = Field(..., ge=1)
    source_tool: Literal["get_weather_forecast_tool"] = WEATHER_TOOL_NAME
    source_tool_call_id: str = Field(..., min_length=1, max_length=256)
    location_scope: WeatherLocationScope
    evidence: WeatherReceiptEvidence
    valid_until: datetime

    @field_validator("source_turn_id", "source_tool_call_id")
    @classmethod
    def _validate_source_text(cls, value: str) -> str:
        return _validate_single_line(value, field_name="weather receipt source")

    @field_validator("valid_until", mode="before")
    @classmethod
    def _parse_valid_until(cls, value: Any) -> datetime:
        return _parse_datetime(value, field_name="weather valid_until")

    @field_validator("valid_until")
    @classmethod
    def _validate_valid_until(cls, value: datetime) -> datetime:
        return _require_utc(value, field_name="weather valid_until")

    @model_validator(mode="after")
    def _validate_receipt(self) -> "WeatherForecastReceipt":
        _validate_scope_evidence(self.location_scope, self.evidence)
        if not (
            self.evidence.fetched_at
            < self.valid_until
            <= self.evidence.fetched_at
            + timedelta(seconds=MAX_WEATHER_RECEIPT_TTL_SECONDS)
        ):
            raise ValueError("weather receipt validity boundary is invalid")
        expected_scope_key = weather_scope_key(
            self.location_scope,
            self.evidence,
        )
        if self.scope_key != expected_scope_key:
            raise ValueError("weather receipt scope key is invalid")
        expected_receipt_id = weather_receipt_id(
            source_turn_id=self.source_turn_id,
            source_tool_call_id=self.source_tool_call_id,
            scope_key=self.scope_key,
            fetched_at=self.evidence.fetched_at,
        )
        if self.receipt_id != expected_receipt_id:
            raise ValueError("weather receipt identifier is invalid")
        return self


def _validate_scope_evidence(
    scope: SavedAreaWeatherScope | ShopperLocationWeatherScope,
    evidence: WeatherReceiptEvidence,
) -> None:
    if isinstance(scope, SavedAreaWeatherScope):
        if evidence.resolved_location is not None:
            raise ValueError(
                "saved-area weather evidence must omit resolved_location"
            )
    elif evidence.resolved_location is None:
        raise ValueError(
            "shopper-location weather evidence requires resolved_location"
        )


def weather_scope_key(
    location_scope: SavedAreaWeatherScope | ShopperLocationWeatherScope,
    evidence: WeatherReceiptEvidence,
) -> str:
    """Return the exact conversation-local location/date scope identity."""

    payload = {
        "receipt_type": WEATHER_RECEIPT_TYPE,
        "location_scope": location_scope.model_dump(
            mode="json",
            exclude_none=True,
        ),
        "requested_start_date": (
            evidence.requested_window.start_date.isoformat()
        ),
        "requested_end_date": evidence.requested_window.end_date.isoformat(),
    }
    return _sha256_digest(payload)


def weather_receipt_id(
    *,
    source_turn_id: str,
    source_tool_call_id: str,
    scope_key: str,
    fetched_at: datetime,
) -> str:
    """Return the stable identity of one promoted provider observation."""

    payload = {
        "receipt_type": WEATHER_RECEIPT_TYPE,
        "source_turn_id": source_turn_id,
        "source_tool_call_id": source_tool_call_id,
        "scope_key": scope_key,
        "fetched_at": fetched_at.isoformat(),
    }
    return _sha256_digest(payload)


def _sha256_digest(value: Any) -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"
