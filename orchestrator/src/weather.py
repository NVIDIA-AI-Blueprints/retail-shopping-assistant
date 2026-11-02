# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Daily weather forecast contracts and the Visual Crossing adapter."""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from datetime import date as CalendarDate
from typing import Any, Literal, Protocol
from urllib.parse import quote
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    field_validator,
    model_validator,
)

VISUAL_CROSSING_BASE_URL = (
    "https://weather.visualcrossing.com/"
    "VisualCrossingWebServices/rest/services/timeline"
)
VISUAL_CROSSING_ATTRIBUTION_URL = "https://www.visualcrossing.com/"
MAX_PROVIDER_RESPONSE_BYTES = 256 * 1024
MAX_WEATHER_DAYS = 15

WeatherCondition = Literal[
    "clear", "cloudy", "rain", "snow", "ice", "storm", "fog", "mixed", "unknown"
]
PrecipitationType = Literal["rain", "snow", "freezing_rain", "ice"]
WeatherFailureCode = Literal[
    "weather_disabled",
    "weather_config_invalid",
    "weather_request_invalid",
    "weather_location_not_found",
    "weather_outside_forecast_horizon",
    "weather_auth_failed",
    "weather_rate_limited",
    "weather_timeout",
    "weather_unavailable",
    "weather_response_invalid",
]

_ISO_DATE_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$", flags=re.ASCII)
#: code -> (stable message, retryable). A failure never carries request or
#: provider text, so nothing sensitive can travel in one.
_FAILURE_DETAILS: dict[str, tuple[str, bool]] = {
    "weather_disabled": ("Weather lookup is disabled.", False),
    "weather_config_invalid": ("Weather configuration is incomplete or invalid.", False),
    "weather_request_invalid": ("The weather request is invalid.", False),
    "weather_location_not_found": ("The place could not be resolved.", False),
    "weather_outside_forecast_horizon": (
        "The requested date is outside the live forecast horizon.",
        False,
    ),
    "weather_auth_failed": ("The weather provider rejected its credentials.", False),
    "weather_rate_limited": ("The weather provider rate limit was reached.", True),
    "weather_timeout": ("The weather provider request timed out.", True),
    "weather_unavailable": ("The weather provider is temporarily unavailable.", True),
    "weather_response_invalid": ("The weather provider response was invalid.", False),
}
_PRECIPITATION: dict[str, PrecipitationType] = {
    "rain": "rain",
    "snow": "snow",
    "freezingrain": "freezing_rain",
    "ice": "ice",
}
_ICON_CONDITIONS: dict[str, WeatherCondition] = {
    "snow": "snow",
    "snow-showers-day": "snow",
    "snow-showers-night": "snow",
    "thunder-rain": "storm",
    "thunder-showers-day": "storm",
    "thunder-showers-night": "storm",
    "rain": "rain",
    "showers-day": "rain",
    "showers-night": "rain",
    "fog": "fog",
    "cloudy": "cloudy",
    "partly-cloudy-day": "cloudy",
    "partly-cloudy-night": "cloudy",
    "clear-day": "clear",
    "clear-night": "clear",
}


class WeatherConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    enabled: StrictBool = False
    provider: Literal["visual_crossing"] = "visual_crossing"
    base_url: str = VISUAL_CROSSING_BASE_URL
    api_key_env: str = Field(
        default="WEATHER_API_KEY", pattern=r"^[A-Za-z_][A-Za-z0-9_]*$"
    )
    timeout_seconds: StrictFloat = 3.0
    max_forecast_horizon_days: StrictInt = Field(default=MAX_WEATHER_DAYS, ge=1, le=MAX_WEATHER_DAYS)
    max_range_days: StrictInt = Field(default=MAX_WEATHER_DAYS, ge=1, le=MAX_WEATHER_DAYS)

    @field_validator("base_url")
    @classmethod
    def _pinned_base_url(cls, value: str) -> str:
        if value != value.strip() or value.rstrip("/") != VISUAL_CROSSING_BASE_URL:
            raise ValueError("weather base_url must be the pinned Visual Crossing endpoint")
        return VISUAL_CROSSING_BASE_URL

    @field_validator("timeout_seconds")
    @classmethod
    def _positive_timeout(cls, value: float) -> float:
        if not math.isfinite(value) or value <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        return value


class WeatherRequest(BaseModel):
    """A place a shopper would say, plus today, one date or an inclusive range.

    The provider resolves the place; the server never geocodes. Bounded and
    single-line so prose or an injected newline cannot reach the provider URL.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    location: str
    date: CalendarDate | None = None
    start_date: CalendarDate | None = None
    end_date: CalendarDate | None = None

    @field_validator("location", mode="before")
    @classmethod
    def _a_place(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("location must be a place name or postal code")
        cleaned = " ".join(value.split())
        if not 1 <= len(cleaned) <= 120:
            raise ValueError("location must be 1 to 120 characters")
        if any(ord(character) < 32 for character in cleaned):
            raise ValueError("location must be a single line")
        return cleaned

    @field_validator("date", "start_date", "end_date", mode="before")
    @classmethod
    def _iso_date(cls, value: Any) -> CalendarDate | None:
        if value is None or (
            isinstance(value, CalendarDate) and not isinstance(value, datetime)
        ):
            return value
        if not isinstance(value, str) or not _ISO_DATE_RE.fullmatch(value):
            raise ValueError("weather dates must use YYYY-MM-DD")
        try:
            return CalendarDate.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("weather dates must be valid calendar dates") from exc

    @model_validator(mode="after")
    def _one_date_mode(self) -> WeatherRequest:
        if self.date is not None and (self.start_date or self.end_date):
            raise ValueError("date is mutually exclusive with a date range")
        if (self.start_date is None) != (self.end_date is None):
            raise ValueError("start_date and end_date must be supplied together")
        if self.start_date and self.end_date:
            if self.start_date > self.end_date:
                raise ValueError("start_date must be on or before end_date")
            if (self.end_date - self.start_date).days + 1 > MAX_WEATHER_DAYS:
                raise ValueError("date ranges may contain at most 15 days")
        return self

    def explicit_window(self) -> tuple[CalendarDate, CalendarDate] | None:
        """The requested window, or ``None`` for the location's today."""

        if self.date is not None:
            return self.date, self.date
        if self.start_date is not None and self.end_date is not None:
            return self.start_date, self.end_date
        return None


class WeatherRequestedWindow(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    start_date: CalendarDate
    end_date: CalendarDate


class WeatherAttribution(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    label: str
    url: str


class WeatherDay(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    date: CalendarDate
    condition: WeatherCondition
    #: Optional: the provider often sends null here on sound forecasts.
    precipitation_probability_pct: float | None = Field(default=None, ge=0, le=100)
    precipitation_types: list[PrecipitationType]
    temperature_low_f: float | None = None
    temperature_high_f: float | None = None


class WeatherResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    ok: Literal[True] = True
    provider: Literal["visual_crossing"] = "visual_crossing"
    fetched_at: datetime
    requested_window: WeatherRequestedWindow
    resolved_location: str = Field(min_length=1, max_length=512)
    timezone: str = Field(min_length=1, max_length=128)
    days: list[WeatherDay] = Field(min_length=1, max_length=MAX_WEATHER_DAYS)
    attribution: WeatherAttribution


class WeatherFailure(BaseModel):
    """A failure whose message and retryability are fixed by its code."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    ok: Literal[False] = False
    code: WeatherFailureCode
    message: str
    retryable: bool

    @model_validator(mode="before")
    @classmethod
    def _stable_details(cls, value: Any) -> Any:
        if isinstance(value, dict) and value.get("code") in _FAILURE_DETAILS:
            message, retryable = _FAILURE_DETAILS[value["code"]]
            value = {**value, "message": message, "retryable": retryable}
        return value


WeatherOutcome = WeatherResult | WeatherFailure


class _Invalid(Exception):
    """Internal: abandon normalization with this failure code."""

    def __init__(self, code: WeatherFailureCode = "weather_response_invalid") -> None:
        self.code = code


def weather_failure(code: WeatherFailureCode) -> WeatherFailure:
    return WeatherFailure(code=code)


class WeatherClient(Protocol):
    def get_forecast(self, request: WeatherRequest) -> WeatherOutcome: ...


class _StaticFailureWeatherClient:
    def __init__(self, failure: WeatherFailure) -> None:
        self._failure = failure

    def get_forecast(self, request: WeatherRequest) -> WeatherOutcome:
        return self._failure.model_copy(deep=True)


class VisualCrossingWeatherClient:
    def __init__(
        self,
        config: WeatherConfig,
        api_key: str,
        *,
        session: Any = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not config.enabled or not api_key or api_key != api_key.strip():
            raise ValueError("weather client configuration is invalid")
        self._config = config
        self._api_key = api_key
        self._session = session or requests.Session()
        self._clock = clock or (lambda: datetime.now(UTC))

    def get_forecast(self, request: WeatherRequest) -> WeatherOutcome:
        window = request.explicit_window()
        if window is not None:
            if (window[1] - window[0]).days + 1 > self._config.max_range_days:
                return weather_failure("weather_request_invalid")
            # Reject windows no timezone could bring into range before paying
            # for a call; the exact check needs the provider's timezone.
            utc_today = self._clock().astimezone(UTC).date()
            if window[1] < utc_today - timedelta(days=1) or window[0] > utc_today + timedelta(
                days=self._config.max_forecast_horizon_days
            ):
                return weather_failure("weather_outside_forecast_horizon")

        path = [quote(request.location, safe="")]
        if window is None:
            path.append("today")
        else:
            path.extend(dict.fromkeys(day.isoformat() for day in window))
        response = None
        try:
            response = self._session.get(
                f"{self._config.base_url}/{'/'.join(path)}",
                params={
                    "unitGroup": "us",
                    "include": "days",
                    "elements": "datetime,tempmax,tempmin,precipprob,preciptype,icon,source",
                    "iconSet": "icons2",
                    "contentType": "json",
                    "key": self._api_key,
                },
                timeout=self._config.timeout_seconds,
                allow_redirects=False,
                stream=True,
            )
            status = response.status_code
            if status != 200:
                return weather_failure(_status_failure(status))
            return self._normalize(request, _read_bounded_body(response))
        except _Invalid as invalid:
            return weather_failure(invalid.code)
        except requests.Timeout:
            return weather_failure("weather_timeout")
        except requests.RequestException:
            return weather_failure("weather_unavailable")
        finally:
            if response is not None:
                # A close error may carry the secret-bearing URL.
                with contextlib.suppress(Exception):
                    response.close()

    def _normalize(self, request: WeatherRequest, body: bytes) -> WeatherResult:
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError, RecursionError) as exc:
            raise _Invalid() from exc
        if not isinstance(payload, dict):
            raise _Invalid()
        resolved = payload.get("resolvedAddress")
        timezone_name = payload.get("timezone")
        raw_days = payload.get("days")
        if (
            not isinstance(resolved, str)
            or not resolved.strip()
            or resolved != resolved.strip()
            or len(resolved) > 512
            or not isinstance(timezone_name, str)
            or not 0 < len(timezone_name) <= 128
            or not isinstance(raw_days, list)
        ):
            raise _Invalid()
        fetched_at = self._clock()
        if fetched_at.tzinfo is None or fetched_at.utcoffset() is None:
            raise _Invalid()
        try:
            fetched_at = fetched_at.astimezone(UTC)
            local_today = fetched_at.astimezone(ZoneInfo(timezone_name)).date()
        except (ZoneInfoNotFoundError, ValueError, OverflowError, OSError) as exc:
            raise _Invalid() from exc

        start, end = request.explicit_window() or (local_today, local_today)
        last_day = local_today + timedelta(days=self._config.max_forecast_horizon_days - 1)
        if start < local_today or end > last_day:
            raise _Invalid("weather_outside_forecast_horizon")
        length = (end - start).days + 1
        if length > self._config.max_range_days:
            raise _Invalid("weather_request_invalid")
        if len(raw_days) != length:
            raise _Invalid()
        days = [
            _normalize_day(raw, start + timedelta(days=offset), local_today)
            for offset, raw in enumerate(raw_days)
        ]
        return WeatherResult(
            fetched_at=fetched_at,
            requested_window=WeatherRequestedWindow(start_date=start, end_date=end),
            resolved_location=resolved,
            timezone=timezone_name,
            days=days,
            attribution=WeatherAttribution(
                label="Weather Data Provided by Visual Crossing",
                url=VISUAL_CROSSING_ATTRIBUTION_URL,
            ),
        )


def build_weather_client(
    config: WeatherConfig,
    *,
    environ: Mapping[str, str] | None = None,
    session: Any = None,
    clock: Callable[[], datetime] | None = None,
) -> WeatherClient:
    """A live client, or one that returns the typed disabled/config failure."""

    if not config.enabled:
        return _StaticFailureWeatherClient(weather_failure("weather_disabled"))
    api_key = (os.environ if environ is None else environ).get(config.api_key_env, "")
    if not api_key or api_key != api_key.strip():
        return _StaticFailureWeatherClient(weather_failure("weather_config_invalid"))
    return VisualCrossingWeatherClient(config, api_key, session=session, clock=clock)


def _status_failure(status_code: int) -> WeatherFailureCode:
    if status_code == 400:
        return "weather_location_not_found"
    if status_code in {401, 403}:
        return "weather_auth_failed"
    if status_code == 429:
        return "weather_rate_limited"
    if status_code >= 500:
        return "weather_unavailable"
    return "weather_response_invalid"


def _read_bounded_body(response: Any) -> bytes:
    raw_length = response.headers.get("Content-Length")
    if raw_length is not None:
        try:
            length = int(raw_length)
        except (TypeError, ValueError) as exc:
            raise _Invalid() from exc
        if not 0 <= length <= MAX_PROVIDER_RESPONSE_BYTES:
            raise _Invalid()
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_content(chunk_size=8192):
        if not isinstance(chunk, bytes):
            raise _Invalid()
        total += len(chunk)
        if total > MAX_PROVIDER_RESPONSE_BYTES:
            raise _Invalid()
        chunks.append(chunk)
    if not total:
        raise _Invalid()
    return b"".join(chunks)


def _normalize_day(raw: Any, expected: CalendarDate, local_today: CalendarDate) -> WeatherDay:
    if not isinstance(raw, dict) or raw.get("datetime") != expected.isoformat():
        raise _Invalid()
    source = raw.get("source")
    if source == "comb":
        if expected != local_today:
            raise _Invalid("weather_outside_forecast_horizon")
    elif source in {"obs", "histfcst", "stats"}:
        raise _Invalid("weather_outside_forecast_horizon")
    elif source != "fcst":
        raise _Invalid()

    probability = _optional_number(raw.get("precipprob"))
    if probability is not None and not 0 <= probability <= 100:
        raise _Invalid()
    low = _optional_number(raw.get("tempmin"))
    high = _optional_number(raw.get("tempmax"))
    if low is not None and high is not None and low > high:
        raise _Invalid()

    raw_types = raw.get("preciptype")
    if raw_types is None:
        raw_types = []
    if not isinstance(raw_types, list) or any(
        not isinstance(item, str) or item not in _PRECIPITATION for item in raw_types
    ):
        raise _Invalid()
    present = {_PRECIPITATION[item] for item in raw_types}
    types: list[PrecipitationType] = [
        kind for kind in ("rain", "snow", "freezing_rain", "ice") if kind in present
    ]

    icon = raw.get("icon")
    if icon is not None and not isinstance(icon, str):
        raise _Invalid()
    condition: WeatherCondition = _ICON_CONDITIONS.get((icon or "").strip().lower(), "unknown")
    if condition != "storm" and types:
        if len(types) > 1:
            condition = "mixed"
        else:
            condition = "ice" if types[0] == "freezing_rain" else types[0]
    return WeatherDay(
        date=expected,
        condition=condition,
        precipitation_probability_pct=probability,
        precipitation_types=types,
        temperature_low_f=low,
        temperature_high_f=high,
    )


def _optional_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _Invalid()
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise _Invalid() from exc
    if not math.isfinite(number):
        raise _Invalid()
    return number
