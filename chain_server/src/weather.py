# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider-neutral, dormant daily weather forecast contracts and adapter."""

from __future__ import annotations

import json
import math
import os
import re
from datetime import date as CalendarDate
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Literal, Mapping, Protocol
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests
from urllib.parse import quote
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

_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$", flags=re.ASCII)
_ISO_DATE_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$", flags=re.ASCII)
_FAILURE_DETAILS: dict[WeatherFailureCode, tuple[str, bool]] = {
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
_PRECIPITATION_MAP: dict[str, PrecipitationType] = {
    "rain": "rain",
    "snow": "snow",
    "freezingrain": "freezing_rain",
    "ice": "ice",
}
_PRECIPITATION_ORDER: tuple[PrecipitationType, ...] = (
    "rain",
    "snow",
    "freezing_rain",
    "ice",
)
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
    "wind": "unknown",
    "cloudy": "cloudy",
    "partly-cloudy-day": "cloudy",
    "partly-cloudy-night": "cloudy",
    "clear-day": "clear",
    "clear-night": "clear",
}


class WeatherConfig(BaseModel):
    """Closed configuration for direct construction of the dormant client."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    enabled: StrictBool = False
    provider: Literal["visual_crossing"] = "visual_crossing"
    base_url: str = VISUAL_CROSSING_BASE_URL
    api_key_env: str = "WEATHER_API_KEY"
    timeout_seconds: StrictFloat = 3.0
    max_forecast_horizon_days: StrictInt = MAX_WEATHER_DAYS
    max_range_days: StrictInt = MAX_WEATHER_DAYS

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        normalized = value.rstrip("/")
        if value != value.strip() or normalized != VISUAL_CROSSING_BASE_URL:
            raise ValueError("weather base_url must be the pinned Visual Crossing endpoint")
        return normalized

    @field_validator("api_key_env")
    @classmethod
    def validate_api_key_env(cls, value: str) -> str:
        if not _ENV_NAME_RE.fullmatch(value):
            raise ValueError("api_key_env must be an environment variable name")
        return value

    @field_validator("timeout_seconds")
    @classmethod
    def validate_timeout(cls, value: float) -> float:
        if not math.isfinite(value) or value <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        return value

    @field_validator("max_forecast_horizon_days", "max_range_days")
    @classmethod
    def validate_day_bound(cls, value: int) -> int:
        if isinstance(value, bool) or not 1 <= value <= MAX_WEATHER_DAYS:
            raise ValueError("weather day bounds must be between 1 and 15")
        return value


class WeatherRequest(BaseModel):
    """Closed place and date input accepted by the tool.

    `location` was a five-digit US ZIP, which rejected "Napa" and "Cancun"
    before any call and made the tool unusable for the destinations shoppers
    actually travel to. The provider's Timeline endpoint resolves place strings
    natively and returns the timezone it resolved, which the forecast-horizon
    check already depends on, so a place takes exactly the path a ZIP took.

    A ZIP is still a valid place string, which matters: the profile fallback
    supplies one.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    location: str
    date: CalendarDate | None = None
    start_date: CalendarDate | None = None
    end_date: CalendarDate | None = None

    @field_validator("location", mode="before")
    @classmethod
    def validate_location(cls, value: Any) -> str:
        """Accept a place a shopper would say, and nothing that is not one.

        Bounded and single-line so a prose paragraph or an injected newline
        cannot travel into a provider URL. Resolving what the place *is* stays
        with the provider; the server never geocodes and never guesses.
        """

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
    def validate_iso_date(cls, value: Any) -> CalendarDate | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            raise ValueError("weather dates must not include a time")
        if isinstance(value, CalendarDate):
            return value
        if not isinstance(value, str) or not _ISO_DATE_RE.fullmatch(value):
            raise ValueError("weather dates must use YYYY-MM-DD")
        try:
            return CalendarDate.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("weather dates must be valid calendar dates") from exc

    @model_validator(mode="after")
    def validate_date_mode(self) -> "WeatherRequest":
        if self.date is not None and (
            self.start_date is not None or self.end_date is not None
        ):
            raise ValueError("date is mutually exclusive with a date range")
        if (self.start_date is None) != (self.end_date is None):
            raise ValueError("start_date and end_date must be supplied together")
        if self.start_date is not None and self.end_date is not None:
            if self.start_date > self.end_date:
                raise ValueError("start_date must be on or before end_date")
            if (self.end_date - self.start_date).days + 1 > MAX_WEATHER_DAYS:
                raise ValueError("date ranges may contain at most 15 days")
        return self

    def explicit_window(self) -> tuple[CalendarDate, CalendarDate] | None:
        """Return an explicit requested window, or ``None`` for local today."""

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
    precipitation_probability_pct: float = Field(ge=0, le=100)
    precipitation_types: list[PrecipitationType]
    temperature_low_f: float | None = None
    temperature_high_f: float | None = None


class WeatherResult(BaseModel):
    """Normalized provider evidence returned by a successful lookup."""

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
    """Stable failure that never carries request or provider-sensitive data."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    ok: Literal[False] = False
    code: WeatherFailureCode
    message: str
    retryable: bool

    @model_validator(mode="before")
    @classmethod
    def enforce_stable_details(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        code = value.get("code")
        if code not in _FAILURE_DETAILS:
            return value
        message, retryable = _FAILURE_DETAILS[code]
        stable = dict(value)
        stable["message"] = message
        stable["retryable"] = retryable
        return stable


WeatherOutcome = WeatherResult | WeatherFailure


def weather_failure(code: WeatherFailureCode) -> WeatherFailure:
    """Create a sanitized failure using the stable message/retry contract."""

    return WeatherFailure(code=code)


class WeatherClient(Protocol):
    """Provider-neutral weather client used by the dormant wrapper."""

    def get_forecast(self, request: WeatherRequest) -> WeatherOutcome:
        """Return normalized evidence or a typed, sanitized failure."""


class _StaticFailureWeatherClient:
    def __init__(self, failure: WeatherFailure) -> None:
        self._failure = failure

    def get_forecast(self, request: WeatherRequest) -> WeatherOutcome:
        del request
        return self._failure.model_copy(deep=True)


class _Response(Protocol):
    status_code: int
    headers: Mapping[str, str]

    def iter_content(self, chunk_size: int) -> Any:
        """Yield response bytes."""

    def close(self) -> None:
        """Close the provider response."""


class _Session(Protocol):
    def get(self, url: str, **kwargs: Any) -> _Response:
        """Issue a provider GET."""


class VisualCrossingWeatherClient:
    """Visual Crossing Timeline adapter behind the provider-neutral contract."""

    def __init__(
        self,
        config: WeatherConfig,
        api_key: str,
        *,
        session: _Session | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not config.enabled or not api_key or api_key != api_key.strip():
            raise ValueError("weather client configuration is invalid")
        self._config = config
        self._api_key = api_key
        self._session = session or requests.Session()
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def get_forecast(self, request: WeatherRequest) -> WeatherOutcome:
        explicit_window = request.explicit_window()
        if (
            explicit_window is not None
            and (explicit_window[1] - explicit_window[0]).days + 1
            > self._config.max_range_days
        ):
            return weather_failure("weather_request_invalid")

        url = self._request_url(request)
        response: _Response | None = None
        try:
            response = self._session.get(
                url,
                params={
                    "unitGroup": "us",
                    "include": "days",
                    "elements": (
                        "datetime,tempmax,tempmin,precipprob,"
                        "preciptype,icon,source"
                    ),
                    "iconSet": "icons2",
                    "contentType": "json",
                    "key": self._api_key,
                },
                timeout=self._config.timeout_seconds,
                allow_redirects=False,
                stream=True,
            )
        except requests.Timeout:
            return weather_failure("weather_timeout")
        except requests.RequestException:
            return weather_failure("weather_unavailable")

        try:
            status_failure = _status_failure(response.status_code)
            if status_failure is not None:
                return status_failure
            body = _read_bounded_body(response)
            if isinstance(body, WeatherFailure):
                return body
            return self._normalize_response(request, body)
        except requests.Timeout:
            return weather_failure("weather_timeout")
        except requests.RequestException:
            return weather_failure("weather_unavailable")
        finally:
            _close_response(response)

    def _request_url(self, request: WeatherRequest) -> str:
        suffix: list[str] = [quote(request.location, safe="")]
        explicit_window = request.explicit_window()
        if explicit_window is None:
            suffix.append("today")
        elif explicit_window[0] == explicit_window[1]:
            suffix.append(explicit_window[0].isoformat())
        else:
            suffix.extend(item.isoformat() for item in explicit_window)
        return f"{self._config.base_url}/{'/'.join(suffix)}"

    def _normalize_response(
        self,
        request: WeatherRequest,
        body: bytes,
    ) -> WeatherOutcome:
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError, RecursionError):
            return weather_failure("weather_response_invalid")
        if not isinstance(payload, dict):
            return weather_failure("weather_response_invalid")

        resolved_location = payload.get("resolvedAddress")
        timezone_name = payload.get("timezone")
        raw_days = payload.get("days")
        if (
            not isinstance(resolved_location, str)
            or not resolved_location.strip()
            or resolved_location != resolved_location.strip()
            or len(resolved_location) > 512
            or not isinstance(timezone_name, str)
            or not timezone_name
            or len(timezone_name) > 128
            or not isinstance(raw_days, list)
        ):
            return weather_failure("weather_response_invalid")

        try:
            location_zone = ZoneInfo(timezone_name)
            fetched_at = self._clock()
            if fetched_at.tzinfo is None or fetched_at.utcoffset() is None:
                return weather_failure("weather_response_invalid")
            fetched_at = fetched_at.astimezone(timezone.utc)
            local_today = fetched_at.astimezone(location_zone).date()
        except (ZoneInfoNotFoundError, ValueError, OverflowError, OSError):
            return weather_failure("weather_response_invalid")

        explicit_window = request.explicit_window()
        start_date, end_date = explicit_window or (local_today, local_today)
        window_length = (end_date - start_date).days + 1
        if (
            start_date < local_today
            or end_date
            > local_today + timedelta(days=self._config.max_forecast_horizon_days - 1)
        ):
            return weather_failure("weather_outside_forecast_horizon")
        if window_length > self._config.max_range_days:
            return weather_failure("weather_request_invalid")
        if len(raw_days) != window_length:
            return weather_failure("weather_response_invalid")

        expected_dates = [
            start_date + timedelta(days=offset) for offset in range(window_length)
        ]
        normalized_days: list[WeatherDay] = []
        seen_dates: set[CalendarDate] = set()
        for raw_day, expected_date in zip(raw_days, expected_dates):
            normalized = _normalize_day(raw_day, expected_date, local_today)
            if isinstance(normalized, WeatherFailure):
                return normalized
            if normalized.date in seen_dates:
                return weather_failure("weather_response_invalid")
            seen_dates.add(normalized.date)
            normalized_days.append(normalized)

        return WeatherResult(
            fetched_at=fetched_at,
            requested_window=WeatherRequestedWindow(
                start_date=start_date,
                end_date=end_date,
            ),
            resolved_location=resolved_location,
            timezone=timezone_name,
            days=normalized_days,
            attribution=WeatherAttribution(
                label="Weather Data Provided by Visual Crossing",
                url=VISUAL_CROSSING_ATTRIBUTION_URL,
            ),
        )


def build_weather_client(
    config: WeatherConfig,
    *,
    environ: Mapping[str, str] | None = None,
    session: _Session | None = None,
    clock: Callable[[], datetime] | None = None,
) -> WeatherClient:
    """Build a direct client or a typed fail-closed disabled/config client."""

    if not config.enabled:
        return _StaticFailureWeatherClient(weather_failure("weather_disabled"))
    environment = os.environ if environ is None else environ
    api_key = environment.get(config.api_key_env, "")
    if not api_key or api_key != api_key.strip():
        return _StaticFailureWeatherClient(
            weather_failure("weather_config_invalid")
        )
    return VisualCrossingWeatherClient(
        config,
        api_key,
        session=session,
        clock=clock,
    )


def _status_failure(status_code: int) -> WeatherFailure | None:
    if status_code == 200:
        return None
    if status_code == 400:
        return weather_failure("weather_location_not_found")
    if status_code in {401, 403}:
        return weather_failure("weather_auth_failed")
    if status_code == 429:
        return weather_failure("weather_rate_limited")
    if status_code >= 500:
        return weather_failure("weather_unavailable")
    return weather_failure("weather_response_invalid")


def _read_bounded_body(response: _Response) -> bytes | WeatherFailure:
    raw_length = response.headers.get("Content-Length")
    if raw_length is not None:
        try:
            parsed_length = int(raw_length)
            if parsed_length < 0 or parsed_length > MAX_PROVIDER_RESPONSE_BYTES:
                return weather_failure("weather_response_invalid")
        except (TypeError, ValueError):
            return weather_failure("weather_response_invalid")

    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_content(chunk_size=8192):
        if not isinstance(chunk, bytes):
            return weather_failure("weather_response_invalid")
        if not chunk:
            continue
        total += len(chunk)
        if total > MAX_PROVIDER_RESPONSE_BYTES:
            return weather_failure("weather_response_invalid")
        chunks.append(chunk)
    if not chunks:
        return weather_failure("weather_response_invalid")
    return b"".join(chunks)


def _close_response(response: _Response) -> None:
    try:
        response.close()
    except Exception:
        # Closing must not replace a sanitized provider outcome with a raw
        # transport exception that may carry the prepared secret-bearing URL.
        return


def _normalize_day(
    raw_day: Any,
    expected_date: CalendarDate,
    local_today: CalendarDate,
) -> WeatherDay | WeatherFailure:
    if not isinstance(raw_day, dict):
        return weather_failure("weather_response_invalid")
    raw_date = raw_day.get("datetime")
    if (
        not isinstance(raw_date, str)
        or not _ISO_DATE_RE.fullmatch(raw_date)
        or raw_date != expected_date.isoformat()
    ):
        return weather_failure("weather_response_invalid")

    source = raw_day.get("source")
    if source == "comb":
        if expected_date != local_today:
            return weather_failure("weather_outside_forecast_horizon")
    elif source != "fcst":
        if source in {"obs", "histfcst", "stats"}:
            return weather_failure("weather_outside_forecast_horizon")
        return weather_failure("weather_response_invalid")

    probability = _finite_number(raw_day.get("precipprob"))
    if probability is None or not 0 <= probability <= 100:
        return weather_failure("weather_response_invalid")

    precipitation_types = _normalize_precipitation_types(raw_day.get("preciptype"))
    if isinstance(precipitation_types, WeatherFailure):
        return precipitation_types

    low = _optional_finite_number(raw_day.get("tempmin"))
    high = _optional_finite_number(raw_day.get("tempmax"))
    if isinstance(low, WeatherFailure) or isinstance(high, WeatherFailure):
        return weather_failure("weather_response_invalid")
    if low is not None and high is not None and low > high:
        return weather_failure("weather_response_invalid")

    icon = raw_day.get("icon")
    if icon is not None and not isinstance(icon, str):
        return weather_failure("weather_response_invalid")
    condition = _normalize_condition(icon or "", precipitation_types)
    return WeatherDay(
        date=expected_date,
        condition=condition,
        precipitation_probability_pct=probability,
        precipitation_types=precipitation_types,
        temperature_low_f=low,
        temperature_high_f=high,
    )


def _normalize_precipitation_types(
    raw_value: Any,
) -> list[PrecipitationType] | WeatherFailure:
    if raw_value is None:
        return []
    if not isinstance(raw_value, list):
        return weather_failure("weather_response_invalid")
    normalized: set[PrecipitationType] = set()
    for item in raw_value:
        if not isinstance(item, str) or item not in _PRECIPITATION_MAP:
            return weather_failure("weather_response_invalid")
        normalized.add(_PRECIPITATION_MAP[item])
    return [item for item in _PRECIPITATION_ORDER if item in normalized]


def _normalize_condition(
    icon: str,
    precipitation_types: list[PrecipitationType],
) -> WeatherCondition:
    normalized_icon = icon.strip().lower()
    icon_condition = _ICON_CONDITIONS.get(normalized_icon, "unknown")
    if icon_condition == "storm":
        return "storm"
    if len(precipitation_types) > 1:
        return "mixed"
    if precipitation_types:
        precipitation = precipitation_types[0]
        if precipitation == "freezing_rain":
            return "ice"
        return precipitation
    return icon_condition


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        normalized = float(value)
    except (OverflowError, ValueError):
        return None
    return normalized if math.isfinite(normalized) else None


def _optional_finite_number(value: Any) -> float | None | WeatherFailure:
    if value is None:
        return None
    normalized = _finite_number(value)
    if normalized is None:
        return weather_failure("weather_response_invalid")
    return normalized
