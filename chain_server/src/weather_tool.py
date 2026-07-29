# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LangChain wrappers for the typed weather client."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date as CalendarDate
from datetime import datetime, timedelta, timezone
import json
import re
from threading import Lock
from typing import Any, Collection, Literal

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from .weather import (
    VISUAL_CROSSING_ATTRIBUTION_LABEL,
    VISUAL_CROSSING_ATTRIBUTION_URL,
    WeatherClient,
    WeatherAttribution,
    WeatherDay,
    WeatherFailure,
    WeatherRequest,
    WeatherRequestedWindow,
    WeatherResult,
    weather_failure,
)


WEATHER_FORECAST_EVIDENCE_PREFIX = "WEATHER_FORECAST_EVIDENCE:"
WEATHER_FORECAST_FAILURE_PREFIX = "WEATHER_FORECAST_FAILURE:"
_ISO_DATE_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$", flags=re.ASCII)
_LOCATION_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z0-9-]*[0-9][A-Za-z0-9-]*(?![A-Za-z0-9])",
    flags=re.ASCII,
)
_NEXT_WEEK_RE = re.compile(r"\bnext\s+week\b", flags=re.IGNORECASE)
_NEXT_WEEK_NEGATED_BEFORE_RE = re.compile(
    r"\b(?:forget|ignore|cancel|skip|not|isn['’]t|is\s+not|no\s+longer)\b"
    r".{0,40}\bnext\s+week\b",
    flags=re.IGNORECASE,
)
_NEXT_WEEK_NEGATED_AFTER_RE = re.compile(
    r"\bnext\s+week\b.{0,40}\b(?:wrong|instead|cancel(?:led)?|changed|"
    r"doesn['’]t\s+work|won['’]t\s+work|will\s+not\s+work|"
    r"isn['’]t\s+right|is\s+not\s+right|is\s+out|was\s+wrong)\b",
    flags=re.IGNORECASE,
)
_OTHER_DATE_RE = re.compile(
    r"(?:"
    r"\b[0-9]{4}-[0-9]{2}-[0-9]{2}\b"
    r"|\b[0-9]{1,2}[/-][0-9]{1,2}(?:[/-][0-9]{2,4})?\b"
    r"|\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
    r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|"
    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+"
    r"[0-9]{1,2}(?:st|nd|rd|th)?\b"
    r"|\b(?:today|tomorrow|this\s+week|next\s+month)\b"
    r")",
    flags=re.IGNORECASE | re.ASCII,
)


class EventWeatherRequest(BaseModel):
    """Closed event-weather input with an explicit location authority."""

    model_config = ConfigDict(extra="forbid", strict=True)

    candidate_action: Literal[
        "reuse_prior_candidates",
        "search_new_candidates",
    ]
    location_source: Literal[
        "confirmed_saved_zip",
        "shopper_provided_location",
    ]
    location: str | None = Field(default=None, min_length=1, max_length=256)
    location_query: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
        description=(
            "Provider-facing named place. Required when location is an "
            "abbreviation or geographically ambiguous: preserve the exact "
            "location as the first component and append one or two "
            "comma-separated region/country qualifiers (for example, "
            "location='NYC', location_query='NYC, NY'). Omit only when "
            "location is already sufficiently qualified. Never add an "
            "unstated ZIP or numeric component."
        ),
    )
    relative_date: Literal["next_week"] | None = None
    date: str | None = None
    start_date: str | None = None
    end_date: str | None = None

    @field_validator("location", "location_query")
    @classmethod
    def validate_location(cls, value: str | None) -> str | None:
        if value is None:
            return None
        WeatherRequest(location=value)
        return value

    @field_validator("date", "start_date", "end_date")
    @classmethod
    def validate_iso_date(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not _ISO_DATE_RE.fullmatch(value):
            raise ValueError("weather dates must use YYYY-MM-DD")
        try:
            CalendarDate.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("weather dates must be valid calendar dates") from exc
        return value

    @model_validator(mode="after")
    def validate_authority_and_window(self) -> "EventWeatherRequest":
        if self.location_source == "confirmed_saved_zip":
            if self.location is not None or self.location_query is not None:
                raise ValueError(
                    "location fields must be omitted for confirmed_saved_zip"
                )
        elif self.location is None:
            raise ValueError(
                "location is required for shopper_provided_location"
            )

        has_explicit_window = any(
            value is not None
            for value in (self.date, self.start_date, self.end_date)
        )
        if self.relative_date is not None and has_explicit_window:
            raise ValueError(
                "relative_date is mutually exclusive with explicit dates"
            )
        if self.relative_date is not None:
            return self
        if (
            self.date is None
            and self.start_date is None
            and self.end_date is None
        ):
            raise ValueError("an exact event date or complete range is required")

        WeatherRequest(
            location=self.location or "00000",
            date=self.date,
            start_date=self.start_date,
            end_date=self.end_date,
        )
        return self


class WeatherForecastEvidence(BaseModel):
    """Bounded model-visible projection with the provider-resolved place."""

    model_config = ConfigDict(extra="forbid", strict=True)

    ok: Literal[True] = True
    provider: Literal["visual_crossing"]
    fetched_at: datetime
    requested_window: WeatherRequestedWindow
    relative_date: Literal["next_week"] | None = None
    resolved_location: str | None = Field(
        default=None,
        min_length=1,
        max_length=512,
    )
    days: list[WeatherDay]
    attribution: WeatherAttribution

    @field_validator("resolved_location")
    @classmethod
    def validate_resolved_location(cls, value: str | None) -> str | None:
        if value is not None and (
            value != value.strip()
            or any(
                ord(character) < 32 or ord(character) == 127
                for character in value
            )
        ):
            raise ValueError("resolved_location must be trimmed and single-line")
        return value

    @model_validator(mode="after")
    def validate_projection(self) -> "WeatherForecastEvidence":
        start = self.requested_window.start_date
        end = self.requested_window.end_date
        expected_count = (end - start).days + 1
        expected_dates = [
            start + timedelta(days=offset)
            for offset in range(expected_count)
        ]
        if (
            not 1 <= expected_count <= 15
            or len(self.days) != expected_count
            or [day.date for day in self.days] != expected_dates
            or self.attribution.label != VISUAL_CROSSING_ATTRIBUTION_LABEL
            or self.attribution.url != VISUAL_CROSSING_ATTRIBUTION_URL
        ):
            raise ValueError("weather forecast evidence is inconsistent")
        return self


def get_weather_forecast_tool(client: WeatherClient) -> BaseTool:
    """Build the provider-neutral direct wrapper."""

    def get_weather_forecast(
        location: str,
        date: CalendarDate | None = None,
        start_date: CalendarDate | None = None,
        end_date: CalendarDate | None = None,
    ) -> dict[str, Any]:
        request = WeatherRequest(
            location=location,
            date=date,
            start_date=start_date,
            end_date=end_date,
        )
        try:
            return client.get_forecast(request).model_dump(mode="json")
        except Exception:  # noqa: BLE001 - never expose provider exception text.
            return weather_failure("weather_unavailable").model_dump(mode="json")

    tool = StructuredTool.from_function(
        func=get_weather_forecast,
        name="get_weather_forecast_tool",
        description=(
            "Get normalized daily live-forecast evidence for exactly one "
            "named place, postal code, or address. Supply no date for local "
            "today, one exact ISO date, or a complete inclusive ISO start/end "
            "range. Never supply unresolved relative dates or invented places."
        ),
        args_schema=WeatherRequest,
        return_direct=False,
        handle_validation_error=lambda _error: weather_failure(
            "weather_request_invalid"
        ).model_dump_json(),
    )
    tool.__name__ = tool.name
    return tool


def get_event_weather_forecast_tool(
    client: WeatherClient,
    *,
    saved_zipcode: str | None,
    saved_zip_authorized: bool,
    shopper_provided_texts: Collection[str],
    current_date: CalendarDate | None = None,
    prior_candidates_available: bool = False,
    on_reuse_prior_candidates: Callable[[], None] | None = None,
) -> BaseTool:
    """Build one request-bound, event-context-only forecast tool.

    The model selects semantic authority, while the server releases the saved
    ZIP only after its narrow confirmation gate and verifies that an explicit
    named place came verbatim from shopper-authored text. The provider-resolved
    location is returned so a geographic assumption is transparent.
    """

    shopper_texts = tuple(
        text for text in shopper_provided_texts if isinstance(text, str)
    )
    request_date = current_date or datetime.now(timezone.utc).date()
    attempt_lock = Lock()
    attempted = False

    def claim_attempt() -> bool:
        nonlocal attempted
        with attempt_lock:
            if attempted:
                return False
            attempted = True
            return True

    def invalid_weather_request(_error: Any) -> str:
        claim_attempt()
        return _weather_failure_evidence(
            weather_failure("weather_request_invalid")
        )

    def get_weather_forecast(
        candidate_action: Literal[
            "reuse_prior_candidates",
            "search_new_candidates",
        ],
        location_source: Literal[
            "confirmed_saved_zip",
            "shopper_provided_location",
        ],
        location: str | None = None,
        location_query: str | None = None,
        relative_date: Literal["next_week"] | None = None,
        date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> str:
        if not claim_attempt():
            return _weather_failure_evidence(
                weather_failure("weather_request_invalid")
            )
        if (
            candidate_action == "reuse_prior_candidates"
            and not prior_candidates_available
        ):
            return _weather_failure_evidence(
                weather_failure("weather_request_invalid")
            )
        if (
            relative_date == "next_week"
            and not _shopper_stated_next_week(shopper_texts)
        ):
            return _weather_failure_evidence(
                weather_failure("weather_request_invalid")
            )

        if location_source == "confirmed_saved_zip":
            selected_location = (
                saved_zipcode
                if saved_zip_authorized
                else None
            )
        else:
            shopper_location = (
                _resolve_shopper_authored_location(location, shopper_texts)
                if location is not None
                else None
            )
            if (
                shopper_location is None
                or _adds_unstated_location_number(
                    location_query,
                    shopper_location,
                )
                or not _location_query_qualifies_shopper_location(
                    location_query,
                    shopper_location,
                )
            ):
                selected_location = None
            else:
                selected_location = location_query or shopper_location

        if selected_location is None:
            return _weather_failure_evidence(
                weather_failure("weather_request_invalid")
            )

        try:
            if relative_date == "next_week":
                start_date = request_date + timedelta(
                    days=7 - request_date.weekday()
                )
                end_date = start_date + timedelta(days=6)
            request = WeatherRequest(
                location=selected_location,
                date=date,
                start_date=start_date,
                end_date=end_date,
            )
            if (
                candidate_action == "reuse_prior_candidates"
                and on_reuse_prior_candidates is not None
            ):
                on_reuse_prior_candidates()
            outcome = client.get_forecast(request)
            if isinstance(outcome, WeatherResult):
                return _weather_success_evidence(
                    outcome,
                    include_resolved_location=(
                        location_source == "shopper_provided_location"
                    ),
                    relative_date=relative_date,
                )
            if isinstance(outcome, WeatherFailure):
                return _weather_failure_evidence(outcome)
            return _weather_failure_evidence(
                weather_failure("weather_response_invalid")
            )
        except Exception:  # noqa: BLE001 - never expose provider exception text.
            return _weather_failure_evidence(
                weather_failure("weather_unavailable")
            )

    tool = StructuredTool.from_function(
        func=get_weather_forecast,
        name="get_weather_forecast_tool",
        description=(
            "Get one live daily forecast for event styling, only after the "
            "event place and exact date or complete date range are established. "
            "Set candidate_action=reuse_prior_candidates only when this turn "
            "supplies event context for candidates already shown and does not "
            "ask for new or refined products; that action closes catalog search "
            "for the rest of the turn. Otherwise set "
            "candidate_action=search_new_candidates. "
            "Use confirmed_saved_zip only after the shopper explicitly confirms "
            "the event is in their usual area, and omit location in that mode. "
            "Otherwise use shopper_provided_location and copy the shortest "
            "sufficient place phrase verbatim from shopper-authored text, such "
            "as Denver, Cancun, Paris France, or a stated postal code. Keep "
            "that phrase in location. For an abbreviation or geographically "
            "ambiguous name, location_query is required: preserve the exact "
            "phrase as its first component and append only a standard region "
            "and/or country qualifier (for example, NYC to NYC, NY or "
            "Springfield to Springfield, TX). Never rewrite the authority "
            "phrase or add a ZIP the shopper did not state. Omit "
            "location_query only when location is already sufficiently "
            "qualified. The provider-resolved place will be disclosed. "
            "Convert unambiguous relative date "
            "language: use relative_date=next_week only for the shopper's exact "
            "'next week' wording; otherwise use an exact ISO date or complete "
            "inclusive ISO range. "
            "Call at most once per turn."
        ),
        args_schema=EventWeatherRequest,
        return_direct=False,
        handle_validation_error=invalid_weather_request,
    )
    tool.__name__ = tool.name
    return tool


def _weather_success_evidence(
    result: WeatherResult,
    *,
    include_resolved_location: bool,
    relative_date: Literal["next_week"] | None,
) -> str:
    payload = WeatherForecastEvidence(
        provider=result.provider,
        fetched_at=result.fetched_at,
        requested_window=result.requested_window,
        relative_date=relative_date,
        resolved_location=(
            result.resolved_location
            if include_resolved_location
            else None
        ),
        days=result.days,
        attribution=result.attribution,
    )
    return (
        f"{WEATHER_FORECAST_EVIDENCE_PREFIX} "
        + json.dumps(
            payload.model_dump(mode="json", exclude_none=True),
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def _weather_failure_evidence(result: WeatherFailure) -> str:
    return (
        f"{WEATHER_FORECAST_FAILURE_PREFIX} "
        + result.model_dump_json()
    )


def parse_weather_tool_evidence(
    content: str,
) -> WeatherForecastEvidence | WeatherFailure | None:
    """Parse only the two server-owned weather evidence markers."""

    if content.startswith(WEATHER_FORECAST_EVIDENCE_PREFIX):
        payload = content[len(WEATHER_FORECAST_EVIDENCE_PREFIX) :].strip()
        try:
            return WeatherForecastEvidence.model_validate_json(payload)
        except (ValidationError, ValueError):
            return None
    if content.startswith(WEATHER_FORECAST_FAILURE_PREFIX):
        payload = content[len(WEATHER_FORECAST_FAILURE_PREFIX) :].strip()
        try:
            return WeatherFailure.model_validate_json(payload)
        except (ValidationError, ValueError):
            return None
    return None


def _resolve_shopper_authored_location(
    location: str,
    shopper_texts: Collection[str],
) -> str | None:
    """Return the exact source span for a bounded shopper-authored place."""

    if not location:
        return None
    pattern = re.compile(
        rf"(?<!\w){re.escape(location)}(?!\w)",
        flags=re.IGNORECASE | re.UNICODE,
    )
    for text in shopper_texts:
        match = pattern.search(text)
        if match is not None:
            return match.group(0)
    return None


def _shopper_stated_next_week(shopper_texts: Collection[str]) -> bool:
    texts = tuple(shopper_texts)
    if not texts:
        return False

    def decision(text: str) -> bool | None:
        has_next_week = _NEXT_WEEK_RE.search(text) is not None
        if has_next_week:
            if (
                _NEXT_WEEK_NEGATED_BEFORE_RE.search(text) is not None
                or _NEXT_WEEK_NEGATED_AFTER_RE.search(text) is not None
                or _OTHER_DATE_RE.search(text) is not None
            ):
                return False
            return True
        if _OTHER_DATE_RE.search(text) is not None:
            return False
        return None

    current_decision = decision(texts[0])
    if current_decision is not None:
        return current_decision
    for text in reversed(texts[1:]):
        prior_decision = decision(text)
        if prior_decision is not None:
            return prior_decision
    return False


def _adds_unstated_location_number(
    location_query: str | None,
    shopper_location: str,
) -> bool:
    if location_query is None:
        return False
    return bool(
        set(_LOCATION_NUMBER_RE.findall(location_query))
        - set(_LOCATION_NUMBER_RE.findall(shopper_location))
    )


def _location_query_qualifies_shopper_location(
    location_query: str | None,
    shopper_location: str,
) -> bool:
    """Allow only region/country qualifiers on the authoritative place span."""

    if location_query is None:
        return True
    source_match = re.match(
        re.escape(shopper_location),
        location_query,
        flags=re.IGNORECASE | re.UNICODE,
    )
    if source_match is None:
        return False
    suffix = location_query[source_match.end() :]
    if not suffix:
        return True
    if not suffix.startswith(","):
        return False
    qualifiers = [part.strip() for part in suffix[1:].split(",")]
    if not 1 <= len(qualifiers) <= 2 or any(not part for part in qualifiers):
        return False
    return all(
        all(
            character.isalpha()
            or character in {" ", ".", "-", "'", "’"}
            for character in qualifier
        )
        for qualifier in qualifiers
    )
