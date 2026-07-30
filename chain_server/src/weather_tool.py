# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LangChain wrappers for the typed weather client."""

from __future__ import annotations

from datetime import date as CalendarDate
from datetime import datetime, timedelta, timezone
import json
import re
from threading import Lock
from typing import Any, Collection, Literal, cast

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
from shared.weather_receipts import (
    SavedAreaWeatherScope,
    ShopperLocationWeatherScope,
    WeatherLocationScope,
    WeatherReceiptWindow,
)
from shared.weather_scope import (
    CurrentWeatherScope,
    CurrentWeatherScopeTransition,
    effective_weather_scope_values,
)


WEATHER_FORECAST_EVIDENCE_PREFIX = "WEATHER_FORECAST_EVIDENCE:"
WEATHER_FORECAST_FAILURE_PREFIX = "WEATHER_FORECAST_FAILURE:"
_ISO_DATE_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$", flags=re.ASCII)
_LOCATION_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z0-9-]*[0-9][A-Za-z0-9-]*(?![A-Za-z0-9])",
    flags=re.ASCII,
)
_NEXT_WEEK_RE = re.compile(r"\bnext\s+week\b", flags=re.IGNORECASE)
_WEEKDAY_NEXT_WEEK_RE = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"\s+next\s+week\b",
    flags=re.IGNORECASE | re.ASCII,
)
_MONTH_DAY_PATTERN = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
    r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|"
    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+"
    r"[0-9]{1,2}(?:st|nd|rd|th)?(?:,?\s+[0-9]{4})?"
)
_MONTH_NUMBERS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
_STRONG_DATE_SIGNAL_RE = re.compile(
    rf"(?:"
    rf"\b[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}\b"
    rf"|\b{_MONTH_DAY_PATTERN}\b"
    rf"|\b(?:today|tomorrow)\b"
    rf")",
    flags=re.IGNORECASE | re.ASCII,
)
_DATE_RANGE_CONNECTOR_RE = re.compile(
    r"\s*(?:through|to|until|[-–—])\s*",
    flags=re.IGNORECASE,
)
_UNSUPPORTED_RELATIVE_DATE_RE = re.compile(
    r"\b(?:this\s+week|next\s+month)\b",
    flags=re.IGNORECASE | re.ASCII,
)
_DATE_AUTHORITY_UNCERTAINTY_RE = re.compile(
    r"(?:"
    r"\b(?:maybe|perhaps|possibly|unsure|uncertain|unknown|tbd)\b"
    r"|\bnot\s+sure\b"
    r"|\bto\s+be\s+determined\b"
    r"|\b(?:might|may)\s+(?:be|change|move|happen)\b"
    r"|\b(?:don['’]t|do\s+not)\s+know\b.{0,40}"
    r"\b(?:date|day|when|new)\b"
    r"|\bdid(?:n['’]t|\s+not)\s+say\b"
    r")",
    flags=re.IGNORECASE,
)
_DATE_UNCERTAINTY_BEFORE_RE = re.compile(
    r"(?:"
    r"\b(?:maybe|perhaps|possibly|unsure|uncertain|unknown|tbd)\b"
    r"|\bnot\s+sure\b"
    r"|\bto\s+be\s+determined\b"
    r"|\b(?:might|may)\s+(?:be|change|move|happen)\b"
    r"|\b(?:don['’]t|do\s+not)\s+know\b"
    r"|\bdid(?:n['’]t|\s+not)\s+say\b"
    r").{0,40}$",
    flags=re.IGNORECASE,
)
_DATE_UNCERTAINTY_AFTER_RE = re.compile(
    r"^\s*[,;]?\s*(?:maybe|perhaps|possibly|unsure|uncertain|unknown|"
    r"not\s+sure|tbd|to\s+be\s+determined)\s*[.!?]?\s*$",
    flags=re.IGNORECASE,
)
_DATE_TOPIC_RE = re.compile(
    r"\b(?:date|when|schedule|timing|new\s+day)\b",
    flags=re.IGNORECASE,
)
_DATE_WORD_RE = re.compile(r"\bdate\b", flags=re.IGNORECASE)
_GENERIC_DATE_RETRACTION_RE = re.compile(
    r"(?:"
    r"\b(?:date|day|schedule|timing)\b.{0,40}"
    r"\b(?:changed|moved|wrong|off|cancel(?:l?ed)?|"
    r"unknown|unsure|uncertain|tbd|no\s+longer\s+right|"
    r"(?:won['’]t|will\s+not|doesn['’]t|does\s+not)\s+work)\b"
    r"|\b(?:changed|moved|cancel(?:l?ed)?)\b.{0,40}"
    r"\b(?:date|day|schedule|timing)\b"
    r"|\b(?:no|without)\s+(?:new\s+)?(?:date|day)\b"
    r"|\b(?:don['’]t|do\s+not)\s+have\s+(?:a\s+)?date\b"
    r"|\b(?:haven['’]t|have\s+not)\s+set\s+(?:the\s+)?date\b"
    r")",
    flags=re.IGNORECASE,
)
_NEXT_WEEK_NEGATED_BEFORE_RE = re.compile(
    r"\b(?:forget|ignore|cancel|skip|not|no|isn['’]t|is\s+not|"
    r"no\s+longer|can['’]t\s+do|cannot\s+do|"
    r"don['’]t\s+use|do\s+not\s+use|"
    r"didn['’]t\s+say|did\s+not\s+say|changed\s+it\s+from|"
    r"change(?:d)?\s+(?:it\s+)?from)\b"
    r".{0,40}\bnext\s+week\b",
    flags=re.IGNORECASE,
)
_NEXT_WEEK_NEGATED_AFTER_RE = re.compile(
    r"\bnext\s+week\b.{0,40}\b(?:wrong|cancel(?:l?ed)?|changed|"
    r"doesn['’]t\s+work|won['’]t\s+work|will\s+not\s+work|"
    r"isn['’]t\s+right|is\s+not\s+right|is\s+out|was\s+wrong|"
    r"is\s+off|is\s+no\s+longer(?:\s+the\s+date)?)\b",
    flags=re.IGNORECASE,
)
_WEEKDAY_RE = re.compile(
    r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    flags=re.IGNORECASE | re.ASCII,
)
_DATE_NEGATED_BEFORE_RE = re.compile(
    r"\b(?:forget|ignore|skip|not|isn['’]t|is\s+not|"
    r"don['’]t\s+use|do\s+not\s+use)\b.{0,40}$",
    flags=re.IGNORECASE,
)
_DATE_NEGATED_AFTER_RE = re.compile(
    r"^.{0,40}\b(?:wrong|cancel(?:l?ed)?|changed|"
    r"isn['’]t\s+right|is\s+not\s+right|is\s+off)\b",
    flags=re.IGNORECASE,
)

RelativeWeekday = Literal[
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
]
_RELATIVE_WEEKDAY_OFFSETS: dict[RelativeWeekday, int] = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


class EventWeatherRequest(BaseModel):
    """Closed event-weather input with an explicit location authority."""

    model_config = ConfigDict(extra="forbid", strict=True)

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
    relative_date: Literal["next_week"] | None = Field(
        default=None,
        description=(
            "Use only when the shopper's latest authoritative date phrase "
            "contains the exact words 'next week'. Never use this as a "
            "placeholder when the event date is missing."
        ),
    )
    weekday: RelativeWeekday | None = Field(
        default=None,
        description=(
            "Exact weekday inside next week. Required only when the shopper's "
            "latest authoritative date phrase is exactly '<weekday> next "
            "week'; otherwise omit it."
        ),
    )
    date: str | None = Field(
        default=None,
        description=(
            "One ISO date that exactly equals the normalized current-turn "
            "shopper date. Never substitute a nearby or prior date."
        ),
    )
    start_date: str | None = Field(
        default=None,
        description=(
            "Inclusive ISO range start that exactly equals the normalized "
            "current-turn shopper range."
        ),
    )
    end_date: str | None = Field(
        default=None,
        description=(
            "Inclusive ISO range end that exactly equals the normalized "
            "current-turn shopper range."
        ),
    )

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
        if self.weekday is not None and self.relative_date != "next_week":
            raise ValueError("weekday requires relative_date=next_week")
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


class WeatherScopeSelection(BaseModel):
    """Semantic current-turn update compiled into the durable weather scope."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["continue", "replace"] = Field(
        description=(
            "Initial scope creation uses replace. Use continue only for the "
            "same event, trip, or weather-planning "
            "subject. Use replace for a new or different subject; omitted "
            "location or date fields are then cleared. When replacing an "
            "existing scope, subject_change_quote must cite the exact "
            "current-turn words that explicitly introduce the new subject. "
            "As a fail-safe, supplying a location under continue when the "
            "scope already has one clears the older date unless this turn "
            "supplies a new date too."
        )
    )
    subject_change_quote: str | None = Field(
        default=None,
        min_length=1,
        max_length=120,
        description=(
            "Exact current-turn shopper phrase that explicitly introduces a "
            "new, different, or separate event, trip, or weather-planning "
            "subject. Required for replace when a scope already exists. A "
            "pronoun, location, date, or occasion alone is not subject-change "
            "evidence. Omit for continue and initial scope creation."
        ),
    )
    location_source: Literal[
        "confirmed_saved_zip",
        "shopper_provided_location",
    ] | None = Field(
        default=None,
        description=(
            "Current-turn location authority. Use shopper_provided_location "
            "for a place the shopper states now, or confirmed_saved_zip only "
            "when the shopper explicitly confirms the saved area now."
        ),
    )
    location: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
        description=(
            "The shortest exact current-turn shopper phrase naming the place. "
            "Do not rewrite it or copy a place from an older subject."
        ),
    )
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
    relative_date: Literal["next_week"] | None = Field(
        default=None,
        description=(
            "Use only when the shopper's current turn says 'next week'. "
            "With no weekday it means the full Monday-Sunday window."
        ),
    )
    weekday: RelativeWeekday | None = Field(
        default=None,
        description=(
            "Exact weekday inside next week. Supply it only when the current "
            "turn explicitly says '<weekday> next week'."
        ),
    )
    date: str | None = Field(
        default=None,
        description=(
            "One ISO date that exactly equals the normalized current-turn "
            "shopper date. Never substitute a nearby or prior date."
        ),
    )
    start_date: str | None = Field(
        default=None,
        description=(
            "Inclusive ISO range start that exactly equals the normalized "
            "current-turn shopper range."
        ),
    )
    end_date: str | None = Field(
        default=None,
        description=(
            "Inclusive ISO range end that exactly equals the normalized "
            "current-turn shopper range."
        ),
    )

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
    def validate_partial_authority(self) -> "WeatherScopeSelection":
        if self.action == "continue" and self.subject_change_quote is not None:
            raise ValueError("continue must omit subject_change_quote")
        if self.location_source is None:
            if self.location is not None or self.location_query is not None:
                raise ValueError("location_source is required with location")
        elif self.location_source == "confirmed_saved_zip":
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
        if self.weekday is not None and self.relative_date != "next_week":
            raise ValueError("weekday requires relative_date=next_week")
        if has_explicit_window:
            WeatherRequest(
                location="scope",
                date=self.date,
                start_date=self.start_date,
                end_date=self.end_date,
            )
        has_location = self.location_source is not None
        has_window = self.relative_date is not None or has_explicit_window
        if self.action == "continue" and not (has_location or has_window):
            raise ValueError("continue requires a current-turn scope update")
        return self


def compile_weather_scope_transition(
    selection: WeatherScopeSelection,
    *,
    current_shopper_text: str,
    saved_zip_authorized: bool,
    expected_projection_version: int,
    current_date: CalendarDate | None = None,
    replacement_evidence_required: bool = False,
    continuation_context_available: bool = True,
    current_location_scope: WeatherLocationScope | None = None,
) -> CurrentWeatherScopeTransition:
    """Validate current-turn provenance and normalize one semantic transition."""

    if selection.action == "continue" and not continuation_context_available:
        raise ValueError("initial weather scope creation must use replace")
    if selection.action == "replace" and replacement_evidence_required:
        replacement_quote = selection.subject_change_quote
        if (
            replacement_quote is None
            or _resolve_shopper_authored_location(
                replacement_quote,
                (current_shopper_text,),
            )
            is None
        ):
            raise ValueError(
                "replacement requires an exact current-turn subject-change quote"
            )

    location_scope: WeatherLocationScope | None = None
    if selection.location_source == "confirmed_saved_zip":
        if not saved_zip_authorized:
            raise ValueError("saved area is not confirmed by the current turn")
        location_scope = SavedAreaWeatherScope()
    elif selection.location_source == "shopper_provided_location":
        shopper_location = (
            _resolve_shopper_authored_location(
                selection.location or "",
                (current_shopper_text,),
            )
        )
        if (
            shopper_location is None
            or _adds_unstated_location_number(
                selection.location_query,
                shopper_location,
            )
            or not _location_query_qualifies_shopper_location(
                selection.location_query,
                shopper_location,
            )
        ):
            raise ValueError(
                "shopper location must be an exact current-turn source span"
            )
        location_scope = ShopperLocationWeatherScope(
            location=shopper_location,
            location_query=selection.location_query,
        )

    requested_window: WeatherReceiptWindow | None = None
    if selection.relative_date == "next_week":
        shopper_texts = (current_shopper_text,)
        if not _shopper_stated_next_week(shopper_texts):
            raise ValueError("next_week is not authoritative in the current turn")
        weekday_required, authoritative_weekday = (
            _shopper_stated_relative_weekday(shopper_texts)
        )
        if (
            weekday_required
            and (
                authoritative_weekday is None
                or selection.weekday != authoritative_weekday
            )
        ) or (not weekday_required and selection.weekday is not None):
            raise ValueError("next_week weekday does not match the current turn")
        request_date = current_date or datetime.now(timezone.utc).date()
        start_date = request_date + timedelta(
            days=7 - request_date.weekday()
        )
        if selection.weekday is None:
            end_date = start_date + timedelta(days=6)
        else:
            start_date += timedelta(
                days=_RELATIVE_WEEKDAY_OFFSETS[selection.weekday]
            )
            end_date = start_date
        requested_window = WeatherReceiptWindow(
            start_date=start_date,
            end_date=end_date,
        )
    elif any(
        value is not None
        for value in (selection.date, selection.start_date, selection.end_date)
    ):
        authority_window = _authoritative_explicit_window(
            current_shopper_text,
            current_date=(
                current_date or datetime.now(timezone.utc).date()
            ),
        )
        if authority_window is None:
            raise ValueError(
                "explicit weather window lacks current-turn date authority"
            )
        request = WeatherRequest(
            location="scope",
            date=selection.date,
            start_date=selection.start_date,
            end_date=selection.end_date,
        )
        window = request.explicit_window()
        if window is None:
            raise ValueError("weather scope requires an explicit window")
        if window != authority_window:
            raise ValueError(
                "explicit weather window does not match current-turn authority"
            )
        requested_window = WeatherReceiptWindow(
            start_date=window[0],
            end_date=window[1],
        )

    return CurrentWeatherScopeTransition(
        expected_projection_version=expected_projection_version,
        action=selection.action,
        location_scope=location_scope,
        requested_window=requested_window,
        clear_window=(
            True
            if (
                selection.action == "continue"
                and location_scope is not None
                and current_location_scope is not None
                and requested_window is None
            )
            else None
        ),
    )


class WeatherScopeForecastBinding:
    """Request-local binding between the semantic scope and provider adapter."""

    def __init__(
        self,
        current_scope: CurrentWeatherScope,
        *,
        saved_zipcode: str | None,
    ) -> None:
        self.current_scope = current_scope
        self.saved_zipcode = saved_zipcode
        self.transition: CurrentWeatherScopeTransition | None = None
        self.relative_date: Literal["next_week"] | None = None
        self.weekday: RelativeWeekday | None = None

    def bind(
        self,
        transition: CurrentWeatherScopeTransition,
        *,
        relative_date: Literal["next_week"] | None,
        weekday: RelativeWeekday | None,
    ) -> None:
        self.transition = transition
        self.relative_date = relative_date
        self.weekday = weekday

    def effective_values(
        self,
    ) -> tuple[WeatherLocationScope | None, WeatherReceiptWindow | None]:
        return effective_weather_scope_values(
            self.current_scope,
            self.transition,
        )


class _ScopedWeatherToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


def get_scoped_weather_forecast_tool(
    client: WeatherClient,
    binding: WeatherScopeForecastBinding,
) -> BaseTool:
    """Build a zero-argument forecast tool bound to one validated scope."""

    attempt_lock = Lock()
    attempted = False

    def claim_attempt() -> bool:
        nonlocal attempted
        with attempt_lock:
            if attempted:
                return False
            attempted = True
            return True

    def get_weather_forecast() -> str:
        if not claim_attempt():
            return _weather_failure_evidence(
                weather_failure("weather_request_invalid")
            )
        location_scope, requested_window = binding.effective_values()
        if location_scope is None or requested_window is None:
            return _weather_failure_evidence(
                weather_failure("weather_request_invalid")
            )
        if isinstance(location_scope, SavedAreaWeatherScope):
            selected_location = binding.saved_zipcode
            include_resolved_location = False
        else:
            selected_location = (
                location_scope.location_query or location_scope.location
            )
            include_resolved_location = True
        if selected_location is None:
            return _weather_failure_evidence(
                weather_failure("weather_request_invalid")
            )
        try:
            outcome = client.get_forecast(
                WeatherRequest(
                    location=selected_location,
                    start_date=requested_window.start_date,
                    end_date=requested_window.end_date,
                )
            )
            if isinstance(outcome, WeatherResult):
                return _weather_success_evidence(
                    outcome,
                    include_resolved_location=include_resolved_location,
                    relative_date=binding.relative_date,
                    weekday=binding.weekday,
                )
            if isinstance(outcome, WeatherFailure):
                return _weather_failure_evidence(outcome)
            return _weather_failure_evidence(
                weather_failure("weather_response_invalid")
            )
        except Exception:  # noqa: BLE001 - sanitize provider failures.
            return _weather_failure_evidence(
                weather_failure("weather_unavailable")
            )

    tool = StructuredTool.from_function(
        func=get_weather_forecast,
        name="get_weather_forecast_tool",
        description=(
            "Get one live daily forecast for the current typed styling scope. "
            "The server has already bound the shopper-authorized location and "
            "date window; this tool accepts no arguments. Call at most once "
            "and only when event-context instructions require current weather."
        ),
        args_schema=_ScopedWeatherToolInput,
        return_direct=False,
    )
    tool.__name__ = tool.name
    return tool


class WeatherForecastEvidence(BaseModel):
    """Bounded model-visible projection with the provider-resolved place."""

    model_config = ConfigDict(extra="forbid", strict=True)

    ok: Literal[True] = True
    provider: Literal["visual_crossing"]
    fetched_at: datetime
    requested_window: WeatherRequestedWindow
    relative_date: Literal["next_week"] | None = None
    weekday: RelativeWeekday | None = None
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
        if self.weekday is not None and self.relative_date != "next_week":
            raise ValueError("weather weekday provenance is inconsistent")
        if self.relative_date == "next_week":
            expected_relative_days = 1 if self.weekday is not None else 7
            if expected_count != expected_relative_days:
                raise ValueError("weather relative-date window is inconsistent")
            if (
                self.weekday is not None
                and start.weekday()
                != _RELATIVE_WEEKDAY_OFFSETS[self.weekday]
            ):
                raise ValueError("weather relative weekday is inconsistent")
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
) -> BaseTool:
    """Build one request-bound, event-context-only forecast tool.

    The server releases the saved ZIP only after its narrow confirmation gate
    and verifies that an explicit named place came verbatim from shopper-authored
    text. The provider-resolved location is returned so a geographic assumption
    is transparent.
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
        location_source: Literal[
            "confirmed_saved_zip",
            "shopper_provided_location",
        ],
        location: str | None = None,
        location_query: str | None = None,
        relative_date: Literal["next_week"] | None = None,
        weekday: RelativeWeekday | None = None,
        date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> str:
        if not claim_attempt():
            return _weather_failure_evidence(
                weather_failure("weather_request_invalid")
            )
        if relative_date == "next_week":
            if not _shopper_stated_next_week(shopper_texts):
                return _weather_failure_evidence(
                    weather_failure("weather_request_invalid")
                )
            weekday_required, authoritative_weekday = (
                _shopper_stated_relative_weekday(shopper_texts)
            )
            if (
                weekday_required
                and (
                    authoritative_weekday is None
                    or weekday != authoritative_weekday
                )
            ) or (not weekday_required and weekday is not None):
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
                if weekday is None:
                    end_date = start_date + timedelta(days=6)
                else:
                    date = start_date + timedelta(
                        days=_RELATIVE_WEEKDAY_OFFSETS[weekday]
                    )
                    start_date = None
                    end_date = None
            request = WeatherRequest(
                location=selected_location,
                date=date,
                start_date=start_date,
                end_date=end_date,
            )
            outcome = client.get_forecast(request)
            if isinstance(outcome, WeatherResult):
                return _weather_success_evidence(
                    outcome,
                    include_resolved_location=(
                        location_source == "shopper_provided_location"
                    ),
                    relative_date=relative_date,
                    weekday=weekday,
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
            "language: for the shopper's exact '<weekday> next week' phrase, "
            "use relative_date=next_week and the matching lowercase weekday; "
            "the server resolves that one day inside the next "
            "Monday-through-Sunday window. For bare 'next week', use "
            "relative_date=next_week and omit weekday so the server resolves "
            "the full window. Never omit or change a shopper-stated weekday. "
            "Otherwise use an exact ISO date or complete inclusive ISO range. "
            "If the shopper has not supplied one of those date forms, do not "
            "call this tool; apply the event-context minimum-question rule "
            "instead. Never use next_week merely to satisfy the required date "
            "input. "
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
    weekday: RelativeWeekday | None,
) -> str:
    payload = WeatherForecastEvidence(
        provider=result.provider,
        fetched_at=result.fetched_at,
        requested_window=result.requested_window,
        relative_date=relative_date,
        weekday=weekday,
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


def _statement_weekdays(text: str) -> set[RelativeWeekday]:
    return {
        cast(RelativeWeekday, match.group(0).casefold())
        for match in _WEEKDAY_RE.finditer(text)
    }


def _has_generic_date_retraction(text: str) -> bool:
    if _DATE_WORD_RE.search(text) is not None:
        return True
    if _GENERIC_DATE_RETRACTION_RE.search(text) is not None:
        return True
    return (
        _DATE_TOPIC_RE.search(text) is not None
        and _DATE_AUTHORITY_UNCERTAINTY_RE.search(text) is not None
    )


def _same_clause_before(text: str, start: int) -> str:
    before = text[max(0, start - 40) : start]
    separator = max(before.rfind(mark) for mark in ".!?;")
    return before[separator + 1 :]


def _date_signal_is_uncertain(
    text: str,
    *,
    start: int,
    end: int,
) -> bool:
    before = _same_clause_before(text, start)
    after = text[end : end + 40]
    return (
        _DATE_UNCERTAINTY_BEFORE_RE.search(before) is not None
        or _DATE_UNCERTAINTY_AFTER_RE.fullmatch(after) is not None
    )


def _strong_date_signal_decision(text: str) -> bool | None:
    """Classify a bounded exact-date signal without interpreting slash values."""

    matches = tuple(_STRONG_DATE_SIGNAL_RE.finditer(text))
    if not matches:
        return None
    if len(matches) > 2:
        return False
    if len(matches) == 2 and _DATE_RANGE_CONNECTOR_RE.fullmatch(
        text[matches[0].end() : matches[1].start()]
    ) is None:
        return False

    start = matches[0].start()
    end = matches[-1].end()
    before = _same_clause_before(text, start)
    after = text[end : end + 40]
    if (
        _date_signal_is_uncertain(text, start=start, end=end)
        or _DATE_NEGATED_BEFORE_RE.search(before) is not None
        or _DATE_NEGATED_AFTER_RE.search(after) is not None
    ):
        return False
    return True


def _authoritative_explicit_window(
    text: str,
    *,
    current_date: CalendarDate,
) -> tuple[CalendarDate, CalendarDate] | None:
    """Normalize the closed current-turn explicit-date grammar exactly."""

    if _strong_date_signal_decision(text) is not True:
        return None
    matches = tuple(_STRONG_DATE_SIGNAL_RE.finditer(text))
    if not 1 <= len(matches) <= 2:
        return None
    start = _parse_explicit_date_signal(
        matches[0].group(0),
        current_date=current_date,
        not_before=current_date,
    )
    if start is None:
        return None
    if len(matches) == 1:
        return start, start
    end = _parse_explicit_date_signal(
        matches[1].group(0),
        current_date=current_date,
        not_before=start,
    )
    if end is None or end < start:
        return None
    return start, end


def _parse_explicit_date_signal(
    signal: str,
    *,
    current_date: CalendarDate,
    not_before: CalendarDate,
) -> CalendarDate | None:
    """Normalize one already-bounded ISO, month/day, today, or tomorrow."""

    lowered = signal.casefold()
    if lowered == "today":
        return current_date
    if lowered == "tomorrow":
        return current_date + timedelta(days=1)
    if _ISO_DATE_RE.fullmatch(signal):
        try:
            return CalendarDate.fromisoformat(signal)
        except ValueError:
            return None

    match = re.fullmatch(
        r"(?P<month>[A-Za-z]+)\.?\s+"
        r"(?P<day>[0-9]{1,2})(?:st|nd|rd|th)?"
        r"(?:,?\s+(?P<year>[0-9]{4}))?",
        signal,
        flags=re.IGNORECASE | re.ASCII,
    )
    if match is None:
        return None
    month_token = match.group("month").casefold()
    if month_token == "sept":
        month_token = "sep"
    month = _MONTH_NUMBERS.get(month_token[:3])
    if month is None:
        return None
    explicit_year = match.group("year")
    year = int(explicit_year) if explicit_year else current_date.year
    try:
        parsed = CalendarDate(year, month, int(match.group("day")))
        if explicit_year is None and parsed < not_before:
            parsed = CalendarDate(year + 1, month, int(match.group("day")))
    except ValueError:
        return None
    return parsed


def _next_week_decision(text: str) -> bool | None:
    """Classify one shopper turn's bounded next-week authority."""

    has_next_week = _NEXT_WEEK_RE.search(text) is not None
    strong_date_decision = _strong_date_signal_decision(text)
    if has_next_week:
        next_week_matches = tuple(_NEXT_WEEK_RE.finditer(text))
        next_week_match = next_week_matches[0]
        same_clause_with_signal = (
            _same_clause_before(text, next_week_match.start())
            + next_week_match.group(0)
        )
        negated_before = (
            _NEXT_WEEK_NEGATED_BEFORE_RE.search(same_clause_with_signal)
            is not None
        )
        weekdays = _statement_weekdays(text)
        qualified_weekdays = {
            cast(RelativeWeekday, match.group(1).casefold())
            for match in _WEEKDAY_NEXT_WEEK_RE.finditer(text)
        }
        if (
            len(next_week_matches) != 1
            or _date_signal_is_uncertain(
                text,
                start=next_week_match.start(),
                end=next_week_match.end(),
            )
            or negated_before
            or _NEXT_WEEK_NEGATED_AFTER_RE.search(text) is not None
            or strong_date_decision is not None
            or _UNSUPPORTED_RELATIVE_DATE_RE.search(text) is not None
            or len(weekdays) > 1
            or (bool(weekdays) and qualified_weekdays != weekdays)
        ):
            return False
        return True
    if (
        strong_date_decision is not None
        or _UNSUPPORTED_RELATIVE_DATE_RE.search(text) is not None
        or _WEEKDAY_RE.search(text) is not None
        or _has_generic_date_retraction(text)
    ):
        return False
    return None


def _authoritative_next_week_statement(
    shopper_texts: Collection[str],
) -> tuple[str | None, bool | None]:
    """Return the latest date-relevant shopper turn and its decision."""

    texts = tuple(shopper_texts)
    if not texts:
        return None, None

    current_decision = _next_week_decision(texts[0])
    if current_decision is not None:
        return texts[0], current_decision
    for text in reversed(texts[1:]):
        prior_decision = _next_week_decision(text)
        if prior_decision is not None:
            return text, prior_decision
    return None, None


def _shopper_stated_next_week(shopper_texts: Collection[str]) -> bool:
    _statement, decision = _authoritative_next_week_statement(shopper_texts)
    return decision is True


def _shopper_stated_relative_weekday(
    shopper_texts: Collection[str],
) -> tuple[bool, RelativeWeekday | None]:
    """Return whether an exact weekday is required and its unambiguous value."""

    statement, decision = _authoritative_next_week_statement(shopper_texts)
    if statement is None or decision is not True:
        return False, None
    weekdays = _statement_weekdays(statement)
    qualified_weekdays = {
        cast(RelativeWeekday, match.group(1).casefold())
        for match in _WEEKDAY_NEXT_WEEK_RE.finditer(statement)
    }
    if not weekdays:
        return False, None
    if len(weekdays) != 1 or qualified_weekdays != weekdays:
        return True, None
    return True, next(iter(weekdays))


def weather_date_context_available(
    shopper_texts: Collection[str],
) -> bool:
    """Return whether the latest shopper date authority can support a lookup."""

    texts = tuple(shopper_texts)
    if not texts:
        return False

    def decision(text: str) -> bool | None:
        if _NEXT_WEEK_RE.search(text) is not None:
            return _next_week_decision(text)
        strong_date_decision = _strong_date_signal_decision(text)
        if strong_date_decision is not None:
            return strong_date_decision
        if (
            _UNSUPPORTED_RELATIVE_DATE_RE.search(text) is not None
            or _WEEKDAY_RE.search(text) is not None
            or _has_generic_date_retraction(text)
        ):
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
