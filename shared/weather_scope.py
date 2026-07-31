# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Closed contract for one conversation's current weather-planning scope."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .weather_receipts import WeatherLocationScope, WeatherReceiptWindow


MAX_CURRENT_WEATHER_SCOPE_SOURCE_TURNS = 3


class _WeatherScopeModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class CurrentWeatherScopeSourceTurn(_WeatherScopeModel):
    """One exact completed turn referenced by the current weather scope."""

    turn_id: str = Field(..., min_length=1, max_length=256)
    sequence: int = Field(..., ge=1)
    shopper_text: str = Field(..., min_length=1, max_length=100_000)
    assistant_text: str = Field(..., max_length=100_000)
    status: Literal["completed"]

    @field_validator("turn_id")
    @classmethod
    def _validate_turn_id(cls, value: str) -> str:
        return _validate_source_turn_id(value)


class WeatherScopeLocationAuthority(_WeatherScopeModel):
    """One validated location value with its shopper-turn provenance."""

    value: WeatherLocationScope
    source_turn_id: str = Field(..., min_length=1, max_length=256)
    source_sequence: int = Field(..., ge=1)

    @field_validator("source_turn_id")
    @classmethod
    def _validate_source_turn_id(cls, value: str) -> str:
        return _validate_source_turn_id(value)


class WeatherScopeWindowAuthority(_WeatherScopeModel):
    """One normalized forecast window with its shopper-turn provenance."""

    value: WeatherReceiptWindow
    source_turn_id: str = Field(..., min_length=1, max_length=256)
    source_sequence: int = Field(..., ge=1)

    @field_validator("source_turn_id")
    @classmethod
    def _validate_source_turn_id(cls, value: str) -> str:
        return _validate_source_turn_id(value)


class WeatherScopeUnavailableAuthority(_WeatherScopeModel):
    """One shopper-stated unavailable component with turn provenance."""

    source_turn_id: str = Field(..., min_length=1, max_length=256)
    source_sequence: int = Field(..., ge=1)

    @field_validator("source_turn_id")
    @classmethod
    def _validate_source_turn_id(cls, value: str) -> str:
        return _validate_source_turn_id(value)


class CurrentWeatherScope(_WeatherScopeModel):
    """Memory-owned singleton; not an event registry or evidence record."""

    revision: int = Field(default=0, ge=0)
    location: WeatherScopeLocationAuthority | None = None
    window: WeatherScopeWindowAuthority | None = None
    location_unavailable: WeatherScopeUnavailableAuthority | None = None
    window_unavailable: WeatherScopeUnavailableAuthority | None = None
    pending_question: Literal["event_location", "event_date"] | None = None
    pending_source_turn_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
    )
    pending_source_sequence: int | None = Field(default=None, ge=1)

    @field_validator("pending_source_turn_id")
    @classmethod
    def _validate_pending_source_turn_id(
        cls,
        value: str | None,
    ) -> str | None:
        return _validate_source_turn_id(value) if value is not None else None

    @model_validator(mode="after")
    def _validate_initial_scope(self) -> "CurrentWeatherScope":
        if self.revision == 0 and (
            self.location is not None
            or self.window is not None
            or self.location_unavailable is not None
            or self.window_unavailable is not None
            or self.pending_question is not None
            or self.pending_source_turn_id is not None
            or self.pending_source_sequence is not None
        ):
            raise ValueError("initial weather scope cannot contain authority")
        pending_binding = (
            self.pending_question,
            self.pending_source_turn_id,
            self.pending_source_sequence,
        )
        if any(value is not None for value in pending_binding) and not all(
            value is not None for value in pending_binding
        ):
            raise ValueError(
                "pending weather question and source binding are required together"
            )
        if (
            self.location is not None
            and self.location_unavailable is not None
        ):
            raise ValueError(
                "weather location value and unavailability are mutually exclusive"
            )
        if self.window is not None and self.window_unavailable is not None:
            raise ValueError(
                "weather window value and unavailability are mutually exclusive"
            )
        if self.pending_question is not None and (
            self.location_unavailable is not None
            or self.window_unavailable is not None
        ):
            raise ValueError(
                "pending weather question requires every component to be askable"
            )
        if self.pending_question == "event_location" and self.location is not None:
            raise ValueError(
                "pending location question requires an askable missing location"
            )
        if self.pending_question == "event_date" and (
            self.window is not None or self.window_unavailable is not None
        ):
            raise ValueError(
                "pending date question requires an askable missing window"
            )
        return self


def current_weather_scope_source_references(
    scope: CurrentWeatherScope,
) -> tuple[tuple[str, int], ...]:
    """Return the scope's unique source-turn identities in durable order."""

    references: set[tuple[str, int]] = set()
    if scope.location is not None:
        references.add(
            (
                scope.location.source_turn_id,
                scope.location.source_sequence,
            )
        )
    if scope.window is not None:
        references.add(
            (
                scope.window.source_turn_id,
                scope.window.source_sequence,
            )
        )
    if scope.location_unavailable is not None:
        references.add(
            (
                scope.location_unavailable.source_turn_id,
                scope.location_unavailable.source_sequence,
            )
        )
    if scope.window_unavailable is not None:
        references.add(
            (
                scope.window_unavailable.source_turn_id,
                scope.window_unavailable.source_sequence,
            )
        )
    if (
        scope.pending_source_turn_id is not None
        and scope.pending_source_sequence is not None
    ):
        references.add(
            (
                scope.pending_source_turn_id,
                scope.pending_source_sequence,
            )
        )
    return tuple(sorted(references, key=lambda item: (item[1], item[0])))


class CurrentWeatherScopeTransition(_WeatherScopeModel):
    """Semantically selected, server-compiled transition applied by memory."""

    expected_projection_version: int = Field(..., ge=0)
    action: Literal["continue", "replace"]
    location_scope: WeatherLocationScope | None = None
    requested_window: WeatherReceiptWindow | None = None
    clear_window: Literal[True] | None = None

    @model_validator(mode="after")
    def _continue_requires_a_patch(self) -> "CurrentWeatherScopeTransition":
        if (
            self.action == "continue"
            and self.location_scope is None
            and self.requested_window is None
        ):
            raise ValueError("continue weather scope transition requires a patch")
        if self.clear_window and (
            self.action != "continue"
            or self.location_scope is None
            or self.requested_window is not None
        ):
            raise ValueError(
                "clear_window requires a location-only continuation"
            )
        return self


class CurrentWeatherScopeResolution(_WeatherScopeModel):
    """Atomic resolution of both components of the current weather scope."""

    expected_projection_version: int = Field(..., ge=0)
    expected_scope_revision: int = Field(..., ge=0)
    location_action: Literal["retain", "set", "clear", "unavailable"]
    window_action: Literal["retain", "set", "clear", "unavailable"]
    location_scope: WeatherLocationScope | None = None
    requested_window: WeatherReceiptWindow | None = None
    pending_question: Literal["event_location", "event_date"] | None = None
    preserve_pending_source_turn_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
    )
    complete_pending_source_turn_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
    )
    decline_pending_source_turn_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
    )
    supersede_pending_source_turn_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
    )

    @field_validator(
        "preserve_pending_source_turn_id",
        "complete_pending_source_turn_id",
        "decline_pending_source_turn_id",
        "supersede_pending_source_turn_id",
    )
    @classmethod
    def _validate_pending_source_turn_id_control(
        cls,
        value: str | None,
    ) -> str | None:
        return _validate_source_turn_id(value) if value is not None else None

    @model_validator(mode="after")
    def _actions_match_payloads(self) -> "CurrentWeatherScopeResolution":
        if (self.location_action == "set") != (
            self.location_scope is not None
        ):
            raise ValueError(
                "location_scope is required exactly when location_action=set"
            )
        if (self.window_action == "set") != (
            self.requested_window is not None
        ):
            raise ValueError(
                "requested_window is required exactly when window_action=set"
            )
        if (
            self.location_action == "retain"
            and self.window_action == "retain"
        ):
            raise ValueError(
                "weather scope resolution must change or clear a component"
            )
        if (
            self.preserve_pending_source_turn_id is not None
            and self.pending_question is None
        ):
            raise ValueError(
                "pending source preservation requires a pending question"
            )
        pending_controls = (
            self.preserve_pending_source_turn_id,
            self.complete_pending_source_turn_id,
            self.decline_pending_source_turn_id,
            self.supersede_pending_source_turn_id,
        )
        if sum(control is not None for control in pending_controls) > 1:
            raise ValueError(
                "pending source preservation, completion, decline, and "
                "supersession are mutually exclusive"
            )
        if (
            self.decline_pending_source_turn_id is not None
            and self.pending_question is not None
        ):
            raise ValueError(
                "pending decline must clear the pending question"
            )
        if (
            self.pending_question == "event_location"
            and self.location_action == "unavailable"
        ) or (
            self.pending_question == "event_date"
            and self.window_action == "unavailable"
        ):
            raise ValueError(
                "pending question cannot target an unavailable component"
            )
        return self


def effective_weather_scope_values(
    current_scope: CurrentWeatherScope,
    transition: CurrentWeatherScopeTransition | None,
) -> tuple[WeatherLocationScope | None, WeatherReceiptWindow | None]:
    """Return the location/window values produced by one scope transition."""

    if transition is None:
        return (
            current_scope.location.value if current_scope.location else None,
            current_scope.window.value if current_scope.window else None,
        )
    location = (
        current_scope.location.value
        if transition.action == "continue" and current_scope.location
        else None
    )
    window = (
        current_scope.window.value
        if (
            transition.action == "continue"
            and not transition.clear_window
            and current_scope.window
        )
        else None
    )
    return (
        transition.location_scope or location,
        transition.requested_window or window,
    )


def effective_resolved_weather_scope_values(
    current_scope: CurrentWeatherScope,
    resolution: CurrentWeatherScopeResolution | None,
) -> tuple[WeatherLocationScope | None, WeatherReceiptWindow | None]:
    """Return one explicitly resolved location/window pair without fallback."""

    if resolution is None:
        return (
            current_scope.location.value if current_scope.location else None,
            current_scope.window.value if current_scope.window else None,
        )
    _validate_resolution_against_scope(current_scope, resolution)
    if resolution.location_action == "retain":
        location = (
            current_scope.location.value
            if current_scope.location is not None
            else None
        )
    elif resolution.location_action == "set":
        location = resolution.location_scope
    else:
        location = None
    if resolution.window_action == "retain":
        window = (
            current_scope.window.value
            if current_scope.window is not None
            else None
        )
    elif resolution.window_action == "set":
        window = resolution.requested_window
    else:
        window = None
    return location, window


def effective_resolved_weather_scope_unavailability(
    current_scope: CurrentWeatherScope,
    resolution: CurrentWeatherScopeResolution | None,
) -> tuple[bool, bool]:
    """Return effective location/date unavailability for one resolution."""

    if resolution is None:
        return (
            current_scope.location_unavailable is not None,
            current_scope.window_unavailable is not None,
        )
    _validate_resolution_against_scope(current_scope, resolution)
    location_unavailable = (
        current_scope.location_unavailable is not None
        if resolution.location_action == "retain"
        else resolution.location_action == "unavailable"
    )
    window_unavailable = (
        current_scope.window_unavailable is not None
        if resolution.window_action == "retain"
        else resolution.window_action == "unavailable"
    )
    return location_unavailable, window_unavailable


def apply_current_weather_scope_transition(
    current_scope: CurrentWeatherScope,
    transition: CurrentWeatherScopeTransition | None,
    *,
    source_turn_id: str,
    source_sequence: int,
) -> CurrentWeatherScope:
    """Apply one transition and stamp only newly supplied authority values."""

    if transition is None:
        return current_scope
    retained_location = (
        current_scope.location if transition.action == "continue" else None
    )
    retained_window = (
        current_scope.window
        if transition.action == "continue" and not transition.clear_window
        else None
    )
    retained_location_unavailable = (
        current_scope.location_unavailable
        if transition.action == "continue"
        else None
    )
    retained_window_unavailable = (
        current_scope.window_unavailable
        if transition.action == "continue" and not transition.clear_window
        else None
    )
    next_location = (
        WeatherScopeLocationAuthority(
            value=transition.location_scope,
            source_turn_id=source_turn_id,
            source_sequence=source_sequence,
        )
        if transition.location_scope is not None
        else retained_location
    )
    next_window = (
        WeatherScopeWindowAuthority(
            value=transition.requested_window,
            source_turn_id=source_turn_id,
            source_sequence=source_sequence,
        )
        if transition.requested_window is not None
        else retained_window
    )
    next_location_unavailable = (
        None
        if transition.location_scope is not None
        else retained_location_unavailable
    )
    next_window_unavailable = (
        None
        if transition.requested_window is not None
        else retained_window_unavailable
    )
    return CurrentWeatherScope(
        revision=current_scope.revision + 1,
        location=next_location,
        window=next_window,
        location_unavailable=next_location_unavailable,
        window_unavailable=next_window_unavailable,
    )


def apply_current_weather_scope_resolution(
    current_scope: CurrentWeatherScope,
    resolution: CurrentWeatherScopeResolution | None,
    *,
    source_turn_id: str,
    source_sequence: int,
) -> CurrentWeatherScope:
    """Apply an explicit atomic resolution, preserving retained provenance."""

    if resolution is None:
        return current_scope
    _validate_resolution_against_scope(current_scope, resolution)
    if resolution.location_action == "retain":
        next_location = current_scope.location
        next_location_unavailable = current_scope.location_unavailable
    elif (
        resolution.location_action == "set"
        and resolution.location_scope is not None
    ):
        next_location = WeatherScopeLocationAuthority(
            value=resolution.location_scope,
            source_turn_id=source_turn_id,
            source_sequence=source_sequence,
        )
        next_location_unavailable = None
    elif resolution.location_action == "unavailable":
        next_location = None
        next_location_unavailable = WeatherScopeUnavailableAuthority(
            source_turn_id=source_turn_id,
            source_sequence=source_sequence,
        )
    else:
        next_location = None
        next_location_unavailable = None
    if resolution.window_action == "retain":
        next_window = current_scope.window
        next_window_unavailable = current_scope.window_unavailable
    elif (
        resolution.window_action == "set"
        and resolution.requested_window is not None
    ):
        next_window = WeatherScopeWindowAuthority(
            value=resolution.requested_window,
            source_turn_id=source_turn_id,
            source_sequence=source_sequence,
        )
        next_window_unavailable = None
    elif resolution.window_action == "unavailable":
        next_window = None
        next_window_unavailable = WeatherScopeUnavailableAuthority(
            source_turn_id=source_turn_id,
            source_sequence=source_sequence,
        )
    else:
        next_window = None
        next_window_unavailable = None
    preserves_pending_binding = (
        resolution.preserve_pending_source_turn_id is not None
    )
    return CurrentWeatherScope(
        revision=current_scope.revision + 1,
        location=next_location,
        window=next_window,
        location_unavailable=next_location_unavailable,
        window_unavailable=next_window_unavailable,
        pending_question=resolution.pending_question,
        pending_source_turn_id=(
            (
                current_scope.pending_source_turn_id
                if preserves_pending_binding
                else source_turn_id
            )
            if resolution.pending_question is not None
            else None
        ),
        pending_source_sequence=(
            (
                current_scope.pending_source_sequence
                if preserves_pending_binding
                else source_sequence
            )
            if resolution.pending_question is not None
            else None
        ),
    )


def _validate_resolution_against_scope(
    current_scope: CurrentWeatherScope,
    resolution: CurrentWeatherScopeResolution,
) -> None:
    if resolution.expected_scope_revision != current_scope.revision:
        raise ValueError("weather scope revision conflict")
    if (
        resolution.location_action == "retain"
        and current_scope.location is None
        and current_scope.location_unavailable is None
    ):
        raise ValueError("weather scope cannot retain a missing location")
    if (
        resolution.window_action == "retain"
        and current_scope.window is None
        and current_scope.window_unavailable is None
    ):
        raise ValueError("weather scope cannot retain a missing window")
    effective_location_unavailable = (
        current_scope.location_unavailable is not None
        if resolution.location_action == "retain"
        else resolution.location_action == "unavailable"
    )
    effective_window_unavailable = (
        current_scope.window_unavailable is not None
        if resolution.window_action == "retain"
        else resolution.window_action == "unavailable"
    )
    if resolution.pending_question is not None and (
        effective_location_unavailable or effective_window_unavailable
    ):
        raise ValueError(
            "pending weather question requires every component to be askable"
        )
    if resolution.preserve_pending_source_turn_id is not None and (
        resolution.pending_question != current_scope.pending_question
        or resolution.preserve_pending_source_turn_id
        != current_scope.pending_source_turn_id
        or current_scope.pending_source_sequence is None
    ):
        raise ValueError("weather scope pending source preservation conflict")
    completion_handle = resolution.complete_pending_source_turn_id
    if completion_handle is not None:
        if (
            current_scope.pending_question is None
            or current_scope.pending_source_turn_id is None
            or current_scope.pending_source_sequence is None
            or completion_handle != current_scope.pending_source_turn_id
        ):
            raise ValueError("weather scope pending completion conflict")
        if current_scope.pending_question == "event_location":
            target_is_set = resolution.location_action == "set"
            counterpart_action = resolution.window_action
            counterpart_exists = (
                current_scope.window is not None
                or current_scope.window_unavailable is not None
            )
            opposite_question = "event_date"
        else:
            target_is_set = resolution.window_action == "set"
            counterpart_action = resolution.location_action
            counterpart_exists = (
                current_scope.location is not None
                or current_scope.location_unavailable is not None
            )
            opposite_question = "event_location"
        if counterpart_action == "set":
            expected_pending_question = None
        elif counterpart_exists and counterpart_action == "retain":
            expected_pending_question = None
        elif not counterpart_exists and counterpart_action == "clear":
            expected_pending_question = opposite_question
        else:
            raise ValueError("weather scope pending completion conflict")
        if (
            not target_is_set
            or resolution.pending_question != expected_pending_question
        ):
            raise ValueError("weather scope pending completion conflict")
    decline_handle = resolution.decline_pending_source_turn_id
    supersede_handle = resolution.supersede_pending_source_turn_id
    if supersede_handle is not None and (
        current_scope.pending_question is None
        or current_scope.pending_source_turn_id is None
        or current_scope.pending_source_sequence is None
        or supersede_handle != current_scope.pending_source_turn_id
        or resolution.location_action == "retain"
        or resolution.window_action == "retain"
    ):
        raise ValueError("weather scope pending supersession conflict")
    pending_target_is_unavailable = (
        current_scope.pending_question == "event_location"
        and resolution.location_action == "unavailable"
    ) or (
        current_scope.pending_question == "event_date"
        and resolution.window_action == "unavailable"
    )
    if decline_handle is not None:
        if current_scope.pending_question == "event_location":
            target_is_unavailable = (
                resolution.location_action == "unavailable"
            )
        else:
            target_is_unavailable = (
                resolution.window_action == "unavailable"
            )
        if (
            current_scope.pending_question is None
            or current_scope.pending_source_turn_id is None
            or current_scope.pending_source_sequence is None
            or decline_handle != current_scope.pending_source_turn_id
            or resolution.pending_question is not None
            or not target_is_unavailable
        ):
            raise ValueError("weather scope pending decline conflict")
    elif pending_target_is_unavailable and supersede_handle is None:
        raise ValueError("weather scope pending decline requires exact handle")
    elif completion_handle is None and supersede_handle is None and (
        (
            current_scope.pending_question == "event_location"
            and resolution.location_action == "set"
            and resolution.window_action == "retain"
            and resolution.pending_question is None
        )
        or (
            current_scope.pending_question == "event_date"
            and resolution.window_action == "set"
            and resolution.location_action == "retain"
            and resolution.pending_question is None
        )
    ):
        raise ValueError("weather scope pending completion requires exact handle")


def _validate_source_turn_id(value: str) -> str:
    if (
        value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError("weather scope source turn must be trimmed and single-line")
    return value
