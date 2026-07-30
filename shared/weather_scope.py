# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Closed contract for one conversation's current weather-planning scope."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .weather_receipts import WeatherLocationScope, WeatherReceiptWindow


class _WeatherScopeModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


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


class CurrentWeatherScope(_WeatherScopeModel):
    """Memory-owned singleton; not an event registry or evidence record."""

    revision: int = Field(default=0, ge=0)
    location: WeatherScopeLocationAuthority | None = None
    window: WeatherScopeWindowAuthority | None = None

    @model_validator(mode="after")
    def _validate_initial_scope(self) -> "CurrentWeatherScope":
        if self.revision == 0 and (
            self.location is not None or self.window is not None
        ):
            raise ValueError("initial weather scope cannot contain authority")
        return self


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
    return CurrentWeatherScope(
        revision=current_scope.revision + 1,
        location=next_location,
        window=next_window,
    )


def _validate_source_turn_id(value: str) -> str:
    if (
        value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError("weather scope source turn must be trimmed and single-line")
    return value
