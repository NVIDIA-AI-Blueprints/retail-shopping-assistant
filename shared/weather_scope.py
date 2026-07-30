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
    """Model-selected continue/replace transition applied by memory."""

    expected_projection_version: int = Field(..., ge=0)
    action: Literal["continue", "replace"]
    location_scope: WeatherLocationScope | None = None
    requested_window: WeatherReceiptWindow | None = None

    @model_validator(mode="after")
    def _continue_requires_a_patch(self) -> "CurrentWeatherScopeTransition":
        if (
            self.action == "continue"
            and self.location_scope is None
            and self.requested_window is None
        ):
            raise ValueError("continue weather scope transition requires a patch")
        return self


def _validate_source_turn_id(value: str) -> str:
    if (
        value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError("weather scope source turn must be trimmed and single-line")
    return value
