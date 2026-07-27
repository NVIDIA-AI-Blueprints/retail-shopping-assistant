# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed read-only boundary for representative shopper profiles."""

from __future__ import annotations

from typing import Any

import requests
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
)


_DEFAULT_TIMEOUT_SECONDS = 10.0
_MANAGED_PROFILE_COUNT = 5


class _ShopperProfileModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ShopperProfile(_ShopperProfileModel):
    """One compact representative shopper displayed by the UI."""

    shopper_profile_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$",
    )
    display_name: str = Field(..., min_length=1, max_length=80)
    shopper_type: str = Field(
        ...,
        min_length=1,
        max_length=80,
        pattern=r"^[a-z][a-z0-9_]*$",
    )
    behavior: str = Field(..., min_length=1, max_length=512)
    zipcode: str = Field(..., pattern=r"^[0-9]{5}$")

    @field_validator(
        "shopper_profile_id",
        "display_name",
        "shopper_type",
        "behavior",
        "zipcode",
    )
    @classmethod
    def _reject_outer_whitespace(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("profile text must not contain outer whitespace")
        return value


class ShopperProfilesError(RuntimeError):
    """Stable failure at the chain-to-memory profile boundary."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.retryable = retryable


_PROFILES_ADAPTER = TypeAdapter(list[ShopperProfile])


class ShopperProfilesClient:
    """Read representative shopper profiles from the memory service."""

    def __init__(
        self,
        memory_retriever_url: str,
        *,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        session: Any | None = None,
    ) -> None:
        self.memory_retriever_url = memory_retriever_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.session = session or requests

    def list_profiles(self) -> list[ShopperProfile]:
        payload = self._get_payload("/shopper-profiles")
        try:
            profiles = _PROFILES_ADAPTER.validate_python(payload)
        except (TypeError, ValueError, ValidationError) as exc:
            raise ShopperProfilesError(
                "shopper_profiles_response_invalid",
                "Shopper profiles returned an invalid response.",
                status_code=200,
            ) from exc
        profile_ids = [profile.shopper_profile_id for profile in profiles]
        shopper_types = [profile.shopper_type for profile in profiles]
        if (
            len(profiles) != _MANAGED_PROFILE_COUNT
            or len(set(profile_ids)) != len(profile_ids)
            or len(set(shopper_types)) != len(shopper_types)
        ):
            raise ShopperProfilesError(
                "shopper_profiles_response_invalid",
                "Shopper profiles returned an invalid response.",
                status_code=200,
            )
        return profiles

    def _get_payload(self, path: str) -> Any:
        try:
            response = self.session.get(
                f"{self.memory_retriever_url}{path}",
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise ShopperProfilesError(
                "shopper_profiles_request_failed",
                "Shopper profiles are temporarily unavailable.",
                retryable=True,
            ) from exc

        status_code = int(getattr(response, "status_code", 500))
        if status_code >= 400:
            raise ShopperProfilesError(
                "shopper_profiles_unavailable",
                "Shopper profiles are temporarily unavailable.",
                status_code=status_code,
                retryable=status_code >= 500,
            )
        try:
            return response.json()
        except (TypeError, ValueError) as exc:
            raise ShopperProfilesError(
                "shopper_profiles_response_invalid",
                "Shopper profiles returned an invalid response.",
                status_code=status_code,
            ) from exc
