# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immutable representative-shopper registry owned by conversation memory."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Path as ApiPath
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from .models import ShopperProfile


EXPECTED_SHOPPER_TYPES = frozenset(
    {
        "occasion_driven_explorer",
        "strict_budget_style_mixer",
        "impatient_decisive",
        "skeptical_researcher",
        "iterative_refiner",
    }
)
EXPECTED_SHOPPER_PROFILE_IDS = frozenset(
    {
        "shopper_alex",
        "shopper_casey",
        "shopper_jordan",
        "shopper_morgan",
        "shopper_riley",
    }
)
SHOPPER_PROFILE_COUNT = 5
SHOPPER_PROFILE_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$"
_DEFAULT_CONFIG_ROOT = Path("/app/shared/configs")
_SEED_RELATIVE_PATH = Path("memory_retriever/shopper_profiles.json")


class ShopperProfileContract(BaseModel):
    """Closed public and seed contract for one representative shopper."""

    model_config = ConfigDict(extra="forbid", from_attributes=True, strict=True)

    shopper_profile_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        pattern=SHOPPER_PROFILE_ID_PATTERN,
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

    @field_validator("behavior")
    @classmethod
    def _require_single_line_behavior(cls, value: str) -> str:
        if "\n" in value or "\r" in value:
            raise ValueError("profile behavior must be one line")
        return value


class ShopperProfileBootstrapError(RuntimeError):
    """Startup failure caused by invalid or conflicting managed profile data."""


def shopper_profile_seed_path() -> Path:
    """Resolve the operator-managed profile seed manifest."""

    config_root = Path(
        os.environ.get("SHARED_CONFIG_ROOT", str(_DEFAULT_CONFIG_ROOT))
    )
    return config_root / _SEED_RELATIVE_PATH


def load_shopper_profile_seed(
    seed_path: str | Path | None = None,
) -> tuple[ShopperProfileContract, ...]:
    """Load and validate the complete managed profile manifest."""

    path = Path(seed_path) if seed_path is not None else shopper_profile_seed_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ShopperProfileBootstrapError(
            "Representative shopper seed manifest is unavailable or invalid."
        ) from exc

    if not isinstance(raw, list):
        raise ShopperProfileBootstrapError(
            "Representative shopper seed manifest must be a JSON array."
        )

    try:
        profiles = tuple(ShopperProfileContract.model_validate(item) for item in raw)
    except (TypeError, ValidationError) as exc:
        raise ShopperProfileBootstrapError(
            "Representative shopper seed manifest contains an invalid profile."
        ) from exc

    ids = [profile.shopper_profile_id for profile in profiles]
    types = [profile.shopper_type for profile in profiles]
    if (
        len(profiles) != SHOPPER_PROFILE_COUNT
        or len(ids) != len(set(ids))
        or len(types) != len(set(types))
        or set(ids) != EXPECTED_SHOPPER_PROFILE_IDS
        or set(types) != EXPECTED_SHOPPER_TYPES
    ):
        raise ShopperProfileBootstrapError(
            "Representative shopper seed manifest must contain the exact managed set."
        )
    return profiles


def bootstrap_shopper_profiles(
    session_factory: Callable[[], Any],
    *,
    seed_path: str | Path | None = None,
) -> None:
    """Insert missing managed rows and fail closed on any registry drift."""

    profiles = load_shopper_profile_seed(seed_path)
    expected_by_id = {
        profile.shopper_profile_id: profile for profile in profiles
    }
    db = session_factory()
    try:
        db.execute(text("BEGIN IMMEDIATE"))
        existing = db.query(ShopperProfile).all()
        existing_by_id = {
            profile.shopper_profile_id: profile for profile in existing
        }
        extra_ids = set(existing_by_id).difference(expected_by_id)
        if extra_ids:
            raise ShopperProfileBootstrapError(
                "Representative shopper registry contains unmanaged rows."
            )

        existing_by_type = {
            profile.shopper_type: profile for profile in existing
        }
        for profile_id, expected in expected_by_id.items():
            stored = existing_by_id.get(profile_id)
            if stored is not None:
                if _profile_values(stored) != expected.model_dump(mode="python"):
                    raise ShopperProfileBootstrapError(
                        "Representative shopper registry conflicts with its seed."
                    )
                continue

            type_owner = existing_by_type.get(expected.shopper_type)
            if type_owner is not None:
                raise ShopperProfileBootstrapError(
                    "Representative shopper type is owned by a different profile."
                )
            db.add(ShopperProfile(**expected.model_dump(mode="python")))

        db.flush()
        persisted = db.query(ShopperProfile).all()
        if len(persisted) != SHOPPER_PROFILE_COUNT:
            raise ShopperProfileBootstrapError(
                "Representative shopper registry did not reach its managed state."
            )
        db.commit()
    except ShopperProfileBootstrapError:
        db.rollback()
        raise
    except IntegrityError as exc:
        db.rollback()
        raise ShopperProfileBootstrapError(
            "Representative shopper registry violates its immutable schema."
        ) from exc
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def create_shopper_profile_router(
    get_db: Callable[..., Any],
) -> APIRouter:
    """Create read-only profile endpoints with request-scoped DB sessions."""

    router = APIRouter()

    @router.get(
        "/shopper-profiles",
        response_model=list[ShopperProfileContract],
    )
    async def list_shopper_profiles(
        db=Depends(get_db),
    ) -> Sequence[ShopperProfile]:
        return (
            db.query(ShopperProfile)
            .order_by(ShopperProfile.shopper_profile_id.asc())
            .all()
        )

    @router.get(
        "/shopper-profiles/{shopper_profile_id}",
        response_model=ShopperProfileContract,
    )
    async def get_shopper_profile(
        shopper_profile_id: str = ApiPath(
            ...,
            min_length=1,
            max_length=64,
            pattern=SHOPPER_PROFILE_ID_PATTERN,
        ),
        db=Depends(get_db),
    ) -> ShopperProfile:
        profile = (
            db.query(ShopperProfile)
            .filter(ShopperProfile.shopper_profile_id == shopper_profile_id)
            .first()
        )
        if profile is None:
            raise HTTPException(
                status_code=404,
                detail="Shopper profile not found",
            )
        return profile

    return router


def _profile_values(profile: ShopperProfile) -> dict[str, str]:
    return {
        "shopper_profile_id": profile.shopper_profile_id,
        "display_name": profile.display_name,
        "shopper_type": profile.shopper_type,
        "behavior": profile.behavior,
        "zipcode": profile.zipcode,
    }
