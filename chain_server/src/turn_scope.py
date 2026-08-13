# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request-local state for one shopper turn."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from .conversation_products import ProductEvidence


@dataclass
class CatalogRepairState:
    """Bookkeeping for at most one in-flight catalog-search repair.

    A rejected search may be repaired once. These fields remember what was
    rejected so the repair cannot silently change product scope, drop
    capability-validated constraints, or re-review a scope already reviewed.
    """

    failed_repair_scope_key: str | None = None
    failed_constraint_scope_key: str | None = None
    constraint_reviewed_scopes: set[str] = field(default_factory=set)
    pending_constraint_reviews: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )
    pending_taxonomy_constraints: dict[str, Any] | None = None
    pending_no_direct_constraint_clear: bool = False
    pending_schema_requirements: list[str] = field(default_factory=list)


@dataclass
class TurnScope:
    """Everything one shopper turn's tools mutate while they run.

    Each field here was previously a ``nonlocal`` inside ``_create_agent``, so
    every tool was welded to one lexical scope and none could be read, tested,
    or relocated independently. Owning this state explicitly is what makes the
    tools separable; it deliberately changes no behavior.
    """

    # Evidence and rendering. ``retrieved`` is deliberately the same dict object
    # as ``State.retrieved``: tools mutate it in place and the runtime reads it.
    product_evidence: ProductEvidence = field(default_factory=ProductEvidence)
    retrieved: dict[str, str] = field(default_factory=dict)

    # Catalog search accounting. Guarded by ``catalog_lock``.
    catalog_lock: Lock = field(default_factory=Lock)
    catalog_searches: int = 0
    searched_catalog_scopes: list[dict[str, Any]] = field(default_factory=list)
    searched_shopper_scopes: set[tuple[str, str]] = field(default_factory=set)

    # Forecast budget. A paid external call, and one turn never needs many:
    # a shopper is at one event, on one date. Guarded because roles can run
    # concurrently.
    weather_lock: Lock = field(default_factory=Lock)
    weather_calls: int = 0

    # Product-detail budget. Deliberately not lock-guarded, preserving existing
    # behavior; the search counter above is guarded and this one never was.
    product_detail_reads: int = 0

    # Historical product resolution. Guarded by ``resolution_lock``.
    resolution_lock: Lock = field(default_factory=Lock)
    #: Set when a call actually resolved something. A call that resolved
    #: nothing used to spend the turn's only attempt, so the correction the
    #: refusal itself asked for could never be made.
    product_resolution_used: bool = False
    #: Attempts made, resolving or not, so a call that keeps missing still
    #: terminates.
    product_resolution_attempts: int = 0

    repair: CatalogRepairState = field(default_factory=CatalogRepairState)
