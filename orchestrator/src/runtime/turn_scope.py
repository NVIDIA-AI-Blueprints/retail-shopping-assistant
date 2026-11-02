# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request-local state for one shopper turn."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from ..conversation_products import ProductEvidence
from .control_signals import ControlSignal, control


@dataclass
class CatalogRepairState:
    """Bookkeeping for at most one in-flight catalog-search repair.

    A rejected search may be repaired once. These fields remember what was
    rejected so the repair cannot silently change product scope or drop
    capability-validated constraints.
    """

    failed_repair_scope_key: str | None = None
    pending_taxonomy_constraints: dict[str, Any] | None = None
    pending_schema_requirements: list[str] = field(default_factory=list)
    #: The last scope this turn rejected, as sent, so a second identical
    #: attempt is the last one instead of looping to the recursion limit.
    last_rejected_scope: str | None = None


@dataclass
class TurnScope:
    """Everything one shopper turn's tools mutate while they run.

    Holding this state on one object, rather than in variables shared by the
    tools' enclosing scope, is what lets each tool be read, tested, and moved
    on its own.
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
    #: What each answered scope returned, so asking again is answered rather
    #: than refused. Keyed by both shopper scope and catalog scope, because a
    #: repeat arrives as either.
    answered_scopes: dict[str, str | tuple[str, dict[str, Any]]] = field(
        default_factory=dict
    )
    # Forecast budget. A paid external call, and one turn never needs many:
    # a shopper is at one event, on one date. Guarded because roles can run
    # concurrently.
    weather_lock: Lock = field(default_factory=Lock)
    weather_calls: int = 0

    # Product-detail budget. Deliberately not lock-guarded, preserving existing
    # behavior; the search counter above is guarded and this one never was.
    product_detail_reads: int = 0

    # Answers already given this turn, keyed by tool name and arguments.
    # A repeat is answered from here. The second repeat of the same arguments
    # stops tool use outright, because a note saying "already answered" is
    # itself something to loop against.
    repeat_lock: Lock = field(default_factory=Lock)
    answers_given: dict[str, str] = field(default_factory=dict)
    repeats_refused: Counter[str] = field(default_factory=Counter)

    # Historical product resolution. Guarded by ``resolution_lock``.
    resolution_lock: Lock = field(default_factory=Lock)
    #: Set when a call actually resolved something. A call that resolved
    #: nothing does not spend the turn's only attempt, so the correction the
    #: refusal itself asks for can still be made.
    product_resolution_used: bool = False
    #: Attempts made, resolving or not, so a call that keeps missing still
    #: terminates.
    product_resolution_attempts: int = 0

    repair: CatalogRepairState = field(default_factory=CatalogRepairState)

    # Cart mutation keys issued this turn. The memory service answers a key it
    # has already seen by replaying that result and changing nothing, so the
    # same change made twice in one turn needs a second key.
    cart_key_lock: Lock = field(default_factory=Lock)
    cart_keys_issued: Counter[str] = field(default_factory=Counter)

    def cart_mutation_key(self, base: str) -> str:
        """`base` the first time this turn, then `base#2`, `base#3`, ..."""

        with self.cart_key_lock:
            self.cart_keys_issued[base] += 1
            issued = self.cart_keys_issued[base]
        return base if issued == 1 else f"{base}#{issued}"

    def answer_already_given(
        self,
        tool_name: str,
        arguments: str,
    ) -> str | tuple[str, dict[str, Any]] | None:
        """Return this turn's answer for these arguments, or None if new."""

        key = f"{tool_name}({arguments})"
        with self.repeat_lock:
            held = self.answers_given.get(key)
            if held is None:
                return None
            self.repeats_refused[key] += 1
            repeats = self.repeats_refused[key]
        if repeats == 1:
            return (
                f"ALREADY_ANSWERED_THIS_TURN: {tool_name} was called with "
                "these same arguments earlier this turn, and nothing since "
                "could have changed the answer. It was:\n\n" + held
            )
        # Saying "you already asked" is itself something to loop against.
        return control(
            f"STOP_TOOL_USE: {tool_name} has been called three times this "
            "turn with the same arguments. Do not call any more tools. "
            "Answer the shopper from the evidence already gathered.",
            ControlSignal.STOP_TOOL_USE,
        )

    def remember_answer(
        self,
        tool_name: str,
        arguments: str,
        answer: str,
    ) -> None:
        """Record what these arguments answered, for the rest of the turn."""

        with self.repeat_lock:
            self.answers_given.setdefault(f"{tool_name}({arguments})", answer)
