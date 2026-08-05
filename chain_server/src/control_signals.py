# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed tool-loop control outcomes carried beside the model-visible text.

A tool result serves two audiences. The model reads rendered text; the runtime
needs to know what the tool decided. Encoding the second into the first forces
the runtime to reverse-engineer its own decisions out of prose written for the
model.

Here the decision travels as a LangChain tool artifact, which rides on the
``ToolMessage`` itself. That means it is checkpointed with the message it
describes and needs no correlation key: the signal is attached to its own tool
call by construction.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class ControlSignal(StrEnum):
    """One deterministic tool-loop outcome, recorded where it is decided."""

    STOP_TOOL_USE = "stop_tool_use"
    SCOPE_COMPLETE = "scope_complete"
    BUDGET_EXHAUSTED = "budget_exhausted"
    UNSUPPORTED_TAXONOMY = "unsupported_taxonomy"
    UNSUPPORTED_CONSTRAINT = "unsupported_constraint"
    CONSTRAINT_REVIEW = "constraint_review"
    REQUEST_REJECTED = "request_rejected"
    SEARCH_SUCCEEDED = "search_succeeded"


class SearchRejection(StrEnum):
    """The gate that turned one catalog search scope back.

    Nine gates in ``catalog_search`` render the same model-visible prefix, so a
    refusal reached diagnostics as one undifferentiated reason: in one full
    evaluation run that left 21 of 48 refusals impossible to attribute to a
    gate. The gate that makes the decision records it here. Nothing the model
    reads changes -- this is the same move as typed tool evidence, applied to
    the rejection path.
    """

    # Admission, before anything is parsed.
    REPAIR_CHANGED_PRODUCT_SCOPE = "repair_changed_product_scope"

    # Argument validation against current catalog capabilities.
    CAPABILITIES_SCHEMA_MISMATCH = "capabilities_schema_mismatch"

    # Provenance review: what the repair had to preserve, and what it changed.
    REPAIR_CHANGED_CONSTRAINTS = "repair_changed_constraints"
    TAXONOMY_NOT_ADVERTISED_FOR_SCOPE = "taxonomy_not_advertised_for_scope"
    CONSTRAINT_REPAIR_CHANGED_REQUEST = "constraint_repair_changed_request"
    SHOPPER_SCOPE_TAXONOMY_MISMATCH = "shopper_scope_taxonomy_mismatch"
    REQUIREMENT_PROVENANCE_UNESTABLISHED = "requirement_provenance_unestablished"
    CONSTRAINT_REVIEW_REQUIRED = "constraint_review_required"
    EXACT_TAXONOMY_NOT_ADVERTISED = "exact_taxonomy_not_advertised"
    ADVERTISED_MATCH_REPORTED_AS_GAP = "advertised_match_reported_as_gap"

    # Outcome for a product type no advertised taxonomy faithfully covers.
    NO_ADVERTISED_TAXONOMY_MATCH = "no_advertised_taxonomy_match"

    # Planning against the capability contract.
    UNSUPPORTED_CATALOG_TAXONOMY = "unsupported_catalog_taxonomy"
    UNSUPPORTED_CATALOG_CONSTRAINT = "unsupported_catalog_constraint"
    UNSUPPORTED_SEARCH_MODE = "unsupported_search_mode"

    # Claiming this turn's search budget.
    DUPLICATE_SHOPPER_SCOPE = "duplicate_shopper_scope"
    #: Diagnostics counts duplicates by this exact value; do not rename it
    #: without updating the reader that sets ``duplicate`` on a tool call.
    DUPLICATE_CATALOG_SCOPE = "duplicate_catalog_scope"
    CATALOG_SEARCH_LIMIT = "catalog_search_limit"


#: Artifact key holding the recorded signals for one tool call.
SIGNALS_KEY = "control_signals"

#: Artifact key holding committed commerce effects for one tool call.
EFFECTS_KEY = "committed_effects"

#: Artifact key holding, per searched scope and in scope order, the gate that
#: turned that scope back -- ``None`` for a scope that was not turned back.
REJECTIONS_KEY = "scope_rejections"


def control(text: str, *signals: ControlSignal) -> tuple[str, dict[str, Any]]:
    """Return one tool result whose control outcome is typed, not parsed.

    ``text`` is what the model sees; ``signals`` are what the runtime acts on.
    """

    return text, {SIGNALS_KEY: [str(signal) for signal in signals]}


def signals_of(message: Any) -> frozenset[str]:
    """Read recorded signals from one tool message without parsing its text."""

    artifact = getattr(message, "artifact", None)
    if not isinstance(artifact, dict):
        return frozenset()
    recorded = artifact.get(SIGNALS_KEY)
    if not isinstance(recorded, list):
        return frozenset()
    return frozenset(str(signal) for signal in recorded)


def rejections_of(message: Any) -> list[Any]:
    """Read the per-scope gate codes recorded on one tool message.

    Returns one entry per searched scope, in scope order, so a call that
    refused two of three roles stays distinguishable from one that refused all
    three. A message that carries no codes returns an empty list, which every
    reader must treat as "unknown", not as "nothing was refused".
    """

    artifact = (
        message.get("artifact")
        if isinstance(message, dict)
        else getattr(message, "artifact", None)
    )
    if not isinstance(artifact, dict):
        return []
    recorded = artifact.get(REJECTIONS_KEY)
    return list(recorded) if isinstance(recorded, list) else []


def normalize_tool_result(result: Any) -> tuple[str, dict[str, Any] | None]:
    """Adapt a plain-string tool return to the artifact contract.

    ``response_format="content_and_artifact"`` requires every return to be a
    two-tuple. Existing returns that carry no control outcome stay untouched in
    the tool body and are normalised here instead.
    """

    if isinstance(result, tuple):
        return result
    return result, None


def committed_effect(
    text: str,
    *,
    operation: str,
    idempotency_key: str,
    product_id: str | None = None,
    cart_line_id: str | None = None,
    quantity: int | None = None,
) -> tuple[str, dict[str, Any]]:
    """Return a mutation result that records what was actually committed.

    A committed effect must survive the turn even when the turn fails. Carrying
    it on the tool artifact means the runtime can still find it in the graph
    snapshot after an error, rather than inferring from prose that a cart change
    may or may not have happened.
    """

    effect: dict[str, Any] = {
        "operation": operation,
        "idempotency_key": idempotency_key,
    }
    if product_id is not None:
        effect["product_id"] = product_id
    if cart_line_id is not None:
        effect["cart_line_id"] = cart_line_id
    if quantity is not None:
        effect["quantity"] = quantity
    return text, {EFFECTS_KEY: [effect]}


def committed_effects_in(messages: Any) -> list[dict[str, Any]]:
    """Collect committed effects recorded across one turn's tool messages."""

    found: list[dict[str, Any]] = []
    for message in messages or ():
        artifact = getattr(message, "artifact", None)
        if not isinstance(artifact, dict):
            continue
        recorded = artifact.get(EFFECTS_KEY)
        if isinstance(recorded, list):
            found.extend(e for e in recorded if isinstance(e, dict))
    return found
