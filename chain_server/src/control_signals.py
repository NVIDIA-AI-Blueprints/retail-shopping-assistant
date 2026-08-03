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


#: Artifact key holding the recorded signals for one tool call.
SIGNALS_KEY = "control_signals"

#: Artifact key holding committed commerce effects for one tool call.
EFFECTS_KEY = "committed_effects"


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
