# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Events recording who the shopper is shopping for, and what was assumed."""

from __future__ import annotations

from typing import Any

from .conversation_memory import ConversationEvent


def _wearer_audience_events(
    state: Any,
    identity: Any,
    *,
    field_name: str,
) -> list[ConversationEvent]:
    """Record an audience this turn declared, so the next turn inherits it.

    Read from the turn's own diagnostics rather than the graph messages: the
    per-product scope stamp already carries the filters each role was searched
    with, and zero-result scopes carry theirs too, so nothing new has to be
    threaded through finalize.

    The server records what the model declared and never reads the shopper's
    prose to work out who an item is for, which it is forbidden to do. A turn
    that filtered on nothing declares nothing and leaves an earlier declaration
    standing -- silence is how the model forgets, not how a shopper changes
    their mind.
    """

    if not field_name:
        return []
    diagnostics = getattr(state, "agent_diagnostics", None) or {}
    scopes: list[Any] = [
        (record or {}).get("search_scope")
        for record in (diagnostics.get("product_evidence") or [])
    ]
    scopes.extend(diagnostics.get("catalog_scope_outcomes") or [])
    declared: list[str] = []
    for scope in scopes:
        if not isinstance(scope, dict):
            continue
        values = (scope.get("confirmed_filters") or {}).get(field_name)
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list):
            continue
        for value in values:
            text = str(value)
            if text and text not in declared:
                declared.append(text)
    if not declared:
        return []
    return [
        ConversationEvent(
            event_key=f"wearer-audience:{identity.request_id}",
            event_type="wearer_audience_declared",
            source_kind="runtime",
            payload={"audience": declared[:8]},
        )
    ]


def _audience_assumption_events(
    state: Any,
    identity: Any,
) -> list[ConversationEvent]:
    """Record that this conversation has now been told what was assumed.

    Catalog search wrote it while the turn ran, so this reads the turn's own
    state rather than re-deriving anything. Nothing is recorded on a turn that
    disclosed nothing, which leaves an earlier disclosure standing: the shopper
    is owed the sentence once, not once per search.

    Deliberately not a `wearer_audience_declared`. That event is a fact the
    model established and may narrow later searches with; this one is only the
    shop admitting a guess, and must never scope anything.
    """

    disclosed = [str(value) for value in (getattr(state, "disclosed_audience", None) or [])]
    if not disclosed:
        return []
    return [
        ConversationEvent(
            event_key=f"audience-assumption:{identity.request_id}",
            event_type="audience_assumption_disclosed",
            source_kind="runtime",
            payload={"audience": disclosed[:8]},
        )
    ]


def _system_identification_events(
    state: Any,
    identity: Any,
) -> list[ConversationEvent]:
    """Record which products the record itself picked this turn.

    Establishment was scoped to the message that produced it. A shopper chose
    "the first pairing" in one turn and, three turns later -- having only
    answered the assistant's own questions -- was still being told the products
    were not established, because answering a question names nothing. The same
    add was refused three times and accepted on the fourth, when they finally
    typed both catalog names. The request never changed.

    The choice itself is durable: it was made against a set of products the
    record wrote down. So it is recorded here, and the memory service files it
    against the set it was made from -- which is what makes it lapse when a new
    set is shown, and nothing else does.
    """

    refs = [
        str(ref)
        for ref in (getattr(state, "system_identified_products", None) or [])
        if ref
    ]
    if not refs:
        return []
    return [
        ConversationEvent(
            event_key=f"system-identified:{identity.request_id}",
            event_type="historical_reference_resolved",
            source_kind="runtime",
            payload={"product_refs": refs[:16]},
        )
    ]


def _turn_audience_events(
    state: Any,
    identity: Any,
    *,
    field_name: str,
) -> list[ConversationEvent]:
    """Record what this turn settled about who is being shopped for.

    A declaration outranks an assumption made earlier in the same turn. That
    is the self-correcting case: an unscoped search returns womenswear, the
    evidence says so, the model recognises the shopper named a husband and
    searches again with the values that suit him. Recording both would leave
    the conversation carrying a guess the turn had already overturned.
    """

    declared = _wearer_audience_events(state, identity, field_name=field_name)
    if declared:
        return declared
    return _audience_assumption_events(state, identity)
