# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Closed semantic-decision contract for weather-scope resolution."""

from __future__ import annotations

import json
from datetime import date
from typing import Any, Literal, Mapping, Sequence

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)


WEATHER_SCOPE_RESOLVER_SYSTEM_PROMPT = """You resolve two independent questions about the shopper's current weather-planning context.

This is a business-tool-disabled semantic decision. Use the rolling summary and
recent turns only to understand conversational meaning. They do not establish
location, date, forecast, product, cart, or policy facts.

The current scope's location and date carry source_sequence. Its pending
question carries both source_sequence and an opaque source_turn_id handle. The
scope_source_turns lane contains the exact prior shopper turns at those
sequences. Treat those turns as the semantic identity of the subject whose
typed authority or unanswered question is currently stored.
Compare the current query's event, trip, or ordinary weather-planning subject
with that identity before selecting subject_relation. An additional event or
trip is a new subject even when the shopper wants the same kind of styling
help. A product search, comparison, or refinement inside the current styling
thread does not introduce a new weather-planning subject.

Choose exactly one subject_relation:
- same_subject: the shopper continues, changes, corrects, or supplies context
  for the same event, trip, or ordinary weather-planning subject.
- new_subject: the shopper introduces a different event, trip, or ordinary
  weather-planning subject.
- unchanged: the current subject and its location/date scope are unchanged.
- unclear: the relationship cannot be established confidently.

Separately choose exactly one pending_disposition:
- not_addressed: the current query neither answers nor explicitly resumes the
  supplied pending question.
- answered: the shopper answers only the supplied pending question and leaves
  the stored opposite component unchanged. Use this only with same_subject.
- resume_requested: the shopper explicitly asks what information is still
  needed, so the existing pending question should be asked again without
  changing its scope or source binding. Use this only with unchanged.

For answered or resume_requested, exactly echo the pending question's opaque
source_turn_id in the top-level pending_source_turn_id field. Omit that handle
for not_addressed. If an answer also changes or withdraws the stored opposite
component, use same_subject/not_addressed so activation can author both current
facts without pending authority.

Return only these semantic axes. Do not extract, normalize, copy, or author
location, date, scope revision, or component actions; the main shopper
activation supplies current-turn scope facts once and the server validates and
compiles them. Follow-up question selection remains the main shopper skill's
responsibility.

Make exactly one required WeatherScopeResolverDecision control call. This is a
schema-only control channel, not a business tool. Do not include prose or
Markdown."""


WeatherScopeSubjectRelation = Literal[
    "same_subject",
    "new_subject",
    "unchanged",
    "unclear",
]
WeatherScopePendingDisposition = Literal[
    "not_addressed",
    "answered",
    "resume_requested",
]

_VALID_RESOLVER_OUTCOMES = frozenset(
    {
        ("same_subject", "not_addressed"),
        ("same_subject", "answered"),
        ("new_subject", "not_addressed"),
        ("unchanged", "not_addressed"),
        ("unchanged", "resume_requested"),
        ("unclear", "not_addressed"),
    }
)


def is_canonical_weather_scope_resolver_outcome(
    subject_relation: object,
    pending_disposition: object,
) -> bool:
    """Return whether the two semantic axes form one supported outcome."""

    outcome = (subject_relation, pending_disposition)
    try:
        return outcome in _VALID_RESOLVER_OUTCOMES
    except TypeError:
        return False


class WeatherScopeResolverDecision(BaseModel):
    """Orthogonal subject and pending-question semantic controls."""

    model_config = ConfigDict(extra="forbid", strict=True)

    subject_relation: WeatherScopeSubjectRelation
    pending_disposition: WeatherScopePendingDisposition
    pending_source_turn_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
        description=(
            "Opaque handle from current_scope.pending_question. Copy it "
            "exactly for answered or resume_requested; omit it for "
            "not_addressed."
        ),
    )

    @field_validator("pending_source_turn_id")
    @classmethod
    def validate_pending_source_turn_id(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None
        if (
            value != value.strip()
            or any(
                ord(character) < 32 or ord(character) == 127
                for character in value
            )
        ):
            raise ValueError(
                "pending source turn must be trimmed and single-line"
            )
        return value

    @model_validator(mode="after")
    def _validate_decision_shape(self) -> "WeatherScopeResolverDecision":
        outcome = (self.subject_relation, self.pending_disposition)
        if not is_canonical_weather_scope_resolver_outcome(*outcome):
            raise ValueError("invalid subject/pending resolver combination")
        pending_handle_required = self.pending_disposition in {
            "answered",
            "resume_requested",
        }
        if pending_handle_required != (self.pending_source_turn_id is not None):
            raise ValueError(
                "pending_source_turn_id is required exactly when the pending "
                "question is answered or resumed"
            )
        return self


def build_weather_scope_resolver_prompt(
    *,
    current_query: str,
    current_scope_json: Mapping[str, Any],
    current_utc_date: date,
    rolling_summary: str,
    scope_source_turns: Sequence[Mapping[str, Any]],
    recent_turns: Sequence[Mapping[str, Any]],
) -> str:
    """Build one compact resolver input with explicit context authority lanes."""

    return json.dumps(
        {
            "current_query": current_query,
            "current_scope": dict(current_scope_json),
            "current_utc_date": current_utc_date.isoformat(),
            "scope_subject_context": {
                "authority": "source_sequence_bound_semantic_identity_only",
                "turns": [dict(turn) for turn in scope_source_turns],
            },
            "semantic_context": {
                "authoritative": False,
                "rolling_summary": rolling_summary,
                "recent_turns": [dict(turn) for turn in recent_turns],
            },
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def parse_weather_scope_resolver_tool_call(
    message: Any,
) -> WeatherScopeResolverDecision | None:
    """Validate exactly one forced control call and fail closed otherwise."""

    calls = (
        message.get("tool_calls")
        if isinstance(message, dict)
        else getattr(message, "tool_calls", None)
    )
    if not isinstance(calls, list) or len(calls) != 1:
        return None
    call = calls[0]
    name = (
        call.get("name")
        if isinstance(call, dict)
        else getattr(call, "name", None)
    )
    arguments = (
        call.get("args")
        if isinstance(call, dict)
        else getattr(call, "args", None)
    )
    if (
        name != WeatherScopeResolverDecision.__name__
        or not isinstance(arguments, dict)
    ):
        return None
    try:
        return WeatherScopeResolverDecision.model_validate(
            arguments,
            strict=True,
        )
    except ValidationError:
        return None
