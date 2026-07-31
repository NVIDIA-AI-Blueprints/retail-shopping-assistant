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


WEATHER_SCOPE_RESOLVER_SYSTEM_PROMPT = """You resolve whether the shopper's current message continues the current weather-planning subject.

This is a business-tool-disabled semantic decision. Use the rolling summary and
recent turns only to understand conversational meaning. They do not establish
location, date, forecast, product, cart, or policy facts.

The current scope's location and date carry source_sequence. Its pending
question carries both source_sequence and an opaque source_turn_id handle. The
scope_source_turns lane contains the exact prior shopper turns at those
sequences. Treat those turns as the semantic identity of the subject whose
typed authority or unanswered question is currently stored.
Compare the current query's event, trip, or ordinary weather-planning subject
with that identity before choosing a decision. An additional event or trip is a
new subject even when the shopper wants the same kind of styling help. A
product search, comparison, or refinement inside the current styling thread
does not by itself answer a pending context question or introduce a new
weather-planning subject.

Choose exactly one decision:
- same_subject: the shopper changes, corrects, or withdraws location or date
  for the same event, trip, or ordinary weather-planning subject. Use this
  when a reply also changes or withdraws the stored component opposite a
  pending question.
- new_subject: the shopper introduces a different event, trip, or ordinary
  weather-planning subject.
- answers_pending: the shopper answers only the supplied pending question and
  leaves the stored opposite component unchanged. Exactly echo that pending
  question's opaque source_turn_id in the top-level pending_source_turn_id
  field. If the same reply also changes or withdraws the opposite component,
  choose same_subject instead.
- unchanged: the current subject and its location/date scope are unchanged.
- unclear: the relationship cannot be established confidently.

Return only that semantic relation. Do not extract, normalize, copy, or author
location, date, scope revision, or component actions; the main shopper
activation supplies current-turn scope facts once and the server validates and
compiles them. Only answers_pending may include the top-level
pending_source_turn_id; omit it for same_subject, new_subject, unchanged, and
unclear. Follow-up question selection remains the main shopper skill's
responsibility.

Make exactly one required WeatherScopeResolverDecision control call. This is a
schema-only control channel, not a business tool. Do not include prose or
Markdown."""


WeatherScopeSemanticRelation = Literal[
    "same_subject",
    "new_subject",
    "answers_pending",
    "unchanged",
    "unclear",
]

class WeatherScopeResolverDecision(BaseModel):
    """One semantic relation plus an exact pending-answer binding."""

    model_config = ConfigDict(extra="forbid", strict=True)

    decision: WeatherScopeSemanticRelation
    pending_source_turn_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
        description=(
            "Opaque handle from current_scope.pending_question. Include it "
            "only for answers_pending and copy it exactly; omit it for every "
            "other decision."
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
        if (
            self.decision == "answers_pending"
        ) != (self.pending_source_turn_id is not None):
            raise ValueError(
                "pending_source_turn_id is required exactly for "
                "answers_pending"
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
