# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Closed semantic-decision contract for weather-scope resolution."""

from __future__ import annotations

import json
from dataclasses import dataclass
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


_WEATHER_SCOPE_SUBJECT_PROMPT = """You resolve the shopper's current weather-planning subject.

This is a business-tool-disabled semantic decision. Use the rolling summary and
recent turns only to understand conversational meaning. They do not establish
location, date, forecast, product, cart, or policy facts.

The current scope's location and date carry source_sequence. Its pending
question carries both source_sequence and an opaque source_turn_id handle. The
scope_source_turns lane contains the exact prior shopper turns at those
sequences. Treat those turns as the semantic identity of the subject whose
typed authority or unanswered question is currently stored.
When input_projection is bounded_head_tail, the marker
`…[middle omitted]…` means only the middle of semantic text was omitted by a
deterministic request-local budget. Never infer the missing text.
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
- unclear: the relationship cannot be established confidently."""

_WEATHER_SCOPE_PENDING_PROMPT = """Separately choose exactly one pending_disposition:
- not_addressed: the current query neither answers nor explicitly resumes the
  supplied pending question.
- answered: the shopper answers only the supplied pending question and leaves
  the stored opposite component unchanged. Use this only with same_subject.
- declined: the shopper explicitly declines or cancels the supplied pending
  question and wants to continue without answering it. Use this only with
  same_subject. This marks only that component unavailable for the current
  subject and consumes only that exact question; it is not a permanent
  preference against weather and does not interpret other location or date
  facts.
- resume_requested: the shopper explicitly asks what information is still
  needed, so the existing pending question should be asked again without
  changing its scope or source binding. Use this only with unchanged.

For answered, declined, or resume_requested, exactly echo the pending
question's opaque source_turn_id in the top-level pending_source_turn_id field.
Omit that handle for not_addressed. If an answer also changes or withdraws the
stored opposite component, use same_subject/not_addressed so activation can
author both current facts without pending authority."""

_WEATHER_SCOPE_OUTPUT_BOUNDARY = """Do not extract, normalize, copy, or author location, date, scope revision, or component actions; the main shopper
activation supplies current-turn scope facts once and the server validates and
compiles them. Follow-up question selection remains the main shopper skill's
responsibility.

Make exactly one required resolver control call. This is a schema-only control
channel, not a business tool. Do not include prose or Markdown."""

_WEATHER_SCOPE_SUBJECT_ONLY_PROMPT = """No durable pending question is bound to this scope. The control therefore
contains only subject_relation. Do not return pending_disposition,
pending_source_turn_id, or any pending-question decision."""

WEATHER_SCOPE_RESOLVER_SYSTEM_PROMPT = "\n\n".join(
    (
        _WEATHER_SCOPE_SUBJECT_PROMPT,
        _WEATHER_SCOPE_PENDING_PROMPT,
        "Return only these semantic axes. "
        + _WEATHER_SCOPE_OUTPUT_BOUNDARY,
    )
)
WEATHER_SCOPE_SUBJECT_RESOLVER_SYSTEM_PROMPT = "\n\n".join(
    (
        _WEATHER_SCOPE_SUBJECT_PROMPT,
        _WEATHER_SCOPE_SUBJECT_ONLY_PROMPT,
        "Return only subject_relation. "
        + _WEATHER_SCOPE_OUTPUT_BOUNDARY,
    )
)

_INPUT_OMISSION_MARKER = "…[middle omitted]…"
_MIN_PROJECTED_TEXT_CHARS = 64


WeatherScopeSubjectRelation = Literal[
    "same_subject",
    "new_subject",
    "unchanged",
    "unclear",
]
WeatherScopePendingDisposition = Literal[
    "not_addressed",
    "answered",
    "declined",
    "resume_requested",
]

_VALID_RESOLVER_OUTCOMES = frozenset(
    {
        ("same_subject", "not_addressed"),
        ("same_subject", "answered"),
        ("same_subject", "declined"),
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
            "exactly for answered, declined, or resume_requested; omit it for "
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
            "declined",
            "resume_requested",
        }
        if pending_handle_required != (self.pending_source_turn_id is not None):
            raise ValueError(
                "pending_source_turn_id is required exactly when the pending "
                "question is answered, declined, or resumed"
            )
        return self


class WeatherScopeSubjectDecision(BaseModel):
    """Subject-only capability when no durable pending binding exists."""

    model_config = ConfigDict(extra="forbid", strict=True)

    subject_relation: WeatherScopeSubjectRelation


def weather_scope_resolver_control_model(
    *,
    has_pending_binding: bool,
) -> type[BaseModel]:
    """Advertise pending controls only when durable state can authorize them."""

    return (
        WeatherScopeResolverDecision
        if has_pending_binding
        else WeatherScopeSubjectDecision
    )


def weather_scope_resolver_system_prompt(
    *,
    has_pending_binding: bool,
) -> str:
    """Match semantic instructions to the resolver's advertised capability."""

    return (
        WEATHER_SCOPE_RESOLVER_SYSTEM_PROMPT
        if has_pending_binding
        else WEATHER_SCOPE_SUBJECT_RESOLVER_SYSTEM_PROMPT
    )


@dataclass(frozen=True)
class WeatherScopeResolverWork:
    """One bounded, immutable resolver request projection."""

    prompt: str
    input_projection: Literal["exact", "bounded_head_tail"]


def build_weather_scope_resolver_prompt(
    *,
    current_query: str,
    current_scope_json: Mapping[str, Any],
    current_utc_date: date,
    rolling_summary: str,
    scope_source_turns: Sequence[Mapping[str, Any]],
    recent_turns: Sequence[Mapping[str, Any]],
    max_input_chars: int,
) -> WeatherScopeResolverWork | None:
    """Build one aggregate-bounded resolver input or fail closed.

    Current query, typed scope, and trusted date remain exact. Only semantic
    text is projected, and every scope-source sequence remains represented.
    """

    exact_payload = _resolver_payload(
        current_query=current_query,
        current_scope_json=current_scope_json,
        current_utc_date=current_utc_date,
        rolling_summary=rolling_summary,
        scope_source_turns=scope_source_turns,
        recent_turns=recent_turns,
    )
    exact_prompt = _serialize_resolver_payload(exact_payload)
    if len(exact_prompt) <= max_input_chars:
        return WeatherScopeResolverWork(
            prompt=exact_prompt,
            input_projection="exact",
        )

    retained_recent_turns = [dict(turn) for turn in recent_turns]
    include_summary = bool(rolling_summary)
    minimum_prompt = _serialize_projected_resolver_payload(
        current_query=current_query,
        current_scope_json=current_scope_json,
        current_utc_date=current_utc_date,
        rolling_summary=rolling_summary,
        scope_source_turns=scope_source_turns,
        recent_turns=retained_recent_turns,
        include_summary=include_summary,
        text_limit=_MIN_PROJECTED_TEXT_CHARS,
    )
    while len(minimum_prompt) > max_input_chars and retained_recent_turns:
        retained_recent_turns.pop(0)
        minimum_prompt = _serialize_projected_resolver_payload(
            current_query=current_query,
            current_scope_json=current_scope_json,
            current_utc_date=current_utc_date,
            rolling_summary=rolling_summary,
            scope_source_turns=scope_source_turns,
            recent_turns=retained_recent_turns,
            include_summary=include_summary,
            text_limit=_MIN_PROJECTED_TEXT_CHARS,
        )
    if len(minimum_prompt) > max_input_chars and include_summary:
        include_summary = False
        minimum_prompt = _serialize_projected_resolver_payload(
            current_query=current_query,
            current_scope_json=current_scope_json,
            current_utc_date=current_utc_date,
            rolling_summary=rolling_summary,
            scope_source_turns=scope_source_turns,
            recent_turns=retained_recent_turns,
            include_summary=False,
            text_limit=_MIN_PROJECTED_TEXT_CHARS,
        )
    if len(minimum_prompt) > max_input_chars:
        return None

    text_lengths = [len(rolling_summary)] if include_summary else []
    for turn in [*scope_source_turns, *retained_recent_turns]:
        text_lengths.extend(
            len(value)
            for key in ("shopper_text", "assistant_text")
            if isinstance((value := turn.get(key)), str)
        )
    lower = _MIN_PROJECTED_TEXT_CHARS
    upper = max([lower, *text_lengths])
    best_prompt = minimum_prompt
    while lower <= upper:
        candidate_limit = (lower + upper) // 2
        candidate_prompt = _serialize_projected_resolver_payload(
            current_query=current_query,
            current_scope_json=current_scope_json,
            current_utc_date=current_utc_date,
            rolling_summary=rolling_summary,
            scope_source_turns=scope_source_turns,
            recent_turns=retained_recent_turns,
            include_summary=include_summary,
            text_limit=candidate_limit,
        )
        if len(candidate_prompt) <= max_input_chars:
            best_prompt = candidate_prompt
            lower = candidate_limit + 1
        else:
            upper = candidate_limit - 1
    return WeatherScopeResolverWork(
        prompt=best_prompt,
        input_projection="bounded_head_tail",
    )


def _resolver_payload(
    *,
    current_query: str,
    current_scope_json: Mapping[str, Any],
    current_utc_date: date,
    rolling_summary: str,
    scope_source_turns: Sequence[Mapping[str, Any]],
    recent_turns: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
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
    }


def _serialize_projected_resolver_payload(
    *,
    current_query: str,
    current_scope_json: Mapping[str, Any],
    current_utc_date: date,
    rolling_summary: str,
    scope_source_turns: Sequence[Mapping[str, Any]],
    recent_turns: Sequence[Mapping[str, Any]],
    include_summary: bool,
    text_limit: int,
) -> str:
    payload = _resolver_payload(
        current_query=current_query,
        current_scope_json=current_scope_json,
        current_utc_date=current_utc_date,
        rolling_summary=(
            _head_tail_projection(rolling_summary, text_limit)
            if include_summary
            else ""
        ),
        scope_source_turns=[
            _project_semantic_turn(turn, text_limit)
            for turn in scope_source_turns
        ],
        recent_turns=[
            _project_semantic_turn(turn, text_limit) for turn in recent_turns
        ],
    )
    payload["input_projection"] = "bounded_head_tail"
    return _serialize_resolver_payload(payload)


def _project_semantic_turn(
    turn: Mapping[str, Any],
    text_limit: int,
) -> dict[str, Any]:
    return {
        "sequence": turn.get("sequence"),
        "shopper_text": _head_tail_projection(
            _semantic_text(turn.get("shopper_text")),
            text_limit,
        ),
        "assistant_text": _head_tail_projection(
            _semantic_text(turn.get("assistant_text")),
            text_limit,
        ),
    }


def _semantic_text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _head_tail_projection(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    available = limit - len(_INPUT_OMISSION_MARKER)
    head_chars = (available + 1) // 2
    tail_chars = available // 2
    return (
        value[:head_chars]
        + _INPUT_OMISSION_MARKER
        + (value[-tail_chars:] if tail_chars else "")
    )


def _serialize_resolver_payload(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def parse_weather_scope_resolver_tool_call(
    message: Any,
    *,
    control_model: type[BaseModel] = WeatherScopeResolverDecision,
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
        name != control_model.__name__
        or not isinstance(arguments, dict)
    ):
        return None
    try:
        parsed = control_model.model_validate(
            arguments,
            strict=True,
        )
    except ValidationError:
        return None
    if isinstance(parsed, WeatherScopeResolverDecision):
        return parsed
    if isinstance(parsed, WeatherScopeSubjectDecision):
        return WeatherScopeResolverDecision(
            subject_relation=parsed.subject_relation,
            pending_disposition="not_addressed",
        )
    return None
