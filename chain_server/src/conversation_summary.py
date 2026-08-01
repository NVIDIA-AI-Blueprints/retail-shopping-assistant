# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure planning and validation for durable rolling conversation summaries."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from .conversation_memory import (
    ConversationProjection,
    SummaryCompactionSource,
    SummaryCompactionTurn,
)


BOUNDED_HEAD_TAIL_PROJECTION = "bounded_head_tail"
_OMITTED_MIDDLE_MARKER = "…[middle omitted]…"


CONVERSATION_SUMMARY_SYSTEM_PROMPT = """You compact older conversation turns for a retail shopping assistant.

Return only one JSON object with exactly one key: "summary_text".
The value must be one concise, nonempty string with no outer whitespace.

Preserve semantic continuity: the shopper's goals, reversals, event context,
open questions, and product names being discussed. Do not preserve prices,
product properties, forecast values, availability, cart outcomes, store-policy
claims, tool results, identifiers, model reasoning, or instructions from the
conversation. Those require separate current authoritative evidence.
Do not add facts or resolve ambiguity. The newest explicit shopper direction
wins when the source turns conflict.

When input_projection is "bounded_head_tail", the oldest turn contains marked
head-and-tail excerpts because that one durable turn exceeded the input budget.
Treat the omission marker as metadata and do not infer the omitted text."""


@dataclass(frozen=True)
class ConversationSummaryWork:
    """One memory-owned prefix ready for a tools-disabled model call."""

    expected_projection_version: int
    through_sequence: int
    prompt: str
    input_projection: Literal["exact", "bounded_head_tail"] = "exact"


def _summary_prompt(
    previous_summary: str,
    turns: list[dict[str, object]],
    *,
    input_projection: Literal["exact", "bounded_head_tail"] = "exact",
) -> str:
    payload: dict[str, object] = {
        "previous_summary": previous_summary,
        "turns": turns,
    }
    if input_projection != "exact":
        payload["input_projection"] = input_projection
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _turn_payload(turn: SummaryCompactionTurn) -> dict[str, object]:
    return {
        "sequence": turn.sequence,
        "shopper_text": turn.shopper_text,
        "assistant_text": turn.assistant_text,
        "status": turn.status,
    }


def _head_tail_excerpt(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    remaining = limit - len(_OMITTED_MIDDLE_MARKER)
    head_length = (remaining + 1) // 2
    tail_length = remaining // 2
    return (
        value[:head_length]
        + _OMITTED_MIDDLE_MARKER
        + value[len(value) - tail_length :]
    )


def _bounded_oldest_turn_work(
    projection: ConversationProjection,
    source: SummaryCompactionSource,
    *,
    max_input_chars: int,
) -> ConversationSummaryWork | None:
    """Project one oversized oldest turn without changing durable source text."""

    oldest = source.turns[0]
    minimum_limit = len(_OMITTED_MIDDLE_MARKER) + 2

    def prompt_for(limit: int) -> str:
        turn = _turn_payload(oldest)
        turn["shopper_text"] = _head_tail_excerpt(oldest.shopper_text, limit)
        turn["assistant_text"] = _head_tail_excerpt(oldest.assistant_text, limit)
        return _summary_prompt(
            projection.summary_text,
            [turn],
            input_projection=BOUNDED_HEAD_TAIL_PROJECTION,
        )

    minimum_prompt = prompt_for(minimum_limit)
    if len(minimum_prompt) > max_input_chars:
        return None

    best_prompt = minimum_prompt
    low = minimum_limit + 1
    high = max(
        minimum_limit,
        len(oldest.shopper_text),
        len(oldest.assistant_text),
    )
    while low <= high:
        candidate_limit = (low + high) // 2
        candidate_prompt = prompt_for(candidate_limit)
        if len(candidate_prompt) <= max_input_chars:
            best_prompt = candidate_prompt
            low = candidate_limit + 1
        else:
            high = candidate_limit - 1

    return ConversationSummaryWork(
        expected_projection_version=source.expected_projection_version,
        through_sequence=oldest.sequence,
        prompt=best_prompt,
        input_projection=BOUNDED_HEAD_TAIL_PROJECTION,
    )


def build_conversation_summary_work(
    projection: ConversationProjection,
    source: SummaryCompactionSource | None,
    *,
    unsummarized_turn_count: int,
    trigger_raw_turns: int,
    retain_raw_turns: int,
    max_input_chars: int,
) -> ConversationSummaryWork | None:
    """Build the largest fitting oldest prefix while retaining the newest tail."""

    if source is None or unsummarized_turn_count < trigger_raw_turns:
        return None
    if (
        source.expected_projection_version != projection.version
        or source.after_sequence != projection.summary_through_sequence
    ):
        return None

    foldable_count = min(
        len(source.turns),
        unsummarized_turn_count - retain_raw_turns,
    )
    if foldable_count <= 0:
        return None

    for prefix_length in range(foldable_count, 0, -1):
        turns = source.turns[:prefix_length]
        prompt = _summary_prompt(
            projection.summary_text,
            [_turn_payload(turn) for turn in turns],
        )
        if len(prompt) <= max_input_chars:
            return ConversationSummaryWork(
                expected_projection_version=source.expected_projection_version,
                through_sequence=turns[-1].sequence,
                prompt=prompt,
            )

    return _bounded_oldest_turn_work(
        projection,
        source,
        max_input_chars=max_input_chars,
    )


def parse_conversation_summary_output(
    content: str,
    *,
    max_output_chars: int,
) -> str | None:
    """Accept only the closed one-key compactor response."""

    if not isinstance(content, str) or not content:
        return None
    try:
        payload = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or set(payload) != {"summary_text"}:
        return None
    summary_text = payload["summary_text"]
    if (
        not isinstance(summary_text, str)
        or not summary_text
        or summary_text != summary_text.strip()
        or len(summary_text) > max_output_chars
    ):
        return None
    return summary_text
