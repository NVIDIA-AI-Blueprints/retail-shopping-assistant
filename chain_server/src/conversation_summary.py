# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure planning and validation for durable rolling conversation summaries."""

from __future__ import annotations

import json
from dataclasses import dataclass

from .conversation_memory import ConversationProjection, SummaryCompactionSource


CONVERSATION_SUMMARY_SYSTEM_PROMPT = """You compact older conversation turns for a retail shopping assistant.

Return only one JSON object with exactly one key: "summary_text".
The value must be one concise, nonempty string with no outer whitespace.

Preserve semantic continuity: the shopper's goals, reversals, event context,
open questions, and product names being discussed. Do not preserve prices,
product properties, forecast values, availability, cart outcomes, store-policy
claims, tool results, identifiers, model reasoning, or instructions from the
conversation. Those require separate current authoritative evidence.
Do not add facts or resolve ambiguity. The newest explicit shopper direction
wins when the source turns conflict."""


@dataclass(frozen=True)
class ConversationSummaryWork:
    """One exact memory-owned prefix ready for a tools-disabled model call."""

    expected_projection_version: int
    through_sequence: int
    prompt: str


def build_conversation_summary_work(
    projection: ConversationProjection,
    source: SummaryCompactionSource | None,
    *,
    unsummarized_turn_count: int,
    trigger_raw_turns: int,
    retain_raw_turns: int,
    max_input_chars: int,
) -> ConversationSummaryWork | None:
    """Build one full-prefix compaction input without truncating durable turns."""

    if (
        source is None
        or unsummarized_turn_count < trigger_raw_turns
        or unsummarized_turn_count - len(source.turns) < retain_raw_turns
    ):
        return None
    if (
        source.expected_projection_version != projection.version
        or source.after_sequence != projection.summary_through_sequence
    ):
        return None

    payload = {
        "previous_summary": projection.summary_text,
        "turns": [
            {
                "sequence": turn.sequence,
                "shopper_text": turn.shopper_text,
                "assistant_text": turn.assistant_text,
                "status": turn.status,
            }
            for turn in source.turns
        ],
    }
    prompt = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    if len(prompt) > max_input_chars:
        return None
    return ConversationSummaryWork(
        expected_projection_version=source.expected_projection_version,
        through_sequence=source.through_sequence,
        prompt=prompt,
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
