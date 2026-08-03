# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shape helpers for LangChain graph messages.

Pure readers over message objects: which messages belong to the current turn,
how to reach a tool result by call id, and how to coerce message content to
text. They hold no runtime state and depend on nothing else in the runtime,
which is what makes them separable ahead of the larger extraction.
"""

from __future__ import annotations

from typing import Any

from .skill_activation import (
    SKILL_ACTIVATION_COMPLETE,
    SKILL_ACTIVATION_REQUIRED,
    SKILL_TOOL_NOT_GRANTED,
)


def _current_turn_messages(messages: list[Any], request_id: str) -> list[Any]:
    marker = f"REQUEST ID: {request_id}"
    start: int | None = None
    for index, message in enumerate(messages):
        if _message_type(message) != "human":
            continue
        if marker in _content_to_text(_value(message, "content")):
            start = index + 1
    return [] if start is None else messages[start:]

def _prior_turn_messages(messages: list[Any], request_id: str) -> list[Any]:
    """Return messages before the current server-owned request marker."""

    marker = f"REQUEST ID: {request_id}"
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if _message_type(message) != "human":
            continue
        if marker in _content_to_text(_value(message, "content")):
            return messages[:index]
    return []

def _tool_results_by_call_id(messages: list[Any]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for message in messages:
        if _message_type(message) != "tool":
            continue
        tool_call_id = str(_value(message, "tool_call_id") or "")
        if tool_call_id:
            results[tool_call_id] = message
    return results

def _message_type(message: Any) -> str:
    message_type = str(_value(message, "type") or _value(message, "role") or "")
    return {"assistant": "ai", "user": "human"}.get(message_type, message_type)

def _extract_final_text(result: Any) -> str:
    if isinstance(result, dict):
        messages = result.get("messages")
        if isinstance(messages, list):
            for message in reversed(messages):
                if _message_type(message) in {"human", "system", "tool"}:
                    continue
                if _value(message, "tool_calls"):
                    continue
                content = getattr(message, "content", None)
                if content is None and isinstance(message, dict):
                    content = message.get("content")
                text = _content_to_text(content)
                if text and not text.startswith(
                    (
                        SKILL_ACTIVATION_COMPLETE,
                        SKILL_ACTIVATION_REQUIRED,
                        SKILL_TOOL_NOT_GRANTED,
                        "SHOPPER_SKILL_ACTIVATION_FAILED:",
                    )
                ):
                    return text
        return _content_to_text(result.get("response")) or ""
    return _content_to_text(getattr(result, "content", result)) or ""

def _result_messages(result: Any) -> list[Any]:
    if isinstance(result, dict):
        messages = result.get("messages")
        if isinstance(messages, list):
            return messages
        return [result]

    messages = getattr(result, "messages", None)
    if isinstance(messages, list):
        return messages
    return [result]

def _value(source: Any, key: str) -> Any:
    if isinstance(source, dict):
        return source.get(key)
    return getattr(source, key, None)

def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts).strip()
    return ""
