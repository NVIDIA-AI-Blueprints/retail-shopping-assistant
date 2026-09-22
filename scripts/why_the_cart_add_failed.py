# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Show a cart tool what it was handed, and what it said back.

J02 turn 11 -- "add the Jade Tone Canvas Tote Bag to my cart" -- replied that
the product reference from earlier was not valid for a new action, and the cart
stayed empty. The journey transcript names the tools that ran and the replay
artifact keeps the reply, but neither keeps a tool's arguments, so neither says
which reference was passed or which check refused it.

The timing endpoint returns `partial_graph_messages`, which does. Two turns:
show the tote bags, then ask to add one, and print every tool call with its
arguments and its result.

    python scripts/why_the_cart_add_failed.py
"""

from __future__ import annotations

import json
import sys
import time

import requests

_URL = "http://localhost:8009/query/timing"
_TURNS = (
    "show me some tote bags",
    "add the Jade Tone Canvas Tote Bag to my cart",
)


def _ask(conversation: str, query: str) -> dict:
    response = requests.post(
        _URL,
        json={
            "user_id": 90_000_411,
            "session_id": conversation,
            "conversation_id": conversation,
            "query": query,
            "guardrails": False,
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


def _tool_traffic(messages: list) -> list[tuple[str, str, str]]:
    """Each tool call paired with the result that came back for it."""

    arguments: dict[str, tuple[str, str]] = {}
    traffic: list[tuple[str, str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        for call in message.get("tool_calls") or []:
            arguments[str(call.get("id"))] = (
                str(call.get("name")),
                json.dumps(call.get("args"), default=str),
            )
        call_id = message.get("tool_call_id")
        if call_id is None:
            continue
        name, args = arguments.get(str(call_id), ("?", "?"))
        content = message.get("content")
        traffic.append((name, args, content if isinstance(content, str) else str(content)))
    return traffic


def main() -> int:
    conversation = f"why-the-cart-add-failed-{int(time.time() * 1000)}"
    for position, query in enumerate(_TURNS, start=1):
        payload = _ask(conversation, query)
        print(f"\n{'=' * 78}\nTURN {position}: {query}\n{'=' * 78}")
        for name, args, result in _tool_traffic(payload.get("partial_graph_messages") or []):
            print(f"\n-- {name}")
            print(f"   args: {args[:600]}")
            print(f"   said: {result[:900]}")
        print(f"\nreply: {str(payload.get('response'))[:400]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
