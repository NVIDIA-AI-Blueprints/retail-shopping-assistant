# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Print every tool call of a turn with the arguments it was given.

A journey artifact keeps tool *names* and, for searches, the scopes. It does
not keep the arguments of anything else, which is why "add the Jade Tone Canvas
Tote Bag to my cart" failing with the cart left empty could be seen but not
explained: the reply said the product reference was not valid, and nothing
recorded which reference was passed.

The stream already carries it. `agent_diagnostics.tool_calls` holds each call's
name, arguments and rejection reason, so this drives a few turns and prints
them.

    python scripts/what_the_cart_turn_asked_for.py
"""

from __future__ import annotations

import json
import sys
import time

import requests

_URL = "http://localhost:8009/query/stream"
#: The tail of J02, from the turn that fails back through enough of the
#: conversation to carry its context.
_TURNS = (
    "put together a work outfit for me, nothing over $150",
    "show me some tote bags",
    "add the Jade Tone Canvas Tote Bag to my cart",
    "add the Ombre Canvas Tote Bag as well",
)


def _turn(conversation: str, query: str) -> tuple[str, dict, list]:
    reply, diagnostics, products = "", {}, []
    with requests.post(
        _URL,
        json={
            "user_id": 90_000_412,
            "session_id": conversation,
            "conversation_id": conversation,
            "query": query,
            "guardrails": False,
        },
        timeout=300,
        stream=True,
    ) as response:
        response.raise_for_status()
        for raw in response.iter_lines():
            line = raw.decode() if isinstance(raw, bytes) else raw
            if not line.startswith("data: ") or "[DONE]" in line:
                continue
            event = json.loads(line[6:])
            body = event.get("payload")
            if event.get("type") == "content" and isinstance(body, str):
                reply += body
            elif event.get("type") == "products" and isinstance(body, list):
                products = body
            if isinstance(body, dict) and body.get("agent_diagnostics"):
                diagnostics = body["agent_diagnostics"]
    return reply, diagnostics, products


def _cart(user_id: int) -> list:
    try:
        response = requests.get(f"http://localhost:8009/cart/{user_id}", timeout=30)
        response.raise_for_status()
        return response.json().get("contents") or []
    except requests.RequestException:
        return []


def main() -> int:
    conversation = f"what-the-cart-turn-asked-for-{int(time.time() * 1000)}"
    for position, query in enumerate(_TURNS, start=1):
        reply, diagnostics, products = _turn(conversation, query)
        print(f"\n{'=' * 78}\nTURN {position}: {query}\n{'=' * 78}")
        for call in diagnostics.get("tool_calls") or []:
            name = call.get("tool_name")
            if name == "activate_shopper_skills_tool":
                continue
            arguments = json.dumps(call.get("arguments"), default=str)
            print(f"\n-- {name}")
            print(f"   args:   {arguments[:700]}")
            for key in ("status", "rejection_reason", "rejected", "result_preview"):
                if call.get(key):
                    print(f"   {key}: {str(call[key])[:400]}")
        print(f"\n   shown:  {[p.get('display_name') for p in products]}")
        print(f"   cart:   {[c.get('name') or c.get('display_name') for c in _cart(90_000_412)]}")
        print(f"   reply:  {reply[:300]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
