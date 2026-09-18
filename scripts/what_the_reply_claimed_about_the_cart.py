# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Put a turn's reply beside the cart the cart service actually holds.

A turn once said "I've added the Ombre Canvas Tote Bag, your cart now has one
item" having called no cart tool at all, over a cart that was empty. The
journey harness catches that because it reads the cart service rather than the
reply, but it catches it only where a scenario happens to assert the cart.

This shows the three facts that disagree in such a turn -- the cart tools that
ran, the cart the service returns afterwards, and what the reply told the
shopper -- so a claim with nothing behind it is visible without a scenario
having been written for it.

    python scripts/what_the_reply_claimed_about_the_cart.py
"""

from __future__ import annotations

import sys
import time

import requests

_ASSISTANT = "http://localhost:8009/query/timing"
_MEMORY = "http://localhost:8011"

#: Fresh per run. The cart is keyed on the user, and a reused id starts the run
#: holding what the last run put there, which is how a first sighting of this
#: read four of a bag into a cart that should have held one.
_USER = 90_000_000 + int(time.time()) % 1_000_000

#: The shape that produced the fabrication: a showing, an add, and then two
#: more asks. The later turns are the interesting ones -- the first add having
#: happened, the model has a pattern in front of it to copy.
_TURNS = (
    "show me some tote bags",
    "add the first one to my cart",
    "add the second one too",
    "what's in my cart?",
)


def _ask(conversation: str, query: str) -> dict:
    response = requests.post(
        _ASSISTANT,
        json={
            "user_id": _USER,
            "session_id": conversation,
            "conversation_id": conversation,
            "query": query,
            "guardrails": False,
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


def _cart() -> list[str]:
    response = requests.get(f"{_MEMORY}/user/{_USER}/cart", timeout=20)
    response.raise_for_status()
    body = response.json()
    lines = body.get("cart", body.get("contents")) if isinstance(body, dict) else body
    return [
        f"{line.get('item')} x{line.get('amount')}"
        for line in (lines or [])
        if isinstance(line, dict)
    ]


def _tools_that_ran(messages: list) -> list[str]:
    names: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        for call in message.get("tool_calls") or []:
            names.append(str(call.get("name")))
    return names


def main() -> int:
    conversation = f"what-the-reply-claimed-{int(time.time() * 1000)}"
    for position, query in enumerate(_TURNS, start=1):
        payload = _ask(conversation, query)
        print(f"\n{'=' * 78}\nTURN {position}: {query}\n{'=' * 78}")
        ran = _tools_that_ran(payload.get("partial_graph_messages") or [])
        print(f"tools that ran    : {ran or '(none seen in this payload)'}")
        print(f"cart service holds: {_cart() or '(empty)'}")
        print(f"reply             : {str(payload.get('response'))[:600]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
