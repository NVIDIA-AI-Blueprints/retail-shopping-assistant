# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What "the first shoes" actually resolves to, and what was asked of the record.

The journey artifacts record which tools ran, not what was passed to them, so
whether an ordinal reference lands on the right garment cannot be read out of a
run. This drives one grouped showing and then one ordinal reference against the
live stack, and prints three things side by side: the products published, the
descriptor the model sent, and the product that came back.

    python scripts/what_an_ordinal_resolved_to.py "add the first shoes to my cart"
"""

from __future__ import annotations

import json
import sys
import time

import requests

_STREAM = "http://localhost:8009/query/stream"
_TIMING = "http://localhost:8009/query/timing"
_OPENING = "I need a dress, shoes and a bag for a wedding"


def _ask(conversation: str, query: str, *, timing: bool) -> dict:
    url = _TIMING if timing else _STREAM
    payload = {
        "user_id": 90_000_211,
        "session_id": conversation,
        "conversation_id": conversation,
        "query": query,
        "guardrails": False,
    }
    if timing:
        response = requests.post(url, json=payload, timeout=600)
        response.raise_for_status()
        return response.json()

    products: list[dict] = []
    reply = []
    with requests.post(url, json=payload, timeout=600, stream=True) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            try:
                event = json.loads(line[6:])
            except ValueError:
                continue
            if event.get("type") == "products":
                products = event.get("payload") or []
            elif event.get("type") == "content":
                reply.append(str(event.get("payload") or ""))
    return {"products": products, "response": "".join(reply)}


def _resolution_calls(state: dict) -> list[dict]:
    """Every resolve call in the turn, with its arguments and its result."""

    calls = []
    pending: dict[str, dict] = {}
    for message in state.get("partial_graph_messages") or []:
        for call in message.get("tool_calls") or []:
            if "resolve_conversation_products" in str(call.get("name") or ""):
                pending[str(call.get("id") or "")] = call.get("args") or {}
        call_id = str(message.get("tool_call_id") or "")
        if call_id and call_id in pending:
            calls.append(
                {
                    "sent": pending.pop(call_id),
                    "came_back": str(message.get("content") or "")[:1200],
                }
            )
    calls.extend({"sent": args, "came_back": "(no result)"} for args in pending.values())
    return calls


def main() -> int:
    reference = sys.argv[1] if len(sys.argv) > 1 else "add the first shoes to my cart"
    conversation = f"ordinal-probe-{int(time.time() * 1000)}"

    shown = _ask(conversation, _OPENING, timing=False)
    print(f"TURN 1  {_OPENING!r}")
    for product in shown["products"]:
        print(
            f"  {product.get('position'):>3}  {product.get('display_name')}"
            f"   [{product.get('category')}]"
        )

    print(f"\nTURN 2  {reference!r}")
    state = _ask(conversation, reference, timing=True)
    inner = state.get("state") or state
    calls = _resolution_calls(inner)
    if not calls:
        # Distinguish "the model never asked the record" from a probe that
        # cannot read the answer.
        messages = inner.get("partial_graph_messages")
        print(
            f"  no resolve call. messages captured: "
            f"{len(messages) if isinstance(messages, list) else messages!r}"
        )
        print(f"  tools this turn: {inner.get('tool_names') or inner.get('tools')}")
    for call in calls:
        print(f"  sent:      {json.dumps(call['sent'])}")
        print(f"  came back: {call['came_back']}\n")
    print("  reply:", str(inner.get("response") or "")[:400])
    print("  cart:", json.dumps(inner.get("cart") or inner.get("cart_state") or {})[:300])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
