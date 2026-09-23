#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Say some turns, then print the arguments every tool call was given.

The evaluation harness keeps the fields it reports on, and the descriptor a
turn handed the resolver is not one of them: a transcript shows that
`resolve_conversation_products_tool` ran, not what it was asked. That gap hid a
real defect. A turn resolved "the black one in a size 8" to a one-size purse,
and the transcript could not say whether the size had reached the resolver at
all or whether the model had already settled on the purse before calling it.
The arguments answered it in one line -- the model sent `display_name`,
`category`, `turn_sequence` and `ordinal`, all naming the purse, with the size
tacked on as an attribute. It had decided, and was asking for confirmation.

Needs `EXPOSE_AGENT_DIAGNOSTICS=true` on the chain server; without it the
metrics event carries no `agent_diagnostics` and this prints nothing.

    python scripts/what_the_tools_were_asked.py "show me black heels" \\
        "add the black one in a size 8"
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import requests


def say(base: str, identity: dict, text: str, timeout: float) -> dict:
    """One turn, returning the diagnostics the metrics event carried."""

    diagnostics: dict = {}
    with requests.post(
        f"{base}/query/stream",
        json={"query": text, **identity},
        timeout=timeout,
        stream=True,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if not body or body == "[DONE]":
                continue
            try:
                event = json.loads(body)
            except json.JSONDecodeError:
                continue
            payload = event.get("payload")
            if isinstance(payload, dict) and payload.get("agent_diagnostics"):
                diagnostics = payload["agent_diagnostics"]
    return diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("turns", nargs="+", help="what the shopper says, in order")
    parser.add_argument("--base", default="http://localhost:8009")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--all-turns",
        action="store_true",
        help="print every turn's tool calls, not just the last one's",
    )
    parser.add_argument(
        "--lanes",
        action="store_true",
        help="also print the memory the turn was read against",
    )
    args = parser.parse_args()

    # A fresh id per run, because the cart keys on the user when no cart id is
    # sent: reusing one leaves the previous run's cart in place, and a turn
    # that reads "already in your cart" is answering a question you did not
    # ask. Cost a confusing transcript once.
    identity = {
        "user_id": 900_000_000 + int(time.time()) % 10_000_000,
        "session_id": f"probe-{int(time.time())}",
        "conversation_id": f"probe-{int(time.time())}",
    }

    for index, text in enumerate(args.turns, start=1):
        diagnostics = say(args.base, identity, text, args.timeout)
        last = index == len(args.turns)
        if not (last or args.all_turns):
            print(f"turn {index}: {text!r}")
            continue
        print(f"\n=== turn {index}: {text!r} ===")
        if not diagnostics:
            print("no diagnostics; is EXPOSE_AGENT_DIAGNOSTICS set?")
            continue
        for call in diagnostics.get("tool_calls") or []:
            print(f"\n{call.get('sequence')}. {call.get('tool_name')}")
            print(json.dumps(call.get("arguments"), indent=2, default=str))
        if args.lanes:
            for lane, body in (diagnostics.get("context_lanes") or {}).items():
                print(f"\n--- {lane} ({len(body)} chars) ---\n{body}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
