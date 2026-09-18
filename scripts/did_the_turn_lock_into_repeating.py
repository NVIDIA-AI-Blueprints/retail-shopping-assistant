"""Replay the turn that locks into repeating, and report whether it did.

J01 turn 16 sends one four-scope search and then sends it again twenty-one
times, each model step carrying empty text content, until the graph's recursion
limit kills the turn. It does not happen as a fresh single turn: the pattern has
to be in the context before the model will copy it, so the fifteen turns before
it are the experiment, not preamble.

Drives `/query/timing` rather than `/query/stream` because only that endpoint
returns `partial_graph_messages`, which is the evidence -- the journey harness
uses the stream and discards them, which is why this went unseen.

Prints the three numbers that say whether lock-in happened: how many identical
argument sets the model produced, how many of its messages carried no text, and
how the turn ended.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

import yaml

_JOURNEY = (
    Path(__file__).resolve().parents[1]
    / "tests/evaluation/datasets/val/scripts/journeys/J01_wedding_abroad.yaml"
)


def _ask(base_url: str, conversation_id: str, said: str) -> dict[str, Any]:
    body = json.dumps(
        {"user_id": 1, "conversation_id": conversation_id, "query": said}
    ).encode()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/query/timing",
        body,
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=400) as response:
        return json.load(response)


def _digest(value: Any) -> str:
    return hashlib.sha1(
        json.dumps(value, sort_keys=True, default=str).encode()
    ).hexdigest()[:10]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8009")
    parser.add_argument("--label", default="unlabelled")
    parser.add_argument(
        "--up-to",
        type=int,
        default=15,
        help="Turns of context to build before the turn under test.",
    )
    args = parser.parse_args()

    turns = yaml.safe_load(_JOURNEY.read_text())["turns"]
    conversation_id = f"lockin-{args.label}-{_digest(args.label)}"

    print(f"[{args.label}] building {args.up_to} turns of context", flush=True)
    for index, turn in enumerate(turns[: args.up_to], start=1):
        result = _ask(args.base_url, conversation_id, turn["say"])
        calls = (result.get("agent_diagnostics") or {}).get("tool_calls") or []
        searches = sum(
            1 for call in calls if call.get("tool_name") == "search_catalog_tool"
        )
        # Turn 1 loops too and still reaches an answer, so the count is worth
        # seeing on every turn rather than only the one under test.
        print(f"  {index:2d}  searches={searches:2d}  {turn['say'][:48]}", flush=True)

    said = turns[args.up_to]["say"]
    print(f"[{args.label}] turn under test: {said}", flush=True)
    result = _ask(args.base_url, conversation_id, said)

    diagnostics = result.get("agent_diagnostics") or {}
    messages = diagnostics.get("partial_graph_messages") or []
    from_the_model = [m for m in messages if m.get("type") == "ai"]
    argument_sets = {
        _digest(m["tool_calls"][0]["arguments"])
        for m in from_the_model
        if m.get("tool_calls")
    }
    result_payloads = {
        _digest(m.get("content")) for m in messages if m.get("type") == "tool"
    }
    empty = [m for m in from_the_model if not str(m.get("content") or "")]
    searches = sum(
        1
        for call in (diagnostics.get("tool_calls") or [])
        if call.get("tool_name") == "search_catalog_tool"
    )

    print()
    print(f"[{args.label}] RESULT")
    print(f"  searches                     : {searches}")
    print(f"  ended                        : {diagnostics.get('final_termination_reason')}")
    print(f"  seconds                      : {(result.get('timings') or {}).get('total', 0):.1f}")
    print(f"  model messages               : {len(from_the_model)}")
    print(f"  with no text at all          : {len(empty)}")
    print(f"  distinct argument sets       : {len(argument_sets)}")
    print(f"  distinct tool-result payloads: {len(result_payloads)}")

    # Lock-in is the conjunction, not any one of these. A turn may legitimately
    # search several times, and a model may legitimately answer with no text
    # while calling a tool once. Repeating one argument set across many
    # text-free steps is the signature.
    locked_in = (
        searches > 3 and len(argument_sets) == 1 and len(empty) == len(from_the_model)
    )
    print(f"  LOCKED IN                    : {'YES' if locked_in else 'no'}")
    print()
    print(f"  reply: {str(result.get('response'))[:160]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
