# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Count what one turn spent, for isolating a tool loop without a full replay.

A journey run is five minutes and a whole conversation; this is one turn. Use
it to reproduce a suspected loop cheaply, then confirm with the journey.

    python scripts/what_one_turn_spent.py "it's going to snow, what should I wear" 3
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter

import requests

_URL = "http://localhost:8009/query/stream"


def spend(query: str) -> dict[str, object]:
    conversation = f"what-one-turn-spent-{int(time.time() * 1000)}"
    started = time.monotonic()
    diagnostics: dict[str, object] = {}
    with requests.post(
        _URL,
        json={
            "user_id": 90_000_126,
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
            body = json.loads(line[6:]).get("payload")
            if isinstance(body, dict) and body.get("agent_diagnostics"):
                diagnostics = body["agent_diagnostics"]
    calls = [c for c in (diagnostics.get("tool_calls") or []) if isinstance(c, dict)]
    seen: set[str] = set()
    repeats = 0
    for call in calls:
        key = json.dumps(
            [call.get("tool_name"), call.get("arguments")], sort_keys=True, default=str
        )
        if key in seen:
            repeats += 1
        seen.add(key)
    return {
        "seconds": round(time.monotonic() - started, 1),
        "tools": Counter(str(c.get("tool_name")) for c in calls),
        "rejected": Counter(
            str(c.get("rejection_reason"))
            for c in calls
            if c.get("status") == "rejected"
        ),
        "repeats": repeats,
        "ended": str(diagnostics.get("final_termination_reason") or ""),
    }


def main() -> int:
    query = sys.argv[1] if len(sys.argv) > 1 else "it's going to snow, what should I wear"
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    print(f"query: {query}\n")
    for attempt in range(1, runs + 1):
        spent = spend(query)
        tools = spent["tools"]
        assert isinstance(tools, Counter)
        print(
            f"  run {attempt}: {sum(tools.values()):>3} calls, "
            f"{spent['repeats']:>3} repeats, {spent['seconds']:>6}s, "
            f"ended={spent['ended']}"
        )
        print(f"          {dict(tools)}")
        if spent["rejected"]:
            print(f"          rejected: {dict(spent['rejected'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
