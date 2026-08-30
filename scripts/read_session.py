#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read a multi-turn conversation out of Phoenix, one line per turn.

A session is a conversation: ``session.id`` is the ``conversation_id`` the
client sent, and one turn is one trace. The Phoenix UI shows this under
Sessions; this prints it, which is what you want when comparing two runs or
pasting a turn into a bug report.

    python3 scripts/read_session.py                    # list the sessions
    python3 scripts/read_session.py demo20-run11       # read one
    python3 scripts/read_session.py demo20-run11 --replies

Reads the ``turn`` span for what the agent did and the ``LangGraph`` span for
what was said. NeMo Relay's spans are deliberately not read here: only its
per-turn scope carries a conversation, and it carries it under a different key,
so a session is told entirely by the OpenTelemetry instrumentation.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request


def _spans(base: str, limit: int) -> list[dict]:
    # Phoenix rejects anything above 1000 outright rather than clamping.
    url = f"{base.rstrip('/')}/v1/projects/default/spans?limit={min(limit, 1000)}"
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)["data"]


def _said(graph: dict | None) -> str:
    """The shopper's words, dug out of the message the runtime actually sent."""

    if not graph:
        return "?"
    try:
        content = json.loads(graph["attributes"]["input.value"])
        message = content["messages"][-1]["data"]["content"]
    except (KeyError, IndexError, ValueError, TypeError):
        return "?"
    # The runtime frames each turn with a header; the query is the part a person
    # would recognise.
    found = re.search(r"USER QUERY:\s*(.+)", message)
    return found.group(1).strip() if found else message.splitlines()[-1][:80]


def _replied(graph: dict | None) -> str:
    if not graph:
        return ""
    try:
        content = json.loads(graph["attributes"]["output.value"])
        return content["messages"][-1]["data"]["content"].strip()
    except (KeyError, IndexError, ValueError, TypeError):
        return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", nargs="?", help="the conversation_id to read")
    parser.add_argument("--phoenix", default="http://localhost:6006")
    parser.add_argument("--limit", type=int, default=1000, help="spans to fetch")
    parser.add_argument("--replies", action="store_true", help="include replies")
    args = parser.parse_args()

    rows = _spans(args.phoenix, args.limit)

    if not args.session:
        seen: dict[str, set] = {}
        for row in rows:
            name = row["attributes"].get("session.id")
            if name:
                seen.setdefault(name, set()).add(row["context"]["trace_id"])
        if not seen:
            print("No sessions. Is OTEL_EXPORTER_OTLP_ENDPOINT set on chain-server?")
            return 1
        for name, traces in sorted(seen.items(), key=lambda kv: -len(kv[1])):
            print(f"{len(traces):4} turns  {name}")
        return 0

    mine = [r for r in rows if r["attributes"].get("session.id") == args.session]
    if not mine:
        print(f"No spans for session {args.session!r}.")
        return 1

    turns: dict[str, dict] = {}
    for row in mine:
        turns.setdefault(row["context"]["trace_id"], {})[row["name"]] = row
    ordered = sorted(
        turns.values(), key=lambda spans: min(s["start_time"] for s in spans.values())
    )

    print(f"{args.session} — {len(ordered)} turns")
    for index, spans in enumerate(ordered, 1):
        turn = spans.get("turn")
        attributes = turn["attributes"] if turn else {}
        skills = attributes.get("metadata.skills") or ["-"]
        tools = attributes.get("metadata.tools") or []
        print(f'\n{index}. "{_said(spans.get("LangGraph"))}"')
        print(f"   skill  {skills[0]}")
        print(f"   tools  {', '.join(tools) or '-'}")
        print(
            f"   ->     {attributes.get('metadata.products_shown')} shown, "
            f"{attributes.get('metadata.tool_calls_rejected')} rejected, "
            f"{attributes.get('metadata.termination_reason')}"
        )
        if args.replies:
            reply = _replied(spans.get("LangGraph"))
            if reply:
                print("   " + reply.replace("\n", "\n   ")[:600])
    return 0


if __name__ == "__main__":
    sys.exit(main())
