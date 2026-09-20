# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One journey, turn by turn, with the agent's working shown beside the reply.

The journey artifacts record which tools ran but not what was passed to them,
and never what the conversation record held afterwards. So a turn that fetched
the same four products a second time, or resolved an ordinal against a record
that had already dropped the showing, reads in a transcript as an ordinary
turn with an ordinary reply.

This drives a journey's turns against the live stack and prints four things
side by side for each: what the model called and with which arguments, what
came back, what the record held once the turn finalized, and the reply. It
asserts nothing -- `tests/evaluation` is where expectations live. It is for
the case where you need to see why a passing turn is doing the wrong thing.

    python scripts/the_flow_of_one_journey.py J22_counting_what_was_shown
    python scripts/the_flow_of_one_journey.py <path-to.yaml> --out trace.md
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests
import yaml

_TIMING = "http://localhost:8009/query/timing"
_JOURNEYS = (
    Path(__file__).resolve().parents[1]
    / "tests/evaluation/datasets/val/scripts/journeys"
)
_USER_ID = 90_000_333


def _journey_path(name: str) -> Path:
    candidate = Path(name)
    if candidate.is_file():
        return candidate
    for suffix in (name, f"{name}.yaml"):
        found = _JOURNEYS / suffix
        if found.is_file():
            return found
    raise SystemExit(f"no journey named {name!r} under {_JOURNEYS}")


def _ask(conversation: str, query: str) -> dict[str, Any]:
    payload = {
        "user_id": _USER_ID,
        "session_id": conversation,
        "conversation_id": conversation,
        "query": query,
        "guardrails": False,
    }
    response = requests.post(_TIMING, json=payload, timeout=600)
    response.raise_for_status()
    return response.json()


def _what_the_record_holds(conversation: str) -> list[dict[str, Any]]:
    """The showings this conversation can still resolve a reference against.

    Read from the database rather than the service: the memory API exposes the
    projection only through `turn/start`, and starting a turn to look at the
    record would write to the thing being measured.
    """

    program = (
        "import json,sqlite3;"
        "c=sqlite3.connect('/data/context.db');"
        "r=c.execute('select product_reference_index_json from "
        "conversation_projection where conversation_id=?',"
        f"('{conversation}',)).fetchone();"
        "print(r[0] if r else '[]')"
    )
    try:
        out = subprocess.run(
            ["docker", "compose", "exec", "-T", "memory-retriever", "python", "-c", program],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        return json.loads(out.stdout.strip() or "[]")
    except Exception as error:  # noqa: BLE001 - a probe must not die on its own probe
        print(f"    (could not read the record: {error})", file=sys.stderr)
        return []


def _arguments_worth_showing(name: str, arguments: Any) -> str:
    """One line per call. Search scopes are unrolled; everything else is JSON."""

    if not isinstance(arguments, dict):
        return json.dumps(arguments, default=str)[:400]
    if name == "search_catalog_tool":
        parts = []
        for scope in arguments.get("scopes") or []:
            taxonomy = scope.get("taxonomy") or {}
            constraints = scope.get("required_constraints") or {}
            parts.append(
                f"{scope.get('requested_product_type') or '?'}"
                f" q={scope.get('semantic_query') or ''!r}"
                f" tax={taxonomy.get('category')}:{taxonomy.get('subcategory')}"
                + (f" filters={json.dumps(constraints)}" if constraints else "")
            )
        return "\n           ".join(parts) or "(no scopes)"
    return json.dumps(arguments, default=str)[:400]


def _render_turn(index: int, said: str, answer: dict[str, Any], record: list) -> str:
    diagnostics = answer.get("agent_diagnostics") or {}
    usage = answer.get("token_usage") or {}
    lines = [f"## Turn {index} — {said!r}", "", "**Agent**", ""]

    calls = diagnostics.get("tool_calls") or []
    if not calls:
        lines.append("    (no tool calls)")
    for call in calls:
        name = str(call.get("tool_name") or "?")
        lines.append(
            f"    {call.get('sequence'):>2}. {name}"
            f"  [{call.get('status')}]\n"
            f"           {_arguments_worth_showing(name, call.get('arguments'))}"
        )

    lines += ["", "**Came back**", ""]
    evidence = diagnostics.get("product_evidence") or []
    if evidence:
        for item in evidence:
            facts = item.get("facts") or {}
            lines.append(
                f"    {item.get('product_name')}"
                f"  [{facts.get('subcategory') or facts.get('category')}]"
                f"  {facts.get('price') or ''}"
            )
    else:
        lines.append("    (no products)")
    for rejection in diagnostics.get("rejected_tool_calls") or []:
        lines.append(f"    REJECTED: {rejection}")
    for duplicate in diagnostics.get("duplicate_tool_calls") or []:
        lines.append(f"    DUPLICATE: {duplicate}")

    lines += ["", "**The record now holds**", ""]
    if not record:
        lines.append("    (nothing)")
    for showing in record:
        # `turn_seq` in the projection, not `turn_sequence` as the descriptor
        # and the match models both spell it.
        turn = showing.get("turn_seq")
        products = showing.get("products") or []
        named = ", ".join(
            f"[{p.get('position')}]"
            + (f" {p['group']}:" if p.get("group") else " ")
            + str(p.get("name") or p.get("display_name") or "?")
            for p in products
        )
        lines.append(f"    turn {turn}: {named}")

    reply = str(answer.get("response") or "").strip()
    lines += [
        "",
        "**Reply**",
        "",
        "\n".join(f"> {line}" for line in reply.splitlines()) or "> (empty)",
        "",
        f"`{usage.get('input_tokens', 0):,} in · "
        f"{usage.get('output_tokens', 0):,} out · "
        f"{usage.get('model_calls', 0)} model calls · "
        f"{(answer.get('timings') or {}).get('total', 0.0):.1f}s`",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("journey", help="journey id, filename, or path")
    parser.add_argument("--out", default=None, help="write markdown here")
    arguments = parser.parse_args()

    path = _journey_path(arguments.journey)
    script = yaml.safe_load(path.read_text())
    turns = [
        str(turn.get("say"))
        for turn in (script.get("turns") or [])
        if isinstance(turn, dict) and turn.get("say")
    ]
    if not turns:
        raise SystemExit(f"{path.name} has no turns with `say`")

    build = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    conversation = f"flow-{path.stem}-{int(time.time())}"

    out = [
        f"# Flow trace: {script.get('id') or path.stem}",
        "",
        f"Build: `{build}`",
        f"Conversation: `{conversation}`",
        f"Turns: {len(turns)}",
        "",
        "---",
        "",
    ]
    print(out[0])
    for index, said in enumerate(turns, start=1):
        print(f"  turn {index}/{len(turns)}: {said[:60]!r} ...", flush=True)
        try:
            answer = _ask(conversation, said)
        except requests.RequestException as error:
            out.append(f"## Turn {index} — {said!r}\n\n    REQUEST FAILED: {error}\n")
            continue
        record = _what_the_record_holds(conversation)
        out.append(_render_turn(index, said, answer, record))

    text = "\n".join(out)
    if arguments.out:
        destination = Path(arguments.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text)
        print(f"\nwrote {destination}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
