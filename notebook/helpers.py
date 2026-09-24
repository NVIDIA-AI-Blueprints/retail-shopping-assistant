# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plumbing shared by the notebooks, so their cells show intent.

Standard library only. Every address can be overridden with an environment
variable of the same name, for a deployment that is not on localhost.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from time import monotonic, sleep

REPO = Path(__file__).resolve().parents[1]

CHAIN_SERVER = os.environ.get("CHAIN_SERVER", "http://localhost:8009")
CATALOG = os.environ.get("CATALOG", "http://localhost:8010")
MEMORY = os.environ.get("MEMORY", "http://localhost:8011")
GUARDRAILS = os.environ.get("GUARDRAILS", "http://localhost:8012")
PHOENIX = os.environ.get("PHOENIX", "http://localhost:6006")
WEB_UI = os.environ.get("WEB_UI", "http://localhost:3000")

#: A fixed shopper for the notebooks, clear of the ids replays use.
NOTEBOOK_USER_ID = 770000111


def get(url: str, timeout: float = 30):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def post(url: str, body: dict, timeout: float = 60):
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def status(url: str, timeout: float = 5) -> str:
    """The HTTP status of a GET, or why there was none."""

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return str(response.status)
    except urllib.error.HTTPError as exc:
        return str(exc.code)
    except Exception as exc:  # noqa: BLE001 - reported, not handled
        return f"unreachable ({type(exc).__name__})"


def wait_until_ready(timeout_s: float = 900, every_s: float = 10) -> bool:
    """Poll each service's readiness until all answer 200, or time runs out.

    The catalog answers `/health` while it has no index, so `/ready` is the
    check that means "can serve a search".
    """

    checks = {
        "chain-server": f"{CHAIN_SERVER}/ready",
        "catalog-retriever": f"{CATALOG}/ready",
        "memory-retriever": f"{MEMORY}/ready",
    }
    started = monotonic()
    while True:
        results = {name: status(url) for name, url in checks.items()}
        if all(code == "200" for code in results.values()):
            print("All services ready:", ", ".join(results))
            return True
        if monotonic() - started > timeout_s:
            print("Not ready after", int(timeout_s), "s:", results)
            return False
        print("waiting:", {k: v for k, v in results.items() if v != "200"})
        sleep(every_s)


def new_conversation(prefix: str = "notebook") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def say(text: str, conversation_id: str, user_id: int = NOTEBOOK_USER_ID) -> dict:
    """Send one shopper turn and collect what streams back.

    The stream carries the reply as it grows (`content`), the products shown
    (`products`), and a final payload with the turn's diagnostics.
    """

    request = urllib.request.Request(
        f"{CHAIN_SERVER}/query/stream",
        data=json.dumps(
            {
                "query": text,
                "user_id": user_id,
                "session_id": conversation_id,
                "conversation_id": conversation_id,
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    reply, products, diagnostics = "", [], {}
    started = monotonic()
    with urllib.request.urlopen(request, timeout=300) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: ") or "[DONE]" in line:
                continue
            event = json.loads(line[6:])
            body = event.get("payload")
            if event.get("type") == "content":
                reply = body or reply
            elif event.get("type") == "products" and isinstance(body, list):
                products = body
            elif event.get("type") == "error":
                reply = f"ERROR: {body}"
            if isinstance(body, dict) and body.get("agent_diagnostics"):
                diagnostics = body["agent_diagnostics"]
    return {
        "reply": reply,
        "products": products,
        "tools": [c.get("tool_name") for c in diagnostics.get("tool_calls") or []],
        "skills": diagnostics.get("skill_files_read") or [],
        "seconds": round(monotonic() - started, 1),
        "diagnostics": diagnostics,
    }


def show(turn: dict, width: int = 300) -> None:
    print(turn["reply"][:width] + ("..." if len(turn["reply"]) > width else ""))
    for product in turn["products"][:6]:
        price = (product.get("price") or {}).get("amount")
        print(f"  - {product.get('display_name')}  ${price}")
    skills = [s.split("/")[-2] for s in turn["skills"]]
    print(f"[skills {skills} | tools {turn['tools']} | {turn['seconds']}s]")


def skill_header(path: Path) -> dict:
    """The front matter of a SKILL.md: scalar fields and one-level lists."""

    lines = path.read_text().split("---")[1].strip().splitlines()
    header: dict = {}
    key = None
    for line in lines:
        if line.startswith("  - ") and key:
            header.setdefault(key, []).append(line[4:].strip())
        elif ":" in line and not line.startswith(" "):
            key, _, value = line.partition(":")
            key, value = key.strip(), value.strip()
            if value == "[]":
                header[key] = []
            elif value:
                header[key] = value
    return header


# Phoenix -------------------------------------------------------------------


def load_spans(max_pages: int = 10, page_size: int = 1000) -> list[dict]:
    """Spans from Phoenix, newest first. Raise max_pages to reach older ones.

    Phoenix rejects a limit above 1000 rather than clamping, and pages with a
    `next_cursor`.
    """

    rows, cursor = [], None
    for _ in range(max_pages):
        url = f"{PHOENIX}/v1/projects/default/spans?limit={page_size}"
        if cursor:
            url += f"&cursor={cursor}"
        page = get(url)
        rows.extend(page["data"])
        cursor = page.get("next_cursor")
        if not cursor:
            break
    return rows


def wait_for_turns(session_id: str, count: int, timeout_s: float = 60) -> list[dict]:
    """Spans, once Phoenix holds `count` turns of the session.

    The chain-server exports spans in batches, so the last turn lands a few
    seconds after its reply.
    """

    started = monotonic()
    while True:
        rows = load_spans(max_pages=1)
        turns = [r for r in rows if r["name"] == "turn"
                 and r["attributes"].get("session.id") == session_id]
        if len(turns) >= count or monotonic() - started > timeout_s:
            print(f"{len(turns)} of {count} turns of {session_id} in Phoenix")
            return rows
        sleep(2)


def by_trace(rows: list[dict]) -> dict[str, list[dict]]:
    traces: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        traces[row["context"]["trace_id"]].append(row)
    for spans in traces.values():
        spans.sort(key=lambda r: r["start_time"])
    return traces


def seconds(span: dict) -> float:
    def when(stamp: str) -> datetime:
        return datetime.fromisoformat(stamp.replace("Z", "+00:00"))

    return (when(span["end_time"]) - when(span["start_time"])).total_seconds()


def as_list(value) -> list:
    """The turn span stores lists as their Python repr, e.g. "['a', 'b']"."""

    if isinstance(value, list):
        return value
    try:
        return json.loads(str(value).replace("'", '"'))
    except ValueError:
        return [value] if value else []
