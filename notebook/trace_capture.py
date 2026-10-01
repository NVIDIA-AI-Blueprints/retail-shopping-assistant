# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model calls recorded in Phoenix, written as an AIPerf trace.

Notebook 4 uses this to turn journey replays into a file AIPerf can send to a
model server again, request for request: `--custom-dataset-type mooncake_trace`
with one line per model call, carrying the exact `messages`, `tools` and
request settings the agent sent.

The source is the LangChain `ChatOpenAI` span, not Relay's model span: it is the
only one that records the tool schemas, and it also covers the final response
editor call, which Relay does not wrap.

Standard library only.
"""

from __future__ import annotations

import json
import lzma
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic, sleep

from helpers import PHOENIX, REPO, get

JOURNEYS = REPO / "tests" / "evaluation" / "datasets" / "val" / "scripts" / "journeys"
#: Committed, so Notebook 5 finds the trace in a plain checkout on the GPU host.
TRACES = REPO / "notebook" / "traces"

#: Request settings that stay behind. The model is named by whoever replays the
#: trace, `stream` by AIPerf's `--streaming`, the length by `output_length`, and
#: the tools travel in their own field.
_NOT_REPLAYED = {
    "model",
    "model_name",
    "stream",
    "_type",
    "stop",
    "tools",
    "max_completion_tokens",
    "max_tokens",
}


def journey_ids() -> list[str]:
    """J01 to J25, by file name. `--only J` would also match probes with a j."""

    return sorted(path.stem.split("_")[0] for path in JOURNEYS.glob("J*.yaml"))


def now() -> str:
    """UTC, in the form Phoenix writes span times, so the two compare as text."""

    return datetime.now(timezone.utc).isoformat()  # noqa: UP017 - datetime.UTC needs 3.11; the notebooks support 3.10


def model_calls(
    session_prefix: str, since: str, page_size: int = 1000, max_pages: int = 200
) -> list[dict]:
    """The `ChatOpenAI` spans of sessions named `session_prefix*`, oldest first.

    Pages newest first and stops once a page reaches back past `since`. Phoenix
    can filter by name, session or time itself, but on a project this size each
    filtered query scans the whole table and takes minutes, where a page of the
    newest spans takes seconds.
    """

    rows, cursor = [], None
    for _ in range(max_pages):
        url = f"{PHOENIX}/v1/projects/default/spans?limit={page_size}"
        if cursor:
            url += f"&cursor={cursor}"
        page = get(url, timeout=300)
        rows.extend(page["data"])
        cursor = page.get("next_cursor")
        if not cursor or min(row["start_time"] for row in page["data"]) < since:
            break
    calls = [
        row
        for row in rows
        if row["name"] == "ChatOpenAI"
        and row["start_time"] >= since
        and str(row["attributes"].get("session.id", "")).startswith(session_prefix)
    ]
    return sorted(calls, key=lambda row: row["start_time"])


def _nested(attributes: dict, prefix: str) -> dict:
    """`{prefix}.0.a.b: v` as `{0: {"a.b": v}}`."""

    found: dict[int, dict] = defaultdict(dict)
    pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.(.+)$")
    for key, value in attributes.items():
        match = pattern.match(key)
        if match:
            found[int(match[1])][match[2]] = value
    return dict(sorted(found.items()))


def _content(fields: dict):
    """A message's content: the plain string, or its list of parts."""

    parts = _nested(fields, "message.contents")
    if not parts:
        return fields.get("message.content", "")
    content = []
    for part in parts.values():
        if part.get("message_content.type") == "image":
            content.append(
                {"type": "image_url", "image_url": {"url": part["message_content.image.image.url"]}}
            )
        else:
            content.append({"type": "text", "text": part.get("message_content.text", "")})
    return content


def messages(span: dict) -> list[dict]:
    """The OpenAI chat messages the call sent, rebuilt from OpenInference's
    flattened `llm.input_messages.*` attributes."""

    rebuilt = []
    for fields in _nested(span["attributes"], "llm.input_messages").values():
        message = {"role": fields["message.role"], "content": _content(fields)}
        calls = _nested(fields, "message.tool_calls")
        if calls:
            message["tool_calls"] = [
                {
                    "id": call.get("tool_call.id", ""),
                    "type": "function",
                    "function": {
                        "name": call["tool_call.function.name"],
                        "arguments": call.get("tool_call.function.arguments", "{}"),
                    },
                }
                for call in calls.values()
            ]
        if "message.tool_call_id" in fields:
            message["tool_call_id"] = fields["message.tool_call_id"]
        if "message.name" in fields:
            message["name"] = fields["message.name"]
        rebuilt.append(message)
    return rebuilt


def settings(span: dict) -> dict:
    """What the agent asked of the model, as it went over the wire."""

    return json.loads(span["attributes"].get("llm.invocation_parameters", "{}"))


def trace_line(span: dict) -> dict:
    """One AIPerf `mooncake_trace` line, without its timing."""

    sent = settings(span)
    extra = {
        key: value for key, value in sent.items() if key not in _NOT_REPLAYED and value is not None
    }
    # LangChain's `extra_body` is its own wrapper; on the wire these are
    # top-level fields of the request, and AIPerf merges `extra` the same way.
    extra.update(extra.pop("extra_body", {}))
    line = {
        "session_id": span["attributes"]["session.id"],
        "messages": messages(span),
        "output_length": int(span["attributes"].get("llm.token_count.completion", 0)),
        "extra": extra,
    }
    if sent.get("tools"):
        line["tools"] = sent["tools"]
    return line


def _ms(timestamp: str) -> float:
    return datetime.fromisoformat(timestamp).timestamp() * 1000


def build_trace(calls: list[dict]) -> tuple[list[dict], list[dict]]:
    """The AIPerf lines, and a manifest with what was recorded for each.

    A session's first call carries `timestamp`, its start relative to the first
    call of the capture. Every later call carries `delay`: the time the agent
    spent between the previous call's reply and this request, running tools.
    The model's own time is left out, so the replay spends the server under
    test's time, not the hosted endpoint's.
    """

    origin = _ms(calls[0]["start_time"])
    lines, manifest = [], []
    previous_end: dict[str, float] = {}
    turns: dict[str, set] = defaultdict(set)
    for span in calls:
        line = trace_line(span)
        session = line["session_id"]
        start, end = _ms(span["start_time"]), _ms(span["end_time"])
        if session in previous_end:
            line["delay"] = round(max(0.0, start - previous_end[session]), 1)
        else:
            line["timestamp"] = round(start - origin, 1)
        previous_end[session] = end
        turns[session].add(span["context"]["trace_id"])
        attributes = span["attributes"]
        lines.append(line)
        manifest.append(
            {
                "session_id": session,
                "call": sum(1 for row in manifest if row["session_id"] == session),
                "turn": len(turns[session]) - 1,
                "trace_id": span["context"]["trace_id"],
                "span_id": span["context"]["span_id"],
                "model": settings(span).get("model"),
                "prompt_tokens": int(attributes.get("llm.token_count.prompt", 0)),
                "completion_tokens": int(attributes.get("llm.token_count.completion", 0)),
                "cache_read_tokens": int(
                    attributes.get("llm.token_count.prompt_details.cache_read", 0)
                ),
                "model_ms": round(end - start, 1),
            }
        )
    return lines, manifest


def check_against_replay(manifest: list[dict], results: Path) -> list[dict]:
    """Per conversation, whether the trace holds every model call the replay
    counted, turn by turn, and the same prompt tokens.

    The replay runner records its own count from the chain-server's reply, so a
    span Phoenix dropped or a call the capture missed shows up here.
    """

    recorded: dict[str, list[dict]] = defaultdict(list)
    for row in manifest:
        recorded[row["session_id"]].append(row)
    checks = []
    for path in sorted((results / "raw").glob("*.json")):
        run = json.loads(path.read_text())
        session = run["identity"]["conversation_id"]
        rows = recorded.get(session, [])
        replayed = [turn["token_usage"]["model_calls"] for turn in run["turns"]]
        captured = [
            sum(1 for row in rows if row["turn"] == index) for index in range(len(run["turns"]))
        ]
        replay_tokens = sum(turn["token_usage"]["input_tokens"] for turn in run["turns"])
        trace_tokens = sum(row["prompt_tokens"] for row in rows)
        checks.append(
            {
                "scenario": run["id"],
                "outcome": run["outcome"],
                "turns": len(run["turns"]),
                "calls": len(rows),
                "calls_match": replayed == captured,
                "prompt_tokens_match": replay_tokens == trace_tokens,
            }
        )
    return checks


def capture(label: str, since: str, results: Path, timeout_s: float = 300, every_s: float = 15):
    """The trace of a replay run, once Phoenix holds every call it made, and
    how many calls the endpoint refused.

    The chain-server exports spans in batches, so the last turns land a few
    seconds after the replay finishes.
    """

    deadline = monotonic() + timeout_s
    while True:
        calls = model_calls(f"{label}-", since)
        # A call the endpoint refused (a 429, say) never reached the model:
        # replaying it would ask the server under test for work nobody got.
        answered = [span for span in calls if span["status_code"] != "ERROR"]
        lines, manifest = build_trace(answered)
        checks = check_against_replay(manifest, results)
        if all(check["calls_match"] for check in checks) or monotonic() > deadline:
            return lines, manifest, checks, len(calls) - len(answered)
        sleep(every_s)


def kind(line: dict) -> str:
    """Which of the three model calls a turn makes this one is."""

    tools = [tool["function"]["name"] for tool in line.get("tools", [])]
    if not tools:
        return "response editor"
    if tools == ["activate_shopper_skills_tool"]:
        return "skill selection"
    return "agent"


def journey_of(session_id: str) -> str:
    """`J01` from `trace-02401c23-0a1b2c3d-J01_wedding_abroad-0`."""

    return re.search(r"-(J\d+)_", session_id)[1]


def by_journey(lines: list[dict]) -> dict[str, list[dict]]:
    """Each journey's calls as a trace of its own, starting at time 0."""

    journeys: dict[str, list[dict]] = defaultdict(list)
    for line in lines:
        journeys[journey_of(line["session_id"])].append(line)
    for calls in journeys.values():
        start = calls[0]["timestamp"]
        calls[:] = [
            {**line, "timestamp": line["timestamp"] - start} if "timestamp" in line else line
            for line in calls
        ]
    return dict(sorted(journeys.items()))


def _open(path: Path, mode: str):
    # Each call repeats most of the one before, tens of kilobytes back: past
    # gzip's 32 KB window, well inside xz's. 38 MB of trace commits as 0.2 MB.
    if path.suffix == ".xz":
        return lzma.open(path, mode + "t", encoding="utf-8", preset=9 if "w" in mode else None)
    return path.open(mode, encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    """JSON lines, compressed with xz when the name ends in `.xz`."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with _open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: Path) -> list[dict]:
    with _open(path, "r") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prompt_text(line: dict) -> str:
    """The request as one string, tools first, for comparing prefixes.

    Only an approximation of what the model reads: the chat template decides
    the real layout, and the server counts in tokens, not characters.
    """

    return json.dumps(line.get("tools", []), sort_keys=True) + "".join(
        json.dumps(message, sort_keys=True) for message in line["messages"]
    )


def shared_prefix(a: str, b: str) -> int:
    size = min(len(a), len(b))
    low, high = 0, size
    while low < high:
        middle = (low + high + 1) // 2
        if a[:middle] == b[:middle]:
            low = middle
        else:
            high = middle - 1
    return low


def prefix_reuse(lines: list[dict], window: int = 300) -> list[dict]:
    """For each call, the longest start of its request an earlier call sent.

    A prefix cache reuses the longest prefix it still holds, whichever call
    left it. `from_session` looks only at the same conversation's earlier
    calls; `from_any` also at the last `window` calls of every conversation,
    which is where a shared system prompt and tool list pay off.
    """

    seen: list[tuple[str, str]] = []
    reuse = []
    for line in lines:
        text = prompt_text(line)
        session = line["session_id"]
        earlier = [
            (other_session, shared_prefix(text, other)) for other_session, other in seen[-window:]
        ]
        from_session = max(
            [size for other_session, size in earlier if other_session == session], default=0
        )
        from_any = max([size for _, size in earlier], default=0)
        reuse.append(
            {
                "session_id": session,
                "chars": len(text),
                "from_session": from_session / len(text),
                "from_any": from_any / len(text),
            }
        )
        seen.append((session, text))
    return reuse
