# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay Notebook 4's trace against a vLLM server with AIPerf, and read the results.

Notebook 5 runs on the GPU machine, which has the model server, AIPerf and a
checkout of this repository, and nothing of the assistant. So this module
imports nothing from the rest of the repo, and only the standard library.
"""

from __future__ import annotations

import csv
import json
import lzma
import os
import re
import subprocess
import urllib.request
import uuid
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRACES = HERE / "traces"
RESULTS = HERE / "stress_results"
AIPERF = Path.home() / "aiperf" / "bin" / "aiperf"

_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_NAMESPACE = uuid.UUID("5f3c1b7e-0d2a-4c1e-9b8f-3a6d2e4c7b10")


def read_jsonl(path: Path) -> list[dict]:
    opener = lzma.open if path.suffix == ".xz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def api_key() -> str | None:
    """The server's key: `NEMOTRON_API_KEY`, else the file `gpu/deploy.sh` wrote."""

    if os.environ.get("NEMOTRON_API_KEY"):
        return os.environ["NEMOTRON_API_KEY"]
    keyfile = Path(os.environ.get("KEYFILE", Path.home() / ".nemotron_api_key"))
    return keyfile.read_text().strip() if keyfile.is_file() else None


def get_text(url: str, key: str | None = None, timeout: float = 10) -> str:
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    with urllib.request.urlopen(
        urllib.request.Request(url, headers=headers), timeout=timeout
    ) as response:
        return response.read().decode()


_CACHE_FIELDS = ("enable_prefix_caching", "block_size", "mamba_block_size", "mamba_cache_mode")


def server_status(base_url: str, metrics_url: str, key: str | None = None) -> dict:
    """Whether the server is ready, what it serves, and how it resolved caching.

    The engine publishes the cache settings it resolved (`vllm:cache_config_info`),
    which is not always what was asked for: vLLM turns prefix caching off for
    hybrid models like this one unless told otherwise.
    """

    status = {"ready": False, "models": [], **dict.fromkeys(_CACHE_FIELDS)}
    for path in ("/health", "/v1/health/ready"):
        try:
            get_text(f"{base_url}{path}", key)
            status["ready"] = True
            break
        except Exception:  # noqa: BLE001 - reported, not handled
            continue
    if not status["ready"]:
        return status
    status["models"] = [
        model["id"] for model in json.loads(get_text(f"{base_url}/v1/models", key))["data"]
    ]
    info = next(
        (
            line
            for line in get_text(metrics_url, key).splitlines()
            if line.startswith("vllm:cache_config_info")
        ),
        "",
    )
    for field in _CACHE_FIELDS:
        found = re.search(rf'\b{field}="([^"]*)"', info)
        if found:
            status[field] = found[1]
    return status


def expand(lines: list[dict], copies: int, run: str) -> list[dict]:
    """`copies` shoppers per journey, every id fresh for this run.

    Each copy gets its own session and request ids, so copies share the system
    prompt and tools, as different shoppers do, and part at the first shopper
    message. A new `run` renames them all again, so a run cannot reuse the
    previous run's conversations from the cache: no server restart needed.
    """

    expanded = []
    for copy in range(copies):
        for line in lines:
            session = line["session_id"]
            renamed = f"{session}~{run}~{copy}"
            text = json.dumps(line["messages"], ensure_ascii=False).replace(session, renamed)
            scope = f"{run}/{copy}"
            text = _UUID.sub(
                lambda found, scope=scope: str(uuid.uuid5(_NAMESPACE, f"{scope}/{found[0]}")), text
            )
            expanded.append({**line, "session_id": renamed, "messages": json.loads(text)})
    # Grouped by session, in recorded order: AIPerf reads each session's
    # lines as its turns.
    order = {
        session: index
        for index, session in enumerate(dict.fromkeys(row["session_id"] for row in expanded))
    }
    return sorted(expanded, key=lambda row: order[row["session_id"]])


def replay(
    trace: Path,
    sessions: int,
    concurrency: int,
    out: Path,
    base_url: str,
    metrics_url: str,
    model: str,
    key: str | None = None,
) -> subprocess.CompletedProcess:
    """One AIPerf run: every session once, `concurrency` at a time."""

    command = [
        str(AIPERF),
        "profile",
        "--model",
        model,
        "--url",
        base_url,
        "--endpoint-type",
        "chat",
        "--streaming",
        "--input-file",
        str(trace),
        "--custom-dataset-type",
        "mooncake_trace",
        # Sessions, not requests: a request cap cuts the last conversations
        # short, because a trace wraps around to its first session.
        "--no-fixed-schedule",
        "--concurrency",
        str(concurrency),
        "--conversation-num",
        str(sessions),
        # Every reply exactly as long as the one recorded.
        "--extra-inputs",
        "ignore_eos:true",
        # Token counts from the server, so no tokenizer download is needed.
        "--use-server-token-count",
        "--tokenizer",
        "builtin",
        "--server-metrics",
        metrics_url,
        "--export-level",
        "records",
        "--artifact-dir",
        str(out),
        "--ui",
        "none",
        *(["--api-key", key] if key else []),
    ]
    out.mkdir(parents=True, exist_ok=True)
    with (out / "aiperf.out").open("w") as log:
        return subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)


def _server_total(server: dict, name: str) -> float | None:
    for key in (name, f"{name}_total"):
        metric = server.get("metrics", {}).get(key)
        if metric:
            return sum(series["stats"].get("total", 0.0) for series in metric["series"])
    return None


def summarize(out: Path) -> dict:
    """The numbers one run is reported by."""

    summary = json.loads((out / "profile_export_aiperf.json").read_text())
    server = json.loads((out / "server_metrics_export.json").read_text())
    hits = _server_total(server, "vllm:prefix_cache_hits")
    queries = _server_total(server, "vllm:prefix_cache_queries")
    errors = summary.get("error_summary") or []
    return {
        "requests": int(summary["request_count"]["avg"]),
        "errors": sum(error.get("count", 0) for error in errors),
        "ttft_p50_ms": summary["time_to_first_token"]["p50"],
        "ttft_p95_ms": summary["time_to_first_token"]["p95"],
        "latency_p95_ms": summary["request_latency"]["p95"],
        "requests_per_s": summary["request_throughput"]["avg"],
        "output_tokens_per_s": summary["output_token_throughput"]["avg"],
        "cache_hit_rate": hits / queries if hits is not None and queries else None,
        "duration_s": summary["benchmark_duration"]["avg"],
    }


def records(out: Path) -> list[dict]:
    return read_jsonl(out / "profile_export.jsonl")


def gates(out: Path, trace: list[dict], manifest: list[dict]) -> dict:
    """Whether the run replayed the trace faithfully.

    - complete: every session played once, every call of it.
    - in_order: each session's calls went out in recorded order.
    - output_lengths: every reply as long as recorded (`ignore_eos` worked).
    - prompt_tokens: the share of calls whose prompt the server counted within 2%
      of the capture, which says it read the same prompt.
    """

    calls = defaultdict(list)
    for row in records(out):
        calls[row["metadata"]["conversation_id"]].append(row)
    expected = defaultdict(list)
    for line in trace:
        expected[line["session_id"]].append(line)

    recorded = defaultdict(list)
    for row in manifest:
        recorded[row["session_id"]].append(row["prompt_tokens"])

    complete = set(calls) == set(expected) and all(
        len(calls[s]) == len(expected[s]) for s in expected
    )
    in_order = all(
        [
            row["metadata"]["turn_index"]
            for row in sorted(rows, key=lambda row: row["metadata"]["request_start_ns"])
        ]
        == list(range(len(rows)))
        for rows in calls.values()
    )
    lengths_ok, close, compared = True, 0, 0
    for session, rows in calls.items():
        original = session.split("~")[0]
        for row in rows:
            turn = row["metadata"]["turn_index"]
            metrics = row["metrics"]
            wanted = expected[session][turn].get("output_length")
            got = metrics.get("output_sequence_length", {}).get("value")
            if wanted and got != wanted:
                lengths_ok = False
            prompt = metrics.get("usage_prompt_tokens", {}).get("value")
            if prompt and turn < len(recorded[original]):
                compared += 1
                close += abs(prompt - recorded[original][turn]) <= 0.02 * recorded[original][turn]
    return {
        "complete": complete,
        "in_order": in_order,
        "output_lengths": lengths_ok,
        "prompt_tokens_within_2pct": close / compared if compared else None,
    }


def append_summary(row: dict, path: Path = RESULTS / "summary.csv") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)
    return path


def read_summary(path: Path = RESULTS / "summary.csv") -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))
