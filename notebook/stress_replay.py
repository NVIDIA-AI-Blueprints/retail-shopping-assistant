# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay Notebook 4's trace against a vLLM server with AIPerf, and read the results.

Notebook 5 runs on the GPU machine, which has the model server, AIPerf and a
checkout of this repository, and nothing of the assistant. So this module
imports nothing from the rest of the repo, and only the standard library, plus
IPython when present for the live GPU view.
"""

from __future__ import annotations

import csv
import json
import lzma
import os
import re
import subprocess
import time
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
    refresh: float = 2,
) -> subprocess.CompletedProcess:
    """One AIPerf run: every session once, `concurrency` at a time.

    Records the serving GPUs each second to `gpu.csv`. In a notebook, shows them
    live, redrawn every `refresh` seconds, and ends with the run's averages.
    """

    if not AIPERF.is_file():
        raise FileNotFoundError(
            f"AIPerf not found at {AIPERF}. Install it as in the notebook's "
            "'Before you start', step 3."
        )
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
    total = _count_lines(trace)
    gpus = serving_gpus()
    board = _Board()
    started = time.monotonic()
    with (out / "aiperf.out").open("w") as log, (out / "gpu.csv").open("w") as gpu_log:
        sampler = (
            subprocess.Popen(
                [
                    "nvidia-smi",
                    f"--query-gpu={_GPU_FIELDS}",
                    "--format=csv,noheader,nounits",
                    "-i",
                    ",".join(gpus),
                    "-l",
                    "1",
                ],
                stdout=gpu_log,
                stderr=subprocess.DEVNULL,
            )
            if gpus
            else None
        )
        run = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            while True:
                try:
                    run.wait(timeout=refresh)
                    break
                except subprocess.TimeoutExpired:
                    done = _count_lines(out / "profile_export.jsonl")
                    latest = {sample["gpu"]: sample for sample in _gpu_samples(out)}
                    board.show(
                        f"{out.name}: {done}/{total} calls ({done / total:.0%}),"
                        f" {_clock(time.monotonic() - started)}",
                        list(latest.values()),
                    )
        finally:
            # An interrupted run must not keep loading the server.
            if run.poll() is None:
                run.kill()
                run.wait()
            if sampler:
                sampler.terminate()
                sampler.wait()
    samples = _gpu_samples(out)
    if samples:
        board.show(
            f"{out.name}: done in {_clock(time.monotonic() - started)}. Average per GPU over the run",
            _averages(samples),
            final=True,
        )
    return subprocess.CompletedProcess(command, run.returncode)


_GPU_FIELDS = "index,utilization.gpu,memory.used,memory.total,power.draw,power.limit"


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def _clock(seconds: float) -> str:
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def serving_gpus() -> list[str]:
    """The GPUs holding the model: more than 1 GiB in use. Empty without `nvidia-smi`."""

    try:
        rows = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        return []
    return [index for index, used in (row.split(", ") for row in rows) if float(used) > 1024]


def _parse_gpu(row: str) -> dict | None:
    try:
        index, busy, memory, total, power, limit = (field.strip() for field in row.split(","))
        return {
            "gpu": index,
            "busy": float(busy),
            "memory_mib": float(memory),
            "total_mib": float(total),
            "power_w": float(power),
            "limit_w": float(limit),
        }
    except ValueError:  # "[N/A]" fields, or a line cut off mid-write
        return None


def _gpu_samples(out: Path) -> list[dict]:
    path = out / "gpu.csv"
    rows = path.read_text().splitlines() if path.exists() else []
    return [sample for sample in map(_parse_gpu, rows) if sample]


def _gpu_query(gpus: list[str]) -> list[dict]:
    rows = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={_GPU_FIELDS}",
            "--format=csv,noheader,nounits",
            "-i",
            ",".join(gpus),
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.splitlines()
    return [sample for sample in map(_parse_gpu, rows) if sample]


def _averages(samples: list[dict]) -> list[dict]:
    """Per GPU: mean busy share and power, peak memory."""

    by_gpu = defaultdict(list)
    for sample in samples:
        by_gpu[sample["gpu"]].append(sample)
    return [
        {
            "gpu": gpu,
            "busy": sum(s["busy"] for s in rows) / len(rows),
            "power_w": sum(s["power_w"] for s in rows) / len(rows),
            "limit_w": rows[0]["limit_w"],
            "memory_mib": max(s["memory_mib"] for s in rows),
            "total_mib": rows[0]["total_mib"],
        }
        for gpu, rows in sorted(by_gpu.items(), key=lambda item: int(item[0]))
    ]


def _bar(share: float, label: str) -> str:
    share = max(0.0, min(1.0, share))
    color = "#76b900" if share < 0.6 else "#f5a623" if share < 0.9 else "#e0452e"
    return (
        '<div style="display:inline-block;width:160px;height:12px;background:#ddd;'
        f'vertical-align:middle"><div style="width:{share:.0%};height:100%;background:{color}">'
        f"</div></div>&nbsp;{label}"
    )


def _gpu_table(head: str, samples: list[dict]) -> str:
    rows = "".join(
        "<tr><td>GPU {}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            s["gpu"],
            _bar(s["busy"] / 100, "{:.0f}%".format(s["busy"])),
            _bar(
                s["power_w"] / s["limit_w"], "{:.0f} / {:.0f} W".format(s["power_w"], s["limit_w"])
            ),
            _bar(
                s["memory_mib"] / s["total_mib"],
                "{:.0f} / {:.0f} GiB".format(s["memory_mib"] / 1024, s["total_mib"] / 1024),
            ),
        )
        for s in samples
    )
    return (
        f"<b>{head}</b><table><tr><th>GPU</th><th>Busy</th><th>Power</th><th>Memory</th></tr>"
        f"{rows}</table>"
    )


def _gpu_lines(head: str, samples: list[dict]) -> str:
    return "\n".join(
        [head]
        + [
            f"  GPU {s['gpu']}: {s['busy']:3.0f}% busy   {s['power_w']:4.0f} of {s['limit_w']:.0f} W"
            f"   {s['memory_mib'] / 1024:.0f} of {s['total_mib'] / 1024:.0f} GiB"
            for s in samples
        ]
    )


class _Board:
    """GPU bars redrawn in place in a notebook; elsewhere, text every `every` seconds."""

    def __init__(self, every: float = 30):
        self.handle, self.every, self.last = None, every, 0.0
        try:
            from IPython import get_ipython
            from IPython.display import HTML, display
        except ImportError:
            return
        shell = get_ipython()
        if shell is not None and "IPKernelApp" in shell.config:
            self.html = HTML
            self.handle = display(HTML(""), display_id=True)

    def show(self, head: str, samples: list[dict], final: bool = False) -> None:
        if self.handle:
            self.handle.update(self.html(_gpu_table(head, samples)))
        elif final or time.monotonic() - self.last >= self.every:
            self.last = time.monotonic()
            print(_gpu_lines(head, samples), flush=True)


def gpu_dashboard(seconds: float = 20) -> list[dict]:
    """Watch the GPUs holding the model for `seconds`, then print their averages."""

    gpus = serving_gpus()
    if not gpus:
        print("No GPU holds the model, or there is no nvidia-smi here.")
        return []
    board = _Board(every=5)
    samples = []
    end = time.monotonic() + seconds
    while (left := end - time.monotonic()) > 0:
        now = _gpu_query(gpus)
        samples += now
        board.show(f"GPUs {', '.join(gpus)} hold the model. Live, {left:.0f} s left", now)
        time.sleep(1)
    averages = _averages(samples)
    board.show(f"GPUs {', '.join(gpus)}: average over {seconds:.0f} s", averages, final=True)
    if board.handle:
        print(_gpu_lines(f"Average over {seconds:.0f} s:", averages))
    return averages


def gpu_summary(out: Path) -> dict:
    """Averages over a run, per GPU. `busy` is the share of time any kernel ran,
    not how much of the GPU it used; power against the limit says more."""

    samples = _gpu_samples(out)
    if not samples:
        return {
            "gpu_busy_pct": None,
            "gpu_power_w": None,
            "gpu_power_limit_w": None,
            "gpu_memory_peak_gib": None,
        }
    return {
        "gpu_busy_pct": sum(sample["busy"] for sample in samples) / len(samples),
        "gpu_power_w": sum(sample["power_w"] for sample in samples) / len(samples),
        "gpu_power_limit_w": samples[0]["limit_w"],
        "gpu_memory_peak_gib": max(sample["memory_mib"] for sample in samples) / 1024,
    }


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
    """Add a row. Rewrites the file, so rows with new columns keep it aligned."""

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [*(read_summary(path) if path.exists() else []), row]
    fields = list(dict.fromkeys(field for existing in rows for field in existing))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_summary(path: Path = RESULTS / "summary.csv") -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))
