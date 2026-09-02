#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How many shoppers can this deployment serve at once, and what stops it.

journey_load.py holds each concurrency level for a fixed number of minutes, so
different levels finish different amounts of work and their latencies are not
strictly comparable. This fixes the work instead: N shoppers each play one
complete journey, and the run ends when the last of them finishes. Every level
does exactly N x (turns per journey), so completion time, latency and
throughput can be read against each other directly.

It reports **time to first token** as the headline latency. A turn keeps
working long after the shopper starts reading -- tools, the grounding pass, the
cart read -- so total turn time measures the system's effort while first-token
measures the wait. Sizing a streaming UI against total turn time buys hardware
nobody needed.

    # the model's ceiling: shoppers who never pause, all starting together
    python3 benchmarks/shopper_study.py --levels 1,2,4,8,16,32 --think 0

    # what real shoppers look like: they read, and they do not arrive in step
    python3 benchmarks/shopper_study.py --levels 1,2,4,8,16,32 --think 20 --stagger 30

    # honest prefix-cache number: different conversations, not N copies of one
    python3 benchmarks/shopper_study.py --levels 8 --mixed

Nothing is judged and nothing is asserted; correctness is replay.py's job.
Checking answers costs further model calls, which is the resource under test.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.evaluation.src.config import load_eval_config  # noqa: E402
from tests.evaluation.src.replay import (  # noqa: E402
    SCRIPTS_ROOT,
    Assistant,
    load_scenarios,
)

#: Far outside any real shopper id, with its own range per level so no two
#: levels ever share a conversation or a cart.
STUDY_USER_BASE = 820_000_000

PROM = "http://localhost:9090"

#: Probed directly, not through the agent. The agent answers a dead model with
#: an apology, and an apology has a first-token time like any other reply.
MODEL = "http://localhost:8000"

#: What the chain server says when a model call fails. These are ordinary
#: replies as far as the harness is concerned -- they stream, they have a
#: first-token time, and they raise nothing -- so without matching them a run
#: against a dead model reports full throughput and no errors. One did.
FALLBACKS = (
    "i could not complete that shopping request",
    "something went wrong",
    "please try again",
    "temporarily unavailable",
    "is unavailable",
)


def model_alive(timeout: float = 10.0) -> bool:
    """Is the model server itself answering?

    A hang here does not look like a crash: the container stays up, the
    weights stay resident, and the process list looks healthy, while every
    request sits until something upstream times out.
    """
    try:
        with urllib.request.urlopen(f"{MODEL}/health", timeout=timeout) as response:
            return response.status == 200
    except Exception:
        return False


@dataclass
class Turn:
    shopper: int
    index: int
    ttft: float | None
    seconds: float
    #: A reply the agent produced without the model. Counted, not raised.
    fallback: bool = False


@dataclass
class Level:
    concurrency: int
    turns: list[Turn] = field(default_factory=list)
    errors: int = 0
    first_error: str | None = None
    seconds: float = 0.0
    started_at: float = 0.0
    ended_at: float = 0.0
    resources: dict = field(default_factory=dict)
    model_ok_before: bool = True
    model_ok_after: bool = True
    #: Why the level is not a measurement, or None if it is one.
    invalid: str | None = None

    @property
    def fallbacks(self) -> int:
        return sum(1 for t in self.turns if t.fallback)

    def pct(self, values: list[float], p: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        return ordered[min(int(len(ordered) * p), len(ordered) - 1)]

    @property
    def ttfts(self) -> list[float]:
        return [t.ttft for t in self.turns if t.ttft is not None]

    @property
    def walls(self) -> list[float]:
        return [t.seconds for t in self.turns]

    @property
    def turns_per_minute(self) -> float:
        return len(self.turns) / self.seconds * 60 if self.seconds else 0.0


def prom_range(expr: str, start: float, end: float, step: int = 5) -> list[float]:
    """Every sample of expr across the window, flattened across series."""
    url = f"{PROM}/api/v1/query_range?" + urllib.parse.urlencode(
        {"query": expr, "start": f"{start:.0f}", "end": f"{end:.0f}", "step": step})
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            result = json.load(resp)["data"]["result"]
    except Exception:
        return []
    return [float(v) for s in result for _, v in s["values"]]


def prom_delta(expr: str, start: float, end: float) -> float:
    """How much a counter advanced across the window.

    Two instant reads rather than increase(), which extrapolates over its
    lookback and would bleed the previous level into this one.
    """
    def at(when: float) -> float | None:
        url = f"{PROM}/api/v1/query?" + urllib.parse.urlencode(
            {"query": expr, "time": f"{when:.0f}"})
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                result = json.load(resp)["data"]["result"]
        except Exception:
            return None
        return float(result[0]["value"][1]) if result else None

    a, b = at(start), at(end)
    return 0.0 if a is None or b is None else b - a


#: Everything outside the model server. Named rather than matched loosely, so a
#: new observability container does not quietly get counted as application load.
APP_SERVICES = ("chain-server|catalog-retriever|memory-retriever|milvus|rails|"
                "etcd|minio|nginx|frontend|otel-collector|phoenix")


def capture_resources(start: float, end: float) -> dict:
    """What the machine was doing, from Prometheus rather than a local sampler.

    Reading the same store the dashboard reads means a number here and a panel
    there cannot disagree, and it covers every service rather than only the
    ones this script thought to watch.
    """
    gpu = prom_range("DCGM_FI_DEV_GPU_UTIL", start, end)
    kv = prom_range("vllm:kv_cache_usage_perc", start, end)
    app_cpu = prom_range(
        f'sum(container_cpu_percent{{service=~"{APP_SERVICES}"}})', start, end)
    nim_cpu = prom_range(
        'container_cpu_percent{service="nemotron"}', start, end)
    app_mem = prom_range(
        f'sum(container_memory_mib{{service=~"{APP_SERVICES}"}})', start, end)
    queries = prom_delta("vllm:prefix_cache_queries_total", start, end)
    hits = prom_delta("vllm:prefix_cache_hits_total", start, end)
    return {
        "gpu_mean_pct": round(sum(gpu) / len(gpu), 1) if gpu else None,
        "gpu_peak_pct": round(max(gpu), 1) if gpu else None,
        "kv_peak_pct": round(max(kv) * 100, 2) if kv else None,
        "app_cpu_peak_cores": round(max(app_cpu) / 100, 2) if app_cpu else None,
        "app_mem_peak_mib": round(max(app_mem)) if app_mem else None,
        "nim_cpu_peak_cores": round(max(nim_cpu) / 100, 2) if nim_cpu else None,
        "prompt_tokens": round(prom_delta("vllm:prompt_tokens_total", start, end)),
        "output_tokens": round(prom_delta("vllm:generation_tokens_total", start, end)),
        "cache_hit_pct": round(hits / queries * 100, 1) if queries else None,
    }


def play(assistant, scenario, identity, level, shopper, think, lock,
         stop) -> None:
    """One shopper, one complete journey, every turn in order."""
    for index, step in enumerate(scenario.get("turns") or [], start=1):
        if stop.is_set():
            return
        attachment = None
        if step.get("attach"):
            attachment = SCRIPTS_ROOT / "assets" / str(step["attach"])
        started = time.monotonic()
        try:
            answer = assistant.say(identity, str(step["say"]), attachment)
        except Exception as exc:
            with lock:
                level.errors += 1
                if level.first_error is None:
                    level.first_error = f"{type(exc).__name__}: {exc}"[:140]
            return
        reply = str(answer.get("reply") or "").strip().lower()
        with lock:
            level.turns.append(Turn(
                shopper=shopper, index=index,
                ttft=answer.get("ttft"),
                seconds=time.monotonic() - started,
                fallback=not reply or any(m in reply for m in FALLBACKS)))
        if think and index < len(scenario.get("turns") or []):
            # Jittered, because a fixed pause would march every shopper in
            # lockstep for the whole journey -- which is the arrival pattern
            # this option exists to avoid.
            time.sleep(random.uniform(think * 0.5, think * 1.5))


def run_level(assistants, scenarios, concurrency: int, think: float,
              stagger: float, settle: float) -> Level:
    level = Level(concurrency=concurrency)
    lock = threading.Lock()
    stop = threading.Event()

    def worker(slot: int) -> None:
        if stagger:
            time.sleep(stagger * slot / max(concurrency - 1, 1))
        assistant = assistants[slot % len(assistants)]
        scenario = scenarios[slot % len(scenarios)]
        marker = f"study-{concurrency}-{slot}-{int(time.time())}"
        identity = {
            "user_id": STUDY_USER_BASE + concurrency * 1000 + slot,
            "session_id": marker,
            "conversation_id": marker,
            "cart_id": marker,
        }
        play(assistant, scenario, identity, level, slot, think, lock, stop)

    level.model_ok_before = model_alive()
    if not level.model_ok_before:
        level.invalid = "model was not answering before the level started"
        return level

    level.started_at = time.time()
    begin = time.monotonic()
    threads = [threading.Thread(target=worker, args=(i,))
               for i in range(concurrency)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    level.seconds = time.monotonic() - begin
    level.ended_at = time.time()

    # Prometheus scrapes on its own schedule, so the tail of the run is not in
    # the store yet when the last thread returns.
    time.sleep(settle)
    level.resources = capture_resources(level.started_at, level.ended_at)
    level.model_ok_after = model_alive()

    # Three independent ways to catch a level that ran against a model which
    # was not serving. Any one of them alone has a blind spot: the probe can
    # pass either side of a hang that happened in between, the counter can
    # move because a handful of early turns got through, and the fallback
    # count can stay low if the agent happens to answer from cache.
    prompt_tokens = level.resources.get("prompt_tokens") or 0
    share = level.fallbacks / len(level.turns) if level.turns else 0.0
    if not level.model_ok_after:
        level.invalid = "model stopped answering during the level"
    elif level.turns and not prompt_tokens:
        level.invalid = "model processed no prompt tokens during the level"
    elif share > 0.2:
        level.invalid = f"{share:.0%} of replies were the agent's failure text"
    return level


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", default="1,2,4,8,16,32")
    parser.add_argument("--journey", default="J01",
                        help="journey every shopper plays (default J01)")
    parser.add_argument("--mixed", action="store_true",
                        help="rotate through every journey instead of N copies "
                             "of one. N copies share every word, which is the "
                             "best case prefix caching will ever see and not "
                             "what real shoppers do.")
    parser.add_argument("--think", type=float, default=0.0,
                        help="seconds a shopper spends reading between turns, "
                             "jittered +/-50%%. 0 is back to back, a worst case.")
    parser.add_argument("--stagger", type=float, default=0.0,
                        help="spread the start of the N shoppers over this many "
                             "seconds. 0 starts them together, which makes every "
                             "request miss the prefix cache at once.")
    parser.add_argument("--text-only", action="store_true",
                        help="skip journeys that attach an image or video. "
                             "Without an image embedding service deployed "
                             "those turns exercise a fallback, so including "
                             "them measures the fallback and not the product.")
    parser.add_argument("--settle", type=float, default=10.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config = load_eval_config()
    assistant = Assistant(config)
    assistants = [assistant]

    scenarios = load_scenarios(None if args.mixed else args.journey)
    scenarios = [s for s in scenarios if s["id"].startswith("J")]
    if args.text_only:
        withheld = [s["id"] for s in scenarios
                    if any(t.get("attach") for t in (s.get("turns") or []))]
        scenarios = [s for s in scenarios
                     if not any(t.get("attach") for t in (s.get("turns") or []))]
        if withheld:
            print(f"  text only   skipping {len(withheld)} journey(s) that "
                  f"attach media: {', '.join(withheld)}")
    if not scenarios:
        raise SystemExit(f"no journey matched {args.journey!r}")
    turns_each = len(scenarios[0].get("turns") or [])

    levels = [int(v) for v in args.levels.split(",") if v.strip()]
    label = ("mixed journeys" if args.mixed
             else f"{scenarios[0]['id']} x N ({turns_each} turns each)")
    print(f"shopper study: {label}")
    print(f"  think time {args.think:g}s between turns"
          + (", jittered" if args.think else " (back to back)"))
    print(f"  start       "
          + (f"spread over {args.stagger:g}s" if args.stagger else "all together"))
    print(f"  levels      {', '.join(map(str, levels))}\n")

    header = (f"{'shoppers':>8} {'turns':>6} {'wall s':>7} {'turns/min':>10} "
              f"{'TTFT p50':>9} {'TTFT p95':>9} {'turn p95':>9} "
              f"{'GPU%':>5} {'app cores':>10} {'KV%':>6} {'cache%':>7} {'err':>4}")
    print(header)
    print("-" * len(header))

    results = []
    for concurrency in levels:
        level = run_level(assistants, scenarios, concurrency, args.think,
                          args.stagger, args.settle)
        r = level.resources
        print(f"{concurrency:8d} {len(level.turns):6d} {level.seconds:7.0f} "
              f"{level.turns_per_minute:10.1f} "
              f"{level.pct(level.ttfts, 0.50):9.2f} "
              f"{level.pct(level.ttfts, 0.95):9.2f} "
              f"{level.pct(level.walls, 0.95):9.1f} "
              f"{r.get('gpu_mean_pct') or 0:5.0f} "
              f"{r.get('app_cpu_peak_cores') or 0:10.2f} "
              f"{r.get('kv_peak_pct') or 0:6.1f} "
              f"{r.get('cache_hit_pct') or 0:7.1f} "
              f"{level.errors:4d}"
              + (f"   <-- DISCARD: {level.invalid}" if level.invalid else ""))
        results.append({
            "invalid": level.invalid,
            "fallback_replies": level.fallbacks,
            "model_ok_before": level.model_ok_before,
            "model_ok_after": level.model_ok_after,
            "concurrency": concurrency,
            "turns": len(level.turns),
            "seconds": round(level.seconds, 1),
            "turns_per_minute": round(level.turns_per_minute, 2),
            "ttft_p50": round(level.pct(level.ttfts, 0.50), 3),
            "ttft_p95": round(level.pct(level.ttfts, 0.95), 3),
            "ttft_mean": round(statistics.fmean(level.ttfts), 3) if level.ttfts else None,
            "turn_p95": round(level.pct(level.walls, 0.95), 2),
            "errors": level.errors,
            "first_error": level.first_error,
            "started_at": level.started_at,
            "ended_at": level.ended_at,
            **level.resources,
        })

        # Every later level would run against the same dead server and report
        # the same confident nonsense, only louder.
        if level.invalid:
            print(f"\n  stopping: {level.invalid}.")
            print("  Levels above this one were not attempted. Restart the "
                  "model server before rerunning.")
            break

    out = Path(args.out or (
        f"/tmp/shopper_study_{'mixed' if args.mixed else args.journey}"
        f"_think{args.think:g}_stagger{args.stagger:g}.json"))
    out.write_text(json.dumps({
        "journey": "mixed" if args.mixed else scenarios[0]["id"],
        "journeys_played": [s["id"] for s in scenarios],
        "text_only": args.text_only,
        "turns_per_journey": turns_each,
        "think_seconds": args.think,
        "stagger_seconds": args.stagger,
        "levels": results,
    }, indent=1))
    print(f"\n  written {out}")

    usable = [r for r in results if not r["invalid"]]
    if not usable:
        raise SystemExit("  no level produced a usable measurement")
    best = max(usable, key=lambda r: r["turns_per_minute"])
    print(f"  peak {best['turns_per_minute']:.1f} turns/min at "
          f"{best['concurrency']} shoppers")
    cores = best.get("app_cpu_peak_cores") or 0
    if cores and best["concurrency"]:
        per = cores / best["concurrency"]
        print(f"  application tier used {cores:.2f} cores at that point, "
              f"{per:.3f} per shopper")


if __name__ == "__main__":
    main()
