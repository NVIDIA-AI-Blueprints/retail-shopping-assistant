#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Push real conversations through the assistant until something gives, and say what.

Capacity planning needs one answer above all others: **which wall did we hit.**
Throughput alone cannot tell you, and the responses are opposite --

  the model is rate limiting     more pods make it worse; you need quota
  the model is saturated         more pods make it worse; you need more model
  chain-server is CPU bound      add chain-server pods, this is the good case
  the memory service refuses     raise MEMORY_MAX_CONCURRENT_REQUESTS

So this reports the limiting factor at every concurrency level, not a number.
Adding pods against a rate limit buys nothing and costs a day finding out.

It drives the val journeys rather than synthetic prompts, because the load that
matters is the load the system actually gets: multi-turn conversations that
search, resolve references, call tools and write carts. A synthetic single-turn
benchmark stresses the wrong thing and flatters the result -- most of a real
turn is the agent loop, not one model call.

    python scripts/journey_load.py                          # ramp 1,2,4,8
    python scripts/journey_load.py --levels 1,4,8,16
    python scripts/journey_load.py --only J12,J17 --minutes 3

Nothing is judged and nothing is asserted. Checking answers costs further model
calls, which is the resource under test. Correctness is `replay.py`'s job.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.evaluation.src.config import load_eval_config  # noqa: E402
from tests.evaluation.src.replay import (  # noqa: E402
    SCRIPTS_ROOT,
    Assistant,
    load_scenarios,
)


#: Far outside any real shopper id, and its own range per level so two levels
#: never share a conversation.
LOAD_USER_BASE = 810_000_000


def discover_chain_servers() -> list[str]:
    """Every chain-server replica compose is running, by published port.

    Scaled replicas get ephemeral host ports, so they cannot be configured in
    advance and there is nothing to keep in step by hand. Asking compose is the
    only answer that stays right as replicas come and go.

        docker compose -f docker-compose.yaml -f docker-compose.scale.yaml \\
            up -d --scale chain-server=4

    Read as JSON rather than a Go template: the template form of Publishers has
    changed shape between Compose versions, and parsing it is how this silently
    finds nothing.
    """

    try:
        out = subprocess.run(
            ["docker", "compose", "ps", "--format", "json", "chain-server"],
            capture_output=True, text=True, timeout=30,
        ).stdout
    except Exception:
        return []

    found: set[str] = set()
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries = json.loads(line)
        except json.JSONDecodeError:
            continue
        for entry in entries if isinstance(entries, list) else [entries]:
            for published in entry.get("Publishers") or []:
                # Only this service's own port, and only a mapping that a
                # client can actually reach -- an unpublished container has a
                # PublishedPort of 0.
                if published.get("TargetPort") != 8009:
                    continue
                port = published.get("PublishedPort")
                if port:
                    found.add(f"http://localhost:{port}")
    return sorted(found)


def chain_server_containers() -> list[str]:
    """Container ids for CPU sampling, however many replicas there are."""

    try:
        out = subprocess.run(
            ["docker", "compose", "ps", "-q", "chain-server"],
            capture_output=True, text=True, timeout=30,
        ).stdout
    except Exception:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


@dataclass
class Level:
    concurrency: int
    seconds: float = 0.0
    turns: list[float] = field(default_factory=list)
    errors: int = 0
    first_error: str | None = None
    rate_limited: int = 0
    model_calls: int = 0
    cpu_samples: list[float] = field(default_factory=list)
    mem_mib: float = 0.0

    @property
    def turns_per_minute(self) -> float:
        return len(self.turns) / self.seconds * 60 if self.seconds else 0.0

    def pct(self, p: float) -> float:
        if not self.turns:
            return 0.0
        ordered = sorted(self.turns)
        return ordered[min(int(len(ordered) * p), len(ordered) - 1)]

    @property
    def peak_cpu(self) -> float:
        return max(self.cpu_samples) if self.cpu_samples else 0.0

    @property
    def calls_per_second(self) -> float:
        """The rate the model actually sees, and the only saturation signal.

        Turns per minute is the wrong unit for this question: one shopper turn
        is an agent loop of several model calls, so two deployments with equal
        turn rates can put very different loads on the model. Saturation shows
        up here first -- the call rate stops rising while concurrency does.
        """

        return self.model_calls / self.seconds if self.seconds else 0.0

    @property
    def calls_per_turn(self) -> float:
        return self.model_calls / len(self.turns) if self.turns else 0.0


def _chain_server_log(since_seconds: int) -> str | None:
    try:
        return subprocess.run(
            ["docker", "compose", "logs", "--since", f"{since_seconds}s", "chain-server"],
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout
    except Exception:
        return None


def _count_model_calls(log: str | None) -> int:
    """Requests that actually reached the model, refusals included.

    Counted from the chain server rather than guessed from turn counts,
    because how many calls a turn makes depends on how many tools the agent
    decides to use -- it is a property of the conversation, not a constant.
    """

    return -1 if log is None else log.count("/v1/chat/completions")


def _count_rate_limits(since_seconds: int) -> int:
    """How many times the model refused during the window.

    Read from the chain server's log rather than inferred from a failed turn:
    a 429 inside the agent loop is retried and may never surface as an error,
    so a run can be rate limited throughout and still look merely slow.
    """

    log = _chain_server_log(since_seconds)
    return -1 if log is None else log.count("429")


class _CpuSampler(threading.Thread):
    """Chain-server CPU and memory while the level runs.

    The point is to separate "the app is working hard" from "the app is
    waiting". A turn is mostly waiting on the model, so a level that is slow
    with the CPU near idle is not a level that more pods will fix.
    """

    def __init__(self, containers: list[str], interval: float = 1.0) -> None:
        super().__init__(daemon=True)
        self.containers = containers
        self.interval = interval
        self.cpu: list[float] = []
        self.mem: list[float] = []
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["docker", "stats", "--no-stream", "--format",
                     "{{.CPUPerc}}|{{.MemUsage}}", *self.containers],
                    capture_output=True, text=True, timeout=30,
                ).stdout.strip()
                # Summed, not averaged. Four replicas at 90% each is four
                # saturated cores, and an average would report 90% and hide
                # that three more pods are already busy.
                total_cpu = total_mem = 0.0
                for line in out.splitlines():
                    cpu, mem = line.split("|")
                    total_cpu += float(cpu.rstrip("%"))
                    total_mem += float(mem.split("MiB")[0].strip())
                self.cpu.append(total_cpu)
                self.mem.append(total_mem)
            except Exception:
                pass
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()


def _play(assistant, scenario, identity, latencies, lock, deadline) -> str | None:
    """One conversation, turn by turn, stopping at the level's deadline."""

    for step in scenario.get("turns") or []:
        if time.monotonic() > deadline:
            return None
        attachment = None
        if step.get("attach"):
            attachment = SCRIPTS_ROOT / "assets" / str(step["attach"])
        started = time.monotonic()
        try:
            assistant.say(identity, str(step["say"]), attachment)
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"[:140]
        with lock:
            latencies.append(time.monotonic() - started)
    return None


def run_level(assistants, scenarios, concurrency: int, minutes: float) -> Level:
    level = Level(concurrency=concurrency)
    lock = threading.Lock()
    deadline = time.monotonic() + minutes * 60
    counter = itertools.count()
    sampler = _CpuSampler(chain_server_containers())
    sampler.start()
    started_at = time.monotonic()

    def worker(slot: int) -> None:
        # Round-robin across replicas, which is what a Service in front of them
        # would do. Pinning a worker to one replica would measure that replica.
        assistant = assistants[slot % len(assistants)]
        # A conversation each, so turns never contend on one conversation --
        # the service refuses a second in-flight turn for the same one, which
        # would measure that rule rather than capacity.
        while time.monotonic() < deadline:
            nonce = next(counter)
            scenario = scenarios[nonce % len(scenarios)]
            user = LOAD_USER_BASE + concurrency * 1000 + slot
            marker = f"load-{concurrency}-{slot}-{nonce}"
            identity = {
                "user_id": user,
                "session_id": marker,
                "conversation_id": marker,
                "cart_id": marker,
            }
            error = _play(assistant, scenario, identity, level.turns, lock, deadline)
            if error:
                with lock:
                    level.errors += 1
                    if level.first_error is None:
                        level.first_error = error

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(concurrency)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    sampler.stop()
    level.seconds = time.monotonic() - started_at
    level.cpu_samples = sampler.cpu
    level.mem_mib = max(sampler.mem) if sampler.mem else 0.0
    window = int(level.seconds) + 5
    log = _chain_server_log(window)
    level.rate_limited = -1 if log is None else log.count("429")
    level.model_calls = _count_model_calls(log)
    return level


def diagnose(level: Level, previous: Level | None) -> str:
    """Name the wall. This is the output that decides what to do next."""

    if level.rate_limited > 0:
        return (
            f"MODEL RATE LIMIT ({level.rate_limited} refusals) -- more pods will "
            "not help, and will make it worse. Raise quota or move endpoint."
        )
    if level.errors and level.errors > len(level.turns) * 0.1:
        return f"ERRORS ({level.errors}) -- {level.first_error}"
    cores = os.cpu_count() or 1
    if level.peak_cpu > cores * 85:
        # Every pod and the load generator share this machine, so past this
        # point the numbers describe the host, not the service. More pods here
        # make it worse: they contend for the cores the existing ones need.
        return (
            f"HOST CPU EXHAUSTED -- {level.peak_cpu:.0f}% used of "
            f"{cores * 100}% available. This machine cannot drive more load; "
            "the result says nothing about the service or the model."
        )
    if level.peak_cpu > 85:
        return "CHAIN-SERVER CPU -- add chain-server pods. This is the good case."

    if previous is not None and previous.calls_per_second > 0:
        # Concurrency doubled; if the model's call rate did not follow, the
        # queue has moved to the model and more callers only lengthen it.
        gain = level.calls_per_second / previous.calls_per_second
        widened = level.concurrency / previous.concurrency
        if widened > 1.2 and gain < 1.2 and level.peak_cpu < 60:
            return (
                f"MODEL SATURATED -- {widened:.0f}x the callers bought "
                f"{gain:.2f}x the call rate, app at {level.peak_cpu:.0f}% CPU. "
                "This is the saturation point."
            )
    return "headroom -- nothing is saturated at this level"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", default="1,2,4,8")
    parser.add_argument(
        "--find-saturation",
        action="store_true",
        help="double the concurrency until something gives, then stop",
    )
    parser.add_argument(
        "--minutes",
        type=float,
        default=2.0,
        help="how long to hold each level",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="scenarios to drive, e.g. J12,J17 (default: every journey)",
    )
    parser.add_argument(
        "--targets",
        default=None,
        help="chain-server base urls, comma separated. Discovered from compose "
             "when omitted, so scaled replicas need no configuration.",
    )
    args = parser.parse_args()

    config = load_eval_config()
    targets = (
        [t.strip() for t in args.targets.split(",") if t.strip()]
        if args.targets
        else discover_chain_servers() or [config.target_agent.base_url]
    )
    assistants = []
    for target in targets:
        replica = Assistant(config)
        replica._url = f"{target.rstrip('/')}/query/stream"
        assistants.append(replica)
    scenarios = [
        s for s in load_scenarios(args.only)
        if args.only or s["id"].startswith("J")
    ]
    if not scenarios:
        raise SystemExit("no scenarios selected")

    levels = (
        [2 ** n for n in range(0, 9)]
        if args.find_saturation
        else [int(v) for v in args.levels.split(",") if v.strip()]
    )
    turns_total = sum(len(s.get("turns") or []) for s in scenarios)
    print(
        f"driving {len(scenarios)} journeys ({turns_total} turns) at "
        f"{','.join(map(str, levels))} concurrent, {args.minutes:g} min each"
    )
    print(f"across {len(assistants)} chain-server(s): {', '.join(targets)}\n")
    print(
        f"{'conc':>4s} {'turns':>6s} {'turns/min':>10s} {'calls/s':>8s} "
        f"{'p95 s':>7s} {'cpu%':>6s} {'429':>5s}  limit"
    )

    previous: Level | None = None
    results: list[Level] = []
    for concurrency in levels:
        level = run_level(assistants, scenarios, concurrency, args.minutes)
        verdict = diagnose(level, previous)
        results.append(level)
        print(
            f"{level.concurrency:4d} {len(level.turns):6d} "
            f"{level.turns_per_minute:10.1f} {level.calls_per_second:8.1f} "
            f"{level.pct(0.95):7.1f} {level.peak_cpu:6.0f} "
            f"{level.rate_limited:5d}  {verdict}"
        )
        previous = level
        if args.find_saturation and not verdict.startswith("headroom"):
            print(f"\n  stopping: the wall is at concurrency {concurrency}")
            break

    print()
    calls_per_turn = [r.calls_per_turn for r in results if r.calls_per_turn]
    if calls_per_turn:
        average = sum(calls_per_turn) / len(calls_per_turn)
        print(
            f"one shopper turn costs {average:.1f} model calls, so a journey of "
            f"{len(scenarios) and int(turns_total / len(scenarios))} turns costs "
            f"about {average * turns_total / max(len(scenarios), 1):.0f}."
        )
    best = max(results, key=lambda r: r.turns_per_minute)
    print(f"peak sustained: {best.turns_per_minute:.1f} turns/min at "
          f"concurrency {best.concurrency}")

    # How many pods, for the case where the answer is "add pods". Only the CPU
    # a turn actually costs can answer it, which is why CPU is sampled at all.
    if best.peak_cpu > 0 and best.concurrency:
        cpu_per_conversation = best.peak_cpu / best.concurrency
        per_pod = int(85 / cpu_per_conversation) if cpu_per_conversation else 0
        print(
            f"{len(assistants)} chain-server(s) used {cpu_per_conversation:.1f}% "
            f"of a core per concurrent conversation, so one pod carries roughly "
            f"{per_pod} before its core is the limit."
        )
        cores = os.cpu_count() or 1
        if best.peak_cpu > cores * 85:
            print(
                f"That figure is not usable: {best.peak_cpu:.0f}% of "
                f"{cores * 100}% available was in use, so the pods were "
                "competing with each other and with this load generator. "
                "Measure pods-per-conversation on a machine with cores to "
                "spare, or drive the load from a different one."
            )
        elif per_pod:
            print(
                f"to drive N concurrent journeys, run about N/{per_pod} "
                "chain-server pods. Below that the app is the ceiling and the "
                "model is never asked for its answer."
            )
    if any(r.rate_limited > 0 for r in results):
        print(
            "\nA level was rate limited, so the numbers above are a floor, not "
            "a capacity: the system was never allowed to work as hard as it "
            "could. Re-run against an endpoint that will take the load."
        )


if __name__ == "__main__":
    main()
