#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Find where the memory service stops keeping up.

Every ceiling in the scaling plan was read off the code and none has ever been
measured. This is the measurement: concurrent cart reads and writes at rising
concurrency, against one pod, recording throughput and latency at each level.

The point is the *shape*, not the absolute numbers. Twelve `async def` endpoints
doing blocking database work make the event loop a lock over the whole service,
so the prediction is that throughput is flat in concurrency -- adding clients
adds queueing and nothing else. If that is what the graph shows, making the
endpoints threadpool-bound is worth doing and the graph after will show it. If
throughput actually rises with concurrency, the reading was wrong and the step
is not worth a day.

Reads and writes are reported separately because they have different ceilings:
SQLite serialises writers database-wide, but reads should be concurrent and are
not, and that difference is the whole argument for the change.

    python scripts/memory_load.py --url http://localhost:8011
    python scripts/memory_load.py --levels 1,2,4,8,16 --seconds 6

Cart rows it creates are removed afterwards. It uses a user_id range of its own
so a failed run cannot disturb a real shopper's cart.
"""

from __future__ import annotations

import argparse
import itertools
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import requests

#: Real shopper ids here run from 997000 to over 10^18, so no constant is
#: provably outside them. `_claim_slots` checks instead of assuming: cleanup
#: empties these carts, and emptying a real shopper's cart is not a thing to be
#: lucky about.
LOAD_USER_BASE = 900_000_000


@dataclass
class Result:
    label: str
    concurrency: int
    completed: int
    errors: int
    seconds: float
    latencies_ms: list[float]
    #: Errors are never summarised away -- an unexplained collapse is the most
    #: informative thing a load run can produce.
    first_error: str | None = None

    @property
    def throughput(self) -> float:
        return self.completed / self.seconds if self.seconds else 0.0

    def pct(self, p: float) -> float:
        if not self.latencies_ms:
            return 0.0
        ordered = sorted(self.latencies_ms)
        index = min(int(len(ordered) * p), len(ordered) - 1)
        return ordered[index]


def _read(session: requests.Session, url: str, user_id: int) -> float:
    start = time.perf_counter()
    response = session.get(f"{url}/user/{user_id}/cart", timeout=30)
    response.raise_for_status()
    return (time.perf_counter() - start) * 1000


def _write(
    session: requests.Session,
    url: str,
    user_id: int,
    counter: itertools.count,
) -> float:
    nonce = next(counter)
    start = time.perf_counter()
    response = session.post(
        f"{url}/user/{user_id}/cart/add",
        json={
            "product_id": f"load_{nonce}",
            "item": "Load Test Item",
            "amount": 1,
            "price": 1.0,
            "idempotency_key": f"load-{user_id}-{nonce}",
        },
        timeout=30,
    )
    response.raise_for_status()
    return (time.perf_counter() - start) * 1000


def _drive(url: str, label: str, concurrency: int, seconds: float) -> Result:
    counter = itertools.count()
    deadline = time.monotonic() + seconds
    latencies: list[float] = []
    errors = 0
    first_error: str | None = None

    def worker(slot: int) -> None:
        nonlocal errors, first_error
        session = requests.Session()
        user_id = LOAD_USER_BASE + slot
        while time.monotonic() < deadline:
            try:
                if label == "read":
                    latencies.append(_read(session, url, user_id))
                else:
                    latencies.append(_write(session, url, user_id, counter))
            except Exception as exc:
                errors += 1
                if first_error is None:
                    first_error = f"{type(exc).__name__}: {exc}"[:160]

    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(worker, range(concurrency)))
    elapsed = time.monotonic() - start
    return Result(
        label, concurrency, len(latencies), errors, elapsed, latencies, first_error
    )


def _cleanup(url: str, slots: int) -> int:
    """Empty every cart this run created, so a real cart is never touched."""

    emptied = 0
    session = requests.Session()
    for slot in range(slots):
        user_id = LOAD_USER_BASE + slot
        try:
            response = session.post(f"{url}/user/{user_id}/cart/clear", timeout=30)
        except Exception:
            continue
        # 404 means the cart was already empty, which is the desired end state.
        if response.status_code == 200:
            emptied += 1
    return emptied


def _claim_slots(url: str, slots: int) -> None:
    """Refuse to run if any user id this run would use already has a cart."""

    session = requests.Session()
    occupied = []
    for slot in range(slots):
        user_id = LOAD_USER_BASE + slot
        cart = session.get(f"{url}/user/{user_id}/cart", timeout=30).json()
        if cart.get("cart"):
            occupied.append(user_id)
    if occupied:
        raise SystemExit(
            "refusing to run: these ids already hold cart lines, and this run "
            f"would delete them -- {occupied}. Change LOAD_USER_BASE."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8011")
    parser.add_argument("--levels", default="1,2,4,8,16")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    levels = [int(value) for value in args.levels.split(",") if value.strip()]
    requests.get(f"{args.url}/health", timeout=10).raise_for_status()
    _claim_slots(args.url, max(levels))

    print(f"memory service load — {args.label or args.url}")
    print(f"{'op':6s} {'conc':>5s} {'req/s':>9s} {'p50 ms':>9s} {'p95 ms':>9s} {'err':>5s}")
    results: list[Result] = []
    try:
        for label in ("read", "write"):
            for concurrency in levels:
                result = _drive(args.url, label, concurrency, args.seconds)
                results.append(result)
                print(
                    f"{label:6s} {concurrency:5d} {result.throughput:9.1f} "
                    f"{result.pct(0.50):9.1f} {result.pct(0.95):9.1f} "
                    f"{result.errors:5d}"
                    + (f"  {result.first_error}" if result.first_error else "")
                )
    finally:
        removed = _cleanup(args.url, max(levels))
        print(f"\ncleanup: emptied {removed} load cart(s)")

    # The number the argument turns on: does throughput rise with concurrency?
    for label in ("read", "write"):
        series = [r for r in results if r.label == label]
        if len(series) < 2:
            continue
        first, last = series[0].throughput, series[-1].throughput
        ratio = last / first if first else 0.0
        print(
            f"{label:6s} throughput at concurrency {series[-1].concurrency} is "
            f"{ratio:.2f}x concurrency 1 "
            f"({'scales' if ratio > 1.5 else 'FLAT — serialised'})"
        )


if __name__ == "__main__":
    main()
