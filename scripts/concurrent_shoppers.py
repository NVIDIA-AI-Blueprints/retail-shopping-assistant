#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Do concurrent shoppers get served concurrently, or do they queue?

The question this answers is not how fast one turn is. It is whether the Nth
shopper waits for the first N-1. One turn takes tens of seconds and is almost
entirely time spent waiting on the model, so if the service is doing that
waiting properly, N shoppers at once should finish in about the time of one. If
instead the wall clock grows with N, something on the turn path is holding the
event loop and every shopper is paying for every other shopper.

    python scripts/concurrent_shoppers.py --shoppers 3

Each shopper gets its own user, session and cart, so nothing is shared and any
queueing observed is the service's, not the scenario's.
"""

from __future__ import annotations

import argparse
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import requests


def one_turn(url: str, query: str, index: int, timeout: float) -> dict:
    """Drive one shopper's turn to completion and time it end to end."""

    marker = uuid.uuid4().hex[:8]
    started = time.monotonic()
    try:
        response = requests.post(
            f"{url}/query/stream",
            json={
                # A distinct shopper each time: shared identity would make two
                # turns contend on the same conversation row, which is a
                # different question from the one being asked here.
                "user_id": 700_000_000 + index,
                "query": query,
                "session_id": f"conc-{marker}",
                "conversation_id": f"conc-{marker}",
                "cart_id": f"conc-{marker}",
            },
            stream=True,
            timeout=timeout,
        )
        response.raise_for_status()
        first_byte = None
        chunks = 0
        for line in response.iter_lines():
            if not line:
                continue
            if first_byte is None:
                first_byte = time.monotonic() - started
            chunks += 1
        return {
            "index": index,
            "seconds": time.monotonic() - started,
            "first_byte": first_byte,
            "chunks": chunks,
            "error": None,
        }
    except Exception as exc:
        return {
            "index": index,
            "seconds": time.monotonic() - started,
            "first_byte": None,
            "chunks": 0,
            "error": f"{type(exc).__name__}: {exc}"[:120],
        }


def run(url: str, query: str, shoppers: int, timeout: float) -> list[dict]:
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=shoppers) as pool:
        results = list(
            pool.map(lambda i: one_turn(url, query, i, timeout), range(shoppers))
        )
    for result in results:
        result["wall"] = time.monotonic() - started
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8009")
    parser.add_argument("--shoppers", type=int, default=3)
    parser.add_argument("--query", default="show me red dresses")
    parser.add_argument("--timeout", type=float, default=240.0)
    args = parser.parse_args()

    print("one shopper alone, to establish what a turn costs")
    alone = run(args.url, args.query, 1, args.timeout)[0]
    print(
        f"  turn {alone['seconds']:.1f}s  first byte "
        f"{(alone['first_byte'] or 0):.1f}s  {alone['chunks']} chunks"
        + (f"  ERROR {alone['error']}" if alone["error"] else "")
    )

    print(f"\n{args.shoppers} shoppers at once")
    together = run(args.url, args.query, args.shoppers, args.timeout)
    for result in sorted(together, key=lambda r: r["index"]):
        print(
            f"  shopper {result['index']}: turn {result['seconds']:.1f}s  "
            f"first byte {(result['first_byte'] or 0):.1f}s  "
            f"{result['chunks']} chunks"
            + (f"  ERROR {result['error']}" if result["error"] else "")
        )

    wall = max(r["wall"] for r in together)
    solo = alone["seconds"]
    print(f"\n  one turn alone      {solo:.1f}s")
    print(f"  {args.shoppers} turns together   {wall:.1f}s wall clock")
    print(f"  if fully serialised {solo * args.shoppers:.1f}s would be expected")
    print(f"  if fully concurrent {solo:.1f}s would be expected")
    # Turn latency varies a lot between runs, so this is a shape, not a score.
    ratio = wall / solo if solo else 0
    print(
        f"\n  observed {ratio:.2f}x one turn for {args.shoppers}x the work "
        + ("-- serialised" if ratio > args.shoppers * 0.7 else "-- overlapping")
    )
    failures = [r for r in together if r["error"]]
    if failures:
        print(f"  {len(failures)} of {args.shoppers} failed")


if __name__ == "__main__":
    main()
