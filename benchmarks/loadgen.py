#!/usr/bin/env python3
"""
Concurrent load generator, so the dashboard panels have data to show.

An idle dashboard looks exactly like a broken one, so this exists mainly to
prove the metrics pipeline works end to end.

    ./bench.sh load                # 8 concurrent, 32 requests
    ./bench.sh load 16 64          # 16 concurrent, 64 requests
"""

import json
import os
import random
import sys
import threading
import time
import urllib.request

BASE = os.environ.get("VLLM_BASE", "http://127.0.0.1:8000")


def resolve_model():
    """Ask the server what it serves rather than assuming a name.

    A hardcoded default silently turns every request into a 404 the moment the
    endpoint is replaced -- which is exactly what happened when the hand-rolled
    vLLM server ('nemotron-3-super') gave way to the NIM
    ('nvidia/nemotron-3-super-120b-a12b'). SERVED_NAME still overrides for the
    case of several models behind one endpoint.
    """
    override = os.environ.get("SERVED_NAME")
    if override:
        return override
    with urllib.request.urlopen(f"{BASE}/v1/models", timeout=10) as r:
        data = json.loads(r.read())["data"]
    if not data:
        sys.exit(f"{BASE} reports no served models")
    return data[0]["id"]


MODEL = resolve_model()

# Mixed shapes so the latency histograms and the prompt/generation length
# heatmaps get a spread of values rather than one spike.
PROMPTS = [
    ("Explain in two sentences what expert parallelism means.", 200),
    ("List five uses for a state-space model.", 400),
    ("Write a haiku about tensor cores.", 120),
    ("Summarise the tradeoff between FP8 and FP4 inference.", 500),
    ("What is 17 * 23? Show your working.", 300),
    ("Name three advantages of Mixture-of-Experts architectures.", 400),
    ("Describe KV cache paging in one paragraph.", 600),
    ("Give a one-line definition of speculative decoding.", 150),
]

lock = threading.Lock()
stats = {"ok": 0, "fail": 0, "prompt_tokens": 0, "completion_tokens": 0,
         "latencies": []}


def worker(n_requests):
    for _ in range(n_requests):
        prompt, max_tokens = random.choice(PROMPTS)
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }
        req = urllib.request.Request(
            f"{BASE}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                d = json.loads(r.read())
            elapsed = time.time() - t0
            with lock:
                stats["ok"] += 1
                stats["prompt_tokens"] += d["usage"]["prompt_tokens"]
                stats["completion_tokens"] += d["usage"]["completion_tokens"]
                stats["latencies"].append(elapsed)
        except Exception as exc:
            with lock:
                stats["fail"] += 1
            print(f"  request failed: {type(exc).__name__}: {exc}")


def main():
    concurrency = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    total = int(sys.argv[2]) if len(sys.argv) > 2 else 32

    per_worker = max(1, total // concurrency)
    actual = per_worker * concurrency
    print(f"  {actual} requests, {concurrency} concurrent, "
          f"{per_worker} per worker")

    t0 = time.time()
    threads = [threading.Thread(target=worker, args=(per_worker,))
               for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0

    lat = sorted(stats["latencies"])
    print(f"\n  completed {stats['ok']} ok, {stats['fail']} failed "
          f"in {wall:.1f}s")
    if lat:
        def pct(p):
            return lat[min(len(lat) - 1, int(p * len(lat)))]
        print(f"  e2e latency   mean {sum(lat) / len(lat):.2f}s   "
              f"p50 {pct(0.5):.2f}s   p90 {pct(0.9):.2f}s   max {lat[-1]:.2f}s")
    print(f"  tokens        {stats['prompt_tokens']:,} prompt, "
          f"{stats['completion_tokens']:,} generated")
    if wall > 0:
        print(f"  throughput    {stats['completion_tokens'] / wall:.1f} "
              f"generated tok/s aggregate, "
              f"{stats['ok'] / wall:.2f} req/s")


if __name__ == "__main__":
    main()
