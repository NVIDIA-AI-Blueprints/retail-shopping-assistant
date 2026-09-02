#!/usr/bin/env python3
"""
Live terminal metrics view for a vLLM server. No Prometheus, no browser.

Reads the server's own /metrics endpoint and derives rates between samples the
same way Prometheus would. Every metric shown below is one vLLM actually
exposes -- verified against a live vLLM 0.17.1 /metrics scrape -- and the
histogram averages are computed as documented, delta(_sum) / delta(_count).

    ./bench.sh top              # refresh every 2s
    ./bench.sh top 5            # every 5s
"""

import os
import sys
import time
import urllib.request
from collections import defaultdict

BASE = os.environ.get("VLLM_BASE", "http://127.0.0.1:8000")

BOLD, DIM, GRN, YLW, RED, CYA, RST = (
    "\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[31m", "\033[36m", "\033[0m"
)


def scrape():
    """Parse the Prometheus text exposition format into usable structures."""
    with urllib.request.urlopen(f"{BASE}/metrics", timeout=10) as r:
        text = r.read().decode()

    plain = {}
    buckets = defaultdict(dict)
    labelled = defaultdict(float)

    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        try:
            key, val = line.rsplit(" ", 1)
            val = float(val)
        except ValueError:
            continue

        if "{" in key:
            name, lbl = key.split("{", 1)
            lbl = lbl.rstrip("}")
        else:
            name, lbl = key, ""

        if name.endswith("_bucket"):
            le = None
            for part in lbl.split(","):
                if part.strip().startswith("le="):
                    le = part.split("=", 1)[1].strip('"')
            if le is not None:
                buckets[name[: -len("_bucket")]][float(le)] = val
        else:
            plain[name] = plain.get(name, 0.0) + val
            if "finished_reason=" in lbl:
                for part in lbl.split(","):
                    if part.strip().startswith("finished_reason="):
                        labelled[part.split("=", 1)[1].strip('"')] += val

    return plain, buckets, labelled


def rate(cur, prev, dt):
    if prev is None or dt <= 0:
        return 0.0
    return max(0.0, (cur - prev)) / dt


def hist_avg(name, plain, prev_plain):
    """Interval mean of a histogram: delta(_sum) / delta(_count)."""
    s, c = plain.get(f"{name}_sum", 0.0), plain.get(f"{name}_count", 0.0)
    if prev_plain is None:
        return None
    ds = s - prev_plain.get(f"{name}_sum", 0.0)
    dc = c - prev_plain.get(f"{name}_count", 0.0)
    return (ds / dc) if dc > 0 else None


def hist_quantile(name, q, buckets, prev_buckets):
    """Quantile over the interval, interpolating within the matched bucket."""
    cur = buckets.get(name, {})
    if not cur:
        return None
    prev = (prev_buckets or {}).get(name, {})
    deltas = sorted((le, cur[le] - prev.get(le, 0.0)) for le in cur)
    if not deltas:
        return None
    total = deltas[-1][1]
    if total <= 0:
        return None
    target = q * total
    lower_le, lower_count = 0.0, 0.0
    for le, count in deltas:
        if count >= target:
            if le == float("inf"):
                return lower_le
            if count == lower_count:
                return le
            frac = (target - lower_count) / (count - lower_count)
            return lower_le + frac * (le - lower_le)
        lower_le, lower_count = le, count
    return deltas[-1][0]


def fmt_s(v):
    if v is None:
        return f"{DIM}--{RST}"
    if v < 1e-3:
        return f"{v * 1e6:.0f}us"
    if v < 1:
        return f"{v * 1e3:.1f}ms"
    return f"{v:.2f}s"


def bar(frac, width=28):
    frac = max(0.0, min(1.0, frac))
    filled = int(frac * width)
    color = GRN if frac < 0.7 else (YLW if frac < 0.9 else RED)
    return f"{color}{'#' * filled}{DIM}{'.' * (width - filled)}{RST}"


def row(label, value, note=""):
    print(f"  {label:<26} {value:<22} {DIM}{note}{RST}")


def main():
    interval = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
    prev_plain = prev_buckets = None
    prev_t = None
    print(f"{DIM}polling {BASE}/metrics every {interval}s - Ctrl-C to exit{RST}")

    while True:
        try:
            plain, buckets, reasons = scrape()
        except Exception as exc:
            print(f"{RED}scrape failed:{RST} {exc}")
            time.sleep(interval)
            continue

        now = time.time()
        dt = (now - prev_t) if prev_t else 0.0

        gen = rate(plain.get("vllm:generation_tokens_total", 0.0),
                   (prev_plain or {}).get("vllm:generation_tokens_total"), dt)
        pro = rate(plain.get("vllm:prompt_tokens_total", 0.0),
                   (prev_plain or {}).get("vllm:prompt_tokens_total"), dt)
        fin = rate(plain.get("vllm:request_success_total", 0.0),
                   (prev_plain or {}).get("vllm:request_success_total"), dt)
        pre = rate(plain.get("vllm:num_preemptions_total", 0.0),
                   (prev_plain or {}).get("vllm:num_preemptions_total"), dt)

        running = plain.get("vllm:num_requests_running", 0.0)
        waiting = plain.get("vllm:num_requests_waiting", 0.0)
        kv = plain.get("vllm:kv_cache_usage_perc", 0.0)

        print("\033[2J\033[H", end="")
        print(f"{BOLD}vLLM metrics{RST}  {DIM}{BASE}  "
              f"{time.strftime('%H:%M:%S')}  window {dt:.1f}s{RST}\n")

        print(f"{BOLD}{CYA}Throughput{RST}")
        row("generation tokens/s", f"{gen:8.1f}", "vllm:generation_tokens_total")
        row("prompt tokens/s", f"{pro:8.1f}", "vllm:prompt_tokens_total")
        row("requests finished/s", f"{fin:8.2f}", "vllm:request_success_total")
        it = hist_avg("vllm:iteration_tokens_total", plain, prev_plain)
        row("tokens per engine step", f"{it:8.1f}" if it else f"{DIM}--{RST}",
            "vllm:iteration_tokens_total")

        print(f"\n{BOLD}{CYA}Concurrency{RST}")
        row("requests running", f"{running:8.0f}", "vllm:num_requests_running")
        row("requests waiting", f"{waiting:8.0f}", "vllm:num_requests_waiting")
        row("preemptions/s", f"{pre:8.2f}", "vllm:num_preemptions_total")

        print(f"\n{BOLD}{CYA}KV cache{RST}")
        row("usage", f"{kv * 100:7.2f}%  {bar(kv)}", "vllm:kv_cache_usage_perc")
        pq = plain.get("vllm:prefix_cache_queries_total", 0.0)
        ph = plain.get("vllm:prefix_cache_hits_total", 0.0)
        row("prefix cache hit rate",
            f"{(ph / pq * 100):7.2f}%" if pq > 0 else f"{DIM}n/a (disabled){RST}",
            "vllm:prefix_cache_hits_total / _queries_total")

        print(f"\n{BOLD}{CYA}Latency{RST}  {DIM}interval mean, then p50 / p90 / p99{RST}")
        for label, metric in (
            ("end-to-end", "vllm:e2e_request_latency_seconds"),
            ("time to first token", "vllm:time_to_first_token_seconds"),
            ("inter-token", "vllm:inter_token_latency_seconds"),
            ("time per output token", "vllm:request_time_per_output_token_seconds"),
            ("queue wait", "vllm:request_queue_time_seconds"),
            ("prefill phase", "vllm:request_prefill_time_seconds"),
            ("decode phase", "vllm:request_decode_time_seconds"),
        ):
            avg = hist_avg(metric, plain, prev_plain)
            qs = " / ".join(
                fmt_s(hist_quantile(metric, q, buckets, prev_buckets))
                for q in (0.5, 0.9, 0.99)
            )
            row(label, fmt_s(avg), qs)

        if reasons:
            print(f"\n{BOLD}{CYA}Finish reasons{RST} {DIM}(cumulative){RST}")
            for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
                row(reason, f"{count:8.0f}", "vllm:request_success_total")

        if prev_plain is None:
            print(f"\n{DIM}rates and quantiles appear after the second sample{RST}")

        prev_plain, prev_buckets, prev_t = plain, buckets, now
        time.sleep(interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
