#!/usr/bin/env python3
"""
Profile where time goes inside a single agent turn.

Answers "what is the agent actually waiting on" by measuring one turn from three
vantage points at once and reconciling them:

  app      chain-server's own /query/timing response: per-phase timings, tokens,
           and which model roles were called
  engine   the NIM's Prometheus counters, sampled either side of the turn, which
           give the LLM's true prefill/decode/queue split and KV peak
  spans    Phoenix/OTel, which is the only source with per-LLM-call and
           per-tool-call granularity

No single source is sufficient. The app's `timings` buckets overlap each other
(`deepagents` contains `catalog_search` and the safety checks), so they cannot be
summed. The engine only sees its own requests, so it cannot see retrieval or the
remote embedding hop. Spans see structure but attribute their own overhead.
Reading all three together is what makes the residual - the time that belongs to
no component - visible and trustworthy.

Usage:
    ./turn_profile.py                       # default query set
    ./turn_profile.py --out /tmp/p1.json
    ./turn_profile.py --only greeting,filtered_search
"""

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

CHAIN = os.environ.get("CHAIN_BASE", "http://localhost:8009")
NIM = os.environ.get("NIM_BASE", "http://127.0.0.1:8000")
PHOENIX = os.environ.get("PHOENIX_BASE", "http://localhost:6006")

# Counters read either side of a turn. Deltas over a single turn are exact
# because the profiler runs one turn at a time against an otherwise idle engine.
COUNTERS = [
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:time_to_first_token_seconds_sum",
    "vllm:time_to_first_token_seconds_count",
    "vllm:request_prefill_time_seconds_sum",
    "vllm:request_decode_time_seconds_sum",
    "vllm:request_queue_time_seconds_sum",
    "vllm:request_inference_time_seconds_sum",
    "vllm:inter_token_latency_seconds_sum",
    "vllm:inter_token_latency_seconds_count",
    "vllm:num_preemptions_total",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
]

# Independent single-turn probes. Each runs in its own session so that a growing
# history does not silently inflate the next one. `greeting` is the control: it
# should need no catalog search, so whatever it costs is the fixed price of a
# turn before any retrieval or tool work happens.
PROBES = [
    ("greeting", "Hello"),
    ("filtered_search", "Show me black dresses under $100"),
    ("simple_search", "I need a red handbag"),
    ("broad_search", "What do you have in shoes?"),
    ("cart_read", "What is in my cart?"),
]

# One multi-turn session, kept in order, to expose how prompt size and cost grow
# with conversation depth. Single-turn probes cannot show this.
CONVERSATION = [
    ("convo_t1", "Show me summer dresses"),
    ("convo_t2", "Only the ones under $80"),
    ("convo_t3", "Add the first one to my cart"),
]


def _get_json(url, timeout=15):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def read_counters():
    """Scrape the NIM's metrics, summing across label sets per metric name."""
    try:
        with urllib.request.urlopen(f"{NIM}/metrics", timeout=15) as r:
            text = r.read().decode()
    except Exception:
        return {}
    out = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        try:
            key, val = line.rsplit(" ", 1)
            name = key.split("{", 1)[0]
        except ValueError:
            continue
        if name in COUNTERS or name == "vllm:kv_cache_usage_perc":
            out[name] = out.get(name, 0.0) + float(val)
    return out


class KVSampler(threading.Thread):
    """Track peak KV usage and in-flight requests while a turn runs.

    kv_cache_usage_perc is a gauge, so it reads zero at idle and is only
    meaningful sampled during the request.
    """

    def __init__(self, interval=0.25):
        super().__init__(daemon=True)
        self.interval = interval
        self.stop_flag = threading.Event()
        self.peak_kv = 0.0
        self.peak_running = 0.0

    def run(self):
        while not self.stop_flag.is_set():
            m = read_counters()
            self.peak_kv = max(self.peak_kv, m.get("vllm:kv_cache_usage_perc", 0.0))
            try:
                with urllib.request.urlopen(f"{NIM}/metrics", timeout=10) as r:
                    for line in r.read().decode().splitlines():
                        if line.startswith("vllm:num_requests_running"):
                            self.peak_running = max(
                                self.peak_running, float(line.rsplit(" ", 1)[1])
                            )
            except Exception:
                pass
            time.sleep(self.interval)


def run_turn(label, query, session_id, user_id=1):
    """Run one turn and reconcile app, engine, and wall-clock views of it."""
    before = read_counters()
    sampler = KVSampler()
    sampler.start()

    payload = json.dumps(
        {"user_id": user_id, "query": query, "session_id": session_id}
    ).encode()
    req = urllib.request.Request(
        f"{CHAIN}/query/timing",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.monotonic()
    error = None
    body = {}
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            body = json.loads(r.read().decode())
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    wall = time.monotonic() - t0

    sampler.stop_flag.set()
    sampler.join(timeout=2)
    after = read_counters()

    engine = {}
    for name in COUNTERS:
        if name in before and name in after:
            engine[name.replace("vllm:", "")] = round(after[name] - before[name], 6)

    timings = body.get("timings", {}) or {}
    # chain-server sets timings["total"] from its own monotonic clock around the
    # runtime call, so it is real wall time and safe to use, unlike the streaming
    # endpoint's total_seconds which sums overlapping buckets.
    app_total = timings.get("total")

    prefill = engine.get("request_prefill_time_seconds_sum", 0.0)
    decode = engine.get("request_decode_time_seconds_sum", 0.0)
    ttft = engine.get("time_to_first_token_seconds_sum", 0.0)
    calls = engine.get("time_to_first_token_seconds_count", 0.0)

    return {
        "label": label,
        "query": query,
        "session_id": session_id,
        "error": error,
        "wall_seconds": round(wall, 3),
        "app_total_seconds": round(app_total, 3) if app_total is not None else None,
        "app_timings": {k: round(v, 3) for k, v in timings.items()
                        if isinstance(v, (int, float))},
        "token_usage": body.get("token_usage", {}),
        "model_usage": body.get("model_usage", {}),
        "tool_calls": [
            t.get("tool_name")
            for t in (body.get("agent_diagnostics", {}) or {}).get("tool_calls", [])
            if isinstance(t, dict)
        ],
        "engine": engine,
        "engine_llm_calls": calls,
        "engine_compute_seconds": round(prefill + decode, 3),
        "engine_lifecycle_seconds": round(ttft + decode, 3),
        # What no component claims. This is the number Phase 1 exists to find.
        "non_llm_seconds": round(wall - (ttft + decode), 3),
        "peak_kv_usage_perc": round(sampler.peak_kv * 100, 2),
        "peak_requests_running": sampler.peak_running,
        "response_chars": len(body.get("response", "") or ""),
    }


def fetch_spans(session_ids):
    """Pull per-LLM-call and per-tool-call spans for the sessions just run."""
    wanted = set(session_ids)
    collected = []
    cursor = None
    for _ in range(40):  # bounded paging; the profiler only just created these
        url = f"{PHOENIX}/v1/projects/default/spans?limit=200"
        if cursor:
            url += f"&cursor={cursor}"
        try:
            page = _get_json(url, timeout=20)
        except Exception:
            break
        for span in page.get("data", []):
            attrs = span.get("attributes", {}) or {}
            sid = attrs.get("session.id") or attrs.get("session_id")
            if sid in wanted:
                collected.append({
                    "session_id": sid,
                    "name": span.get("name"),
                    "kind": span.get("span_kind"),
                    "parent_id": span.get("parent_id"),
                    "span_id": (span.get("context") or {}).get("span_id"),
                    "start": span.get("start_time"),
                    "end": span.get("end_time"),
                })
        cursor = page.get("next_cursor")
        if not cursor:
            break
    return collected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/turn_profile.json")
    ap.add_argument("--only", default="", help="comma-separated probe labels")
    ap.add_argument("--skip-conversation", action="store_true")
    args = ap.parse_args()

    try:
        urllib.request.urlopen(f"{CHAIN}/ready", timeout=10).read()
    except Exception as exc:
        sys.exit(f"chain-server not ready at {CHAIN}: {exc}")

    wanted = {s for s in args.only.split(",") if s}
    stamp = int(time.time())
    plan = [(lbl, q, f"prof-{stamp}-{lbl}") for lbl, q in PROBES
            if not wanted or lbl in wanted]
    if not args.skip_conversation and not wanted:
        convo_session = f"prof-{stamp}-convo"
        plan += [(lbl, q, convo_session) for lbl, q in CONVERSATION]

    if not plan:
        sys.exit("no probes selected")

    print(f"profiling {len(plan)} turns against {CHAIN} / {NIM}\n")
    hdr = f"{'probe':<18} {'wall':>7} {'engine':>7} {'non-LLM':>8} {'calls':>6} {'in tok':>8} {'peak KV':>8}"
    print(hdr)
    print("-" * len(hdr))

    results = []
    for label, query, session in plan:
        row = run_turn(label, query, session)
        results.append(row)
        if row["error"]:
            print(f"{label:<18} FAILED  {row['error'][:60]}")
            continue
        tu = row["token_usage"] or {}
        print(f"{label:<18} {row['wall_seconds']:>6.1f}s "
              f"{row['engine_lifecycle_seconds']:>6.1f}s "
              f"{row['non_llm_seconds']:>7.1f}s "
              f"{int(row['engine_llm_calls']):>6} "
              f"{tu.get('input_tokens', 0):>8} "
              f"{row['peak_kv_usage_perc']:>7.1f}%")

    print("\nfetching spans from Phoenix...")
    time.sleep(6)  # OTel batch export needs a moment to flush
    spans = fetch_spans([s for _, _, s in plan])
    print(f"  {len(spans)} spans matched")

    ok = [r for r in results if not r["error"]]
    payload = {
        "collected_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "chain_base": CHAIN,
        "nim_base": NIM,
        "turns": results,
        "spans": spans,
    }
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)

    if ok:
        tot_wall = sum(r["wall_seconds"] for r in ok)
        tot_engine = sum(r["engine_lifecycle_seconds"] for r in ok)
        print(f"\nacross {len(ok)} turns: {tot_wall:.1f}s wall, "
              f"{tot_engine:.1f}s in the LLM "
              f"({tot_engine / tot_wall * 100:.0f}%), "
              f"{tot_wall - tot_engine:.1f}s elsewhere")
    print(f"written: {args.out}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted")
