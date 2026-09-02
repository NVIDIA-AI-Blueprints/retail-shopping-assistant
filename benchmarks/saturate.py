#!/usr/bin/env python3
"""
Find the saturation point of a vLLM deployment.

Answers "can this system take more traffic?" by sweeping concurrency and
watching for the signals that mean the server has stopped keeping up.

Load is driven by vLLM's own benchmark (`vllm bench serve`) rather than a
homegrown client, so the throughput and latency numbers are the ones vLLM's
maintainers report. While each level runs, this script polls the server's
/metrics endpoint to capture things the benchmark cannot see from outside:
peak queue depth, peak KV cache usage, and whether the scheduler had to preempt.

Saturation is called on four independent signals, in priority order:

  1. PREEMPTION   vllm:num_preemptions_total increased. The scheduler evicted
                  running requests to free KV cache. Hard over-capacity.
  2. QUEUEING     vllm:num_requests_waiting went above zero. Requests are
                  waiting for a slot rather than being served.
  3. KV PRESSURE  vllm:kv_cache_usage_perc peaked above 90%. Cache-bound.
  4. PLATEAU      output token throughput gained less than PLATEAU_PCT over the
                  previous level. More concurrency is buying latency, not work.

Usage:
    ./bench.sh sweep                        # default ladder
    ./bench.sh sweep 1,4,16,64,128          # explicit levels
"""

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

BASE = os.environ.get("VLLM_BASE", "http://127.0.0.1:8000")
VENV = os.environ.get("VENV_BIN", os.path.dirname(sys.executable))
TOKENIZER = os.environ.get("TOKENIZER", "")


def resolve_served_name():
    """The name the endpoint answers to, asked rather than assumed.

    `bench serve` puts this in the request body, so a stale value returns 404
    for every request and reports as a benchmark failure rather than a config
    mistake. Set SERVED_NAME to pin it when one endpoint serves several models.
    """
    override = os.environ.get("SERVED_NAME")
    if override:
        return override
    try:
        with urllib.request.urlopen(f"{BASE}/v1/models", timeout=10) as r:
            data = json.loads(r.read())["data"]
        return data[0]["id"] if data else ""
    except Exception:
        return ""


MODEL = resolve_served_name()

INPUT_LEN = int(os.environ.get("SWEEP_INPUT_LEN", "1024"))
OUTPUT_LEN = int(os.environ.get("SWEEP_OUTPUT_LEN", "256"))
PLATEAU_PCT = float(os.environ.get("SWEEP_PLATEAU_PCT", "10"))

# Tokens of identical prefix on the front of every request, on top of
# INPUT_LEN. Zero -- the benchmark default -- makes every prompt independently
# random, so no request can ever reuse another's prefill and the prefix cache
# records queries with no hits at all.
#
# That is fine for a shape with nothing to share, and badly wrong for an agent,
# where a system prompt, tool schemas and conversation history repeat on every
# call. Measured against this application the shared part is around 44% of the
# prompt, so a sweep without it does roughly twice the prefill the real system
# does and understates throughput accordingly.
#
# Set it to the number of tokens that genuinely repeat:
#   SWEEP_PREFIX_LEN=6176 SWEEP_INPUT_LEN=7990   # 14,166 total, 44% shared
PREFIX_LEN = int(os.environ.get("SWEEP_PREFIX_LEN", "0"))

BOLD, DIM, GRN, YLW, RED, CYA, RST = (
    "\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[31m", "\033[36m", "\033[0m"
)


def get_metrics():
    with urllib.request.urlopen(f"{BASE}/metrics", timeout=10) as r:
        text = r.read().decode()
    out = {}
    for line in text.splitlines():
        if line.startswith("#") or not line:
            continue
        try:
            key, val = line.rsplit(" ", 1)
            name = key.split("{", 1)[0]
            out[name] = out.get(name, 0.0) + float(val)
        except ValueError:
            continue
    return out


class MetricSampler(threading.Thread):
    """Polls /metrics during a benchmark run to capture peaks."""

    def __init__(self, interval=0.5):
        super().__init__(daemon=True)
        self.interval = interval
        self.stop_flag = threading.Event()
        self.peak_running = 0.0
        self.peak_waiting = 0.0
        self.peak_kv = 0.0
        self.start_preemptions = None
        self.end_preemptions = None

    def run(self):
        while not self.stop_flag.is_set():
            try:
                m = get_metrics()
            except Exception:
                time.sleep(self.interval)
                continue
            self.peak_running = max(self.peak_running,
                                    m.get("vllm:num_requests_running", 0.0))
            self.peak_waiting = max(self.peak_waiting,
                                    m.get("vllm:num_requests_waiting", 0.0))
            self.peak_kv = max(self.peak_kv,
                               m.get("vllm:kv_cache_usage_perc", 0.0))
            p = m.get("vllm:num_preemptions_total", 0.0)
            if self.start_preemptions is None:
                self.start_preemptions = p
            self.end_preemptions = p
            time.sleep(self.interval)

    @property
    def preempted(self):
        if self.start_preemptions is None or self.end_preemptions is None:
            return 0.0
        return max(0.0, self.end_preemptions - self.start_preemptions)


def run_level(concurrency, num_prompts):
    """Run one benchmark level, returning bench results plus metric peaks."""
    out_json = tempfile.mktemp(suffix=".json")
    cmd = [
        os.path.join(VENV, "vllm"), "bench", "serve",
        "--base-url", BASE,
        "--model", MODEL,
        "--dataset-name", "random",
        "--random-input-len", str(INPUT_LEN),
        "--random-output-len", str(OUTPUT_LEN),
        "--num-prompts", str(num_prompts),
        "--max-concurrency", str(concurrency),
        # Fixed output length keeps levels comparable; without it the model
        # stops at its own EOS and each level measures a different workload.
        "--ignore-eos",
        "--save-result", "--result-filename", out_json,
    ]
    if PREFIX_LEN:
        cmd += ["--random-prefix-len", str(PREFIX_LEN)]
    if TOKENIZER:
        cmd += ["--tokenizer", TOKENIZER]

    sampler = MetricSampler()
    sampler.start()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    sampler.stop_flag.set()
    sampler.join(timeout=3)

    if proc.returncode != 0:
        return None, proc.stderr[-500:], sampler

    try:
        with open(out_json) as fh:
            res = json.load(fh)
        os.unlink(out_json)
    except Exception as exc:
        return None, f"could not read results: {exc}", sampler

    return res, None, sampler


def main():
    if len(sys.argv) > 1 and sys.argv[1]:
        levels = [int(x) for x in sys.argv[1].replace(" ", "").split(",") if x]
    else:
        levels = [1, 4, 8, 16, 32, 64, 128]

    try:
        urllib.request.urlopen(f"{BASE}/v1/models", timeout=5)
    except Exception:
        sys.exit(f"endpoint {BASE} is not responding")

    print(f"{BOLD}Saturation sweep{RST}  {DIM}{BASE}{RST}")
    print(f"{DIM}load driven by 'vllm bench serve'; "
          f"{INPUT_LEN + PREFIX_LEN} in / {OUTPUT_LEN} out tokens per request, "
          f"ignore-eos"
          + (f"; {PREFIX_LEN} of the input is a prefix shared by every request "
             f"({PREFIX_LEN / (INPUT_LEN + PREFIX_LEN) * 100:.0f}%), "
             f"so the prefix cache has something to reuse"
             if PREFIX_LEN else
             "; every prompt independently random, so the prefix cache "
             "cannot hit"))
    print(f"levels: {', '.join(map(str, levels))}{RST}\n")

    hdr = (f"{'conc':>5} {'reqs':>5} {'out tok/s':>10} {'tot tok/s':>10} "
           f"{'req/s':>7} {'TTFT p99':>9} {'TPOT p99':>9} "
           f"{'peak run':>9} {'peak wait':>10} {'peak KV':>8} {'preempt':>8}")
    print(f"{BOLD}{hdr}{RST}")
    print(DIM + "-" * len(hdr) + RST)

    rows = []
    for conc in levels:
        num_prompts = min(max(3 * conc, 24), 384)
        res, err, sampler = run_level(conc, num_prompts)
        if res is None:
            print(f"{conc:>5} {RED}failed: {err[:80]}{RST}")
            continue

        row = {
            "concurrency": conc,
            "num_prompts": num_prompts,
            "completed": res.get("completed"),
            "failed": res.get("failed"),
            "output_throughput": res.get("output_throughput", 0.0),
            "total_token_throughput": res.get("total_token_throughput", 0.0),
            "request_throughput": res.get("request_throughput", 0.0),
            "mean_ttft_ms": res.get("mean_ttft_ms", 0.0),
            "p99_ttft_ms": res.get("p99_ttft_ms", 0.0),
            "mean_tpot_ms": res.get("mean_tpot_ms", 0.0),
            "p99_tpot_ms": res.get("p99_tpot_ms", 0.0),
            "peak_running": sampler.peak_running,
            "peak_waiting": sampler.peak_waiting,
            "peak_kv": sampler.peak_kv,
            "preempted": sampler.preempted,
            "duration": res.get("duration", 0.0),
        }
        rows.append(row)

        warn = ""
        if row["preempted"] > 0:
            warn = RED
        elif row["peak_waiting"] > 0 or row["peak_kv"] > 0.9:
            warn = YLW
        print(f"{warn}{conc:>5} {row['completed']:>5} "
              f"{row['output_throughput']:>10.1f} "
              f"{row['total_token_throughput']:>10.1f} "
              f"{row['request_throughput']:>7.2f} "
              f"{row['p99_ttft_ms']:>8.0f}m {row['p99_tpot_ms']:>8.1f}m "
              f"{row['peak_running']:>9.0f} {row['peak_waiting']:>10.0f} "
              f"{row['peak_kv'] * 100:>7.1f}% {row['preempted']:>8.0f}{RST}")

    # Written before the analysis gate, and to a per-run name by default. A sweep
    # is minutes of load against a shared server, so a run that cannot be
    # summarised -- a single level, or a ladder cut short -- must still leave its
    # measurements behind rather than exiting and discarding them.
    # The shape is in the name as well as the ladder. Two sweeps that differ
    # only in prefix length are exactly the comparison worth making, and naming
    # them by concurrency alone means the second silently overwrites the first.
    out = os.environ.get(
        "SWEEP_OUT",
        f"/tmp/sweep_{INPUT_LEN + PREFIX_LEN}in-{OUTPUT_LEN}out"
        f"-{PREFIX_LEN}shared"
        f"_{'-'.join(str(r['concurrency']) for r in rows)}.json")
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\n  full results: {out}")

    if len(rows) < 2:
        print("  single level: no trend to analyse, compare against other runs")
        return

    analyze(rows)


def analyze(rows):
    """Turn sweep rows into a verdict. Separate so saved results can be
    re-read with --analyze without paying for another sweep."""
    print(f"\n{BOLD}{CYA}Analysis{RST}")

    best = max(rows, key=lambda r: r["output_throughput"])
    print(f"  peak measured throughput  {BOLD}{best['output_throughput']:.1f} "
          f"output tok/s{RST} at concurrency {best['concurrency']}")

    # Two limits matter and they are not the same number. Reporting only the
    # first queueing event understates capacity badly: throughput can keep
    # climbing well past the point where a few requests start waiting.
    #
    #   capacity ceiling - concurrency where output throughput stops improving
    #   latency knee     - highest concurrency still free of queueing
    capacity = rows[0]
    for prev, cur in zip(rows, rows[1:]):
        if prev["output_throughput"] <= 0:
            continue
        gain = (cur["output_throughput"] - prev["output_throughput"]) \
            / prev["output_throughput"] * 100
        if gain < PLATEAU_PCT:
            capacity = prev
            break
        capacity = cur

    clean = [r for r in rows if r["peak_waiting"] == 0]
    knee = clean[-1] if clean else None

    print(f"  {BOLD}capacity ceiling{RST}          concurrency "
          f"{capacity['concurrency']} at "
          f"{capacity['output_throughput']:.0f} output tok/s "
          f"{DIM}(throughput stops improving past here){RST}")
    if knee:
        print(f"  {BOLD}latency-clean ceiling{RST}     concurrency "
              f"{knee['concurrency']} at {knee['output_throughput']:.0f} "
              f"output tok/s, p99 TTFT {knee['p99_ttft_ms']:.0f}ms "
              f"{DIM}(no queueing at all){RST}")
    else:
        print(f"  {YLW}every level queued{RST} - the clean ceiling is below "
              f"concurrency {rows[0]['concurrency']}")

    # What is actually binding? Distinguish a scheduler cap from real hardware
    # limits, because the fix is completely different.
    print(f"\n{BOLD}{CYA}What is the binding constraint?{RST}")
    saturated = [r for r in rows if r["concurrency"] >= capacity["concurrency"]]
    max_running = max(r["peak_running"] for r in rows)
    plateaued_running = [r for r in saturated
                         if abs(r["peak_running"] - max_running) < 1]
    queued_beyond = [r for r in rows
                     if r["concurrency"] > max_running and r["peak_waiting"] > 0]
    peak_kv_overall = max(r["peak_kv"] for r in rows)
    any_preempt = any(r["preempted"] > 0 for r in rows)

    if len(plateaued_running) >= 2 and queued_beyond and not any_preempt \
            and peak_kv_overall < 0.85:
        print(f"  {YLW}SCHEDULER CAP{RST}, not hardware.")
        print(f"    Requests in flight never exceeded {max_running:.0f} no "
              f"matter how much load was offered, while the queue grew to "
              f"{max(r['peak_waiting'] for r in rows):.0f}.")
        print(f"    That is max_num_seqs={max_running:.0f} holding the batch "
              f"size down.")
        print(f"    Meanwhile KV cache peaked at only "
              f"{peak_kv_overall * 100:.1f}% and there were zero preemptions,")
        print(f"    so there is memory headroom to raise it.")
        print(f"    {BOLD}Try MAX_NUM_SEQS={int(max_running) * 4} and re-run "
              f"this sweep.{RST}")
    elif any_preempt:
        print(f"  {RED}KV CACHE / MEMORY{RST}. The scheduler had to preempt "
              f"running requests.")
        print(f"    Lower MAX_MODEL_LEN or raise GPU_MEM_UTIL to grow the "
              f"cache.")
    elif peak_kv_overall > 0.85:
        print(f"  {YLW}KV CACHE{RST}. Usage peaked at "
              f"{peak_kv_overall * 100:.1f}% without preemption - close to the "
              f"edge.")
        print(f"    Lower MAX_MODEL_LEN to fit more concurrent sequences.")
    elif capacity["concurrency"] == rows[-1]["concurrency"]:
        print(f"  {GRN}NOT YET FOUND{RST}. Throughput was still climbing at "
              f"the top of the ladder.")
        print(f"    Re-run higher: ./bench.sh sweep "
              f"{rows[-1]['concurrency'] * 2},{rows[-1]['concurrency'] * 4}")
    else:
        print(f"  {BOLD}COMPUTE{RST}. Throughput plateaued with KV cache at "
              f"only {peak_kv_overall * 100:.1f}% and no preemption,")
        print(f"    so the GPUs are the limit rather than memory or the "
              f"scheduler.")
        print(f"    A tuned fused-MoE kernel config or enabling MTP is the "
              f"lever here, not more concurrency.")

    # Latency cost of concurrency, which is the real operational tradeoff.
    first, last = rows[0], rows[-1]
    print(f"\n{BOLD}{CYA}Cost of pushing past the ceiling{RST}")
    print(f"  p99 TTFT   {first['p99_ttft_ms']:>9.0f}ms at conc "
          f"{first['concurrency']:<4} -> {last['p99_ttft_ms']:>9.0f}ms at conc "
          f"{last['concurrency']}   {DIM}queueing shows up here first{RST}")
    print(f"  p99 TPOT   {first['p99_tpot_ms']:>9.1f}ms at conc "
          f"{first['concurrency']:<4} -> {last['p99_tpot_ms']:>9.1f}ms at conc "
          f"{last['concurrency']}   {DIM}per-token speed once running{RST}")
    if first["output_throughput"] > 0:
        print(f"  batching gain: "
              f"{last['output_throughput'] / first['output_throughput']:.1f}x "
              f"throughput, conc {first['concurrency']} -> "
              f"{last['concurrency']}")
    if last["p99_ttft_ms"] > 3 * (knee or last)["p99_ttft_ms"] and knee:
        print(f"  {DIM}TTFT inflating while TPOT stays flat is the signature of "
              f"queueing, not slow decode.{RST}")


def print_table(rows):
    hdr = (f"{'conc':>5} {'reqs':>5} {'out tok/s':>10} {'tot tok/s':>10} "
           f"{'req/s':>7} {'TTFT p99':>9} {'TPOT p99':>9} "
           f"{'peak run':>9} {'peak wait':>10} {'peak KV':>8} {'preempt':>8}")
    print(f"{BOLD}{hdr}{RST}")
    print(DIM + "-" * len(hdr) + RST)
    for row in rows:
        warn = RED if row["preempted"] > 0 else (
            YLW if row["peak_waiting"] > 0 or row["peak_kv"] > 0.9 else "")
        print(f"{warn}{row['concurrency']:>5} {row['completed']:>5} "
              f"{row['output_throughput']:>10.1f} "
              f"{row['total_token_throughput']:>10.1f} "
              f"{row['request_throughput']:>7.2f} "
              f"{row['p99_ttft_ms']:>8.0f}m {row['p99_tpot_ms']:>8.1f}m "
              f"{row['peak_running']:>9.0f} {row['peak_waiting']:>10.0f} "
              f"{row['peak_kv'] * 100:>7.1f}% {row['preempted']:>8.0f}{RST}")


if __name__ == "__main__":
    try:
        if len(sys.argv) > 2 and sys.argv[1] == "--analyze":
            with open(sys.argv[2]) as fh:
                saved = json.load(fh)
            print(f"{BOLD}Re-analysis of {sys.argv[2]}{RST}\n")
            print_table(saved)
            analyze(saved)
        else:
            main()
    except KeyboardInterrupt:
        print("\ninterrupted")
