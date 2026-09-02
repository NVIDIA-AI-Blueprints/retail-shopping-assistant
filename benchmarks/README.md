# Benchmarks

Tools that put load through this application and measure what happens.

The Prometheus + Grafana stack that collects the results is next door in
[`monitoring/`](../monitoring/README.md). The split is deliberate: standing up a
scraper and driving traffic are different jobs, and conflating them made it easy
to benchmark a deployment nobody had verified.

**Read [`docs/PERFORMANCE.md`](../docs/PERFORMANCE.md) before using any of
this.** It walks through the six measurements in the order that makes each one
interpretable. This file is only a map of what is here.

## Prerequisites

The app LLM must be a **local NIM** (`docker-compose-nim-local.yaml`). A hosted
endpoint publishes no metrics and rate-limits sustained load.

Start the monitoring stack first, or the resource columns come back blank:

```bash
cd ../monitoring && ./dashboard.sh up
```

## The tools, by what they drive

Grouped by which layer of the system they exercise, which is also the order you
should use them in.

### The LLM directly

| Tool | Question it answers |
|---|---|
| `bench.sh load [c] [n]` | Are the instruments working? Does anything move? |
| `bench.sh sweep [levels]` | Where does the engine saturate, and *why* — queueing, preemption, memory or compute? |
| `bench.sh top` | What is happening right now, without a browser? |

These bypass the application entirely, so they measure the engine's ceiling
rather than anything a shopper would experience.

### One turn through the whole application

| Tool | Question it answers |
|---|---|
| `turn_profile.py` | Where does one turn's time actually go, and what is the token shape? |

Run this **second, and before any sweep.** It produces the per-call input and
output token counts that `sweep` needs; without them the sweep runs at a generic
1:1 benchmark shape, and this application reads far more than it writes, so the
answer can be off by a multiple.

Needs `EXPOSE_AGENT_DIAGNOSTICS=true` or the tool-call breakdown — most of the
value — comes back empty. The script warns when it does.

### Whole journeys under concurrency

| Tool | Question it answers |
|---|---|
| `shopper_study.py` | How many concurrent shoppers, at a stated latency target? |
| `journey_load.py` | What breaks first, holding each level for a fixed duration? |
| `concurrent_shoppers.py` | Do concurrent shoppers queue behind each other end to end? |

`shopper_study.py` is the one that answers the capacity question. It fixes the
*work* per level rather than the time — N shoppers each play one complete
journey, and the level ends when the last finishes — so every level does the
same amount of work and the levels are comparable. `journey_load.py` holds each
level for a fixed duration instead, which is quicker but makes levels finish
different amounts of work.

Both report **time to first token** as the headline latency, because that is
what a shopper waits for on a streaming interface.

## Files

| Path | Purpose |
|---|---|
| `bench.sh` | Entry point for `load`, `sweep` and `top` |
| `loadgen.py` | Concurrent load; discovers the served model from `/v1/models` |
| `saturate.py` | Wraps `vllm bench serve`, samples gauges, detects saturation and names the cause |
| `metrics_top.py` | Terminal metrics view |
| `turn_profile.py` | Per-turn latency budget, reconciling app timings, engine counters and Phoenix spans |
| `shopper_study.py` | Capacity harness: fixed work per level, TTFT headline, liveness checks |
| `journey_load.py` | Journeys at rising concurrency, fixed duration per level |
| `concurrent_shoppers.py` | End-to-end serialisation check |

`loadgen.py` and `saturate.py` both resolve the served model name from
`/v1/models` at run time instead of hardcoding it, because the name differs
between a NIM and a hand-built vLLM server, and a stale name fails as an opaque
HTTP 404.

## The one thing you must not skip

**A dead model looks like a fast one.** When a model call fails, the application
returns canned text — "I could not complete that shopping request." That text
streams like any other reply: it has a first-token time and raises no exception.
A harness measuring latency and counting exceptions cannot tell it from a real
answer, and will report a full sweep of completed turns with zero errors against
a model that answered none of them.

`shopper_study.py` therefore applies three independent checks per level — probes
the model server directly before and after, confirms `vllm:prompt_tokens_total`
advanced, and matches the application's failure strings in the replies — and
aborts rather than climbing into levels that would produce the same confident
nonsense. Levels it distrusts are marked `<-- DISCARD` and excluded from the
summary.

If you write your own harness against this stack, it needs the equivalent.
