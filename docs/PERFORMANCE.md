# Performance

How to measure this application, in the order that makes each step
interpretable, and how to read what comes back.

Written for someone who has not done performance work before. It follows a
ladder: each rung tells you something the next one needs. Skipping to the bottom
rung — "how many users can we serve" — produces a number you cannot defend,
because you will not know what limited it.

This document deliberately contains **no measured results**. Throughput and
latency depend on the GPU, the model profile, the catalog and the env profile in
use, so a figure recorded here would be read as a target on hardware it never
described. Where a number appears, it is illustrating the *shape* of an output,
not reporting a finding. Measure your own, on the deployment you are actually
asking about.

Everything here needs the app LLM to be a **local NIM**
([`docker-compose-nim-local.yaml`](../docker-compose-nim-local.yaml), configured
by [`.env.local-llm.example`](../.env.local-llm.example)). A hosted endpoint
publishes no metrics and rate-limits sustained load, so there is nothing to
measure and no way to measure it.

---

## Contents

- [Where things live](#where-things-live)
- [The vocabulary](#the-vocabulary)
- [Before you measure anything](#before-you-measure-anything)
- [The measurement ladder](#the-measurement-ladder) — six steps, in order
- [The scripts, one by one](#the-scripts-one-by-one)
- [Pitfalls that produce confident, wrong numbers](#pitfalls-that-produce-confident-wrong-numbers)

---

## Where things live

Two directories, split by what they do.

| | Directory | Entry point |
|---|---|---|
| Collecting metrics | [`monitoring/`](../monitoring/README.md) | `./dashboard.sh up` |
| Generating load and measuring | [`benchmarks/`](../benchmarks/README.md) | `./bench.sh` |

They are separate because standing up a scraper and driving traffic through the
system are different jobs, and conflating them made it easy to benchmark a
deployment nobody had verified. Each entry point will redirect you if you ask it
for the other's subcommands.

```
monitoring/                    benchmarks/
  dashboard.sh   up/down/         bench.sh          load/sweep/top
                 status/urls/     loadgen.py
                 logs             saturate.py
  docker-compose.yaml            metrics_top.py
  prometheus.yaml                turn_profile.py
  grafana/                       shopper_study.py
  docker_stats_exporter.py       journey_load.py
                                 concurrent_shoppers.py
```

---

## The vocabulary

**Latency** is how long one shopper waits. **Throughput** is how many you finish
per second. They pull against each other: running 32 requests at once instead of
4 finishes more work per second, but each request now shares the GPU, so each
takes longer. Capacity planning is choosing a point on that curve — which means
**a capacity number is meaningless without a latency target attached.**

**Time to first token (TTFT)** is how long the shopper stares at nothing. It
dominates perceived speed: a reply that starts in 300 ms and runs for 8 s feels
faster than one that starts after 5 s and finishes in 6. For a streaming UI this
is the number to size against, not total turn time.

**Time per output token (TPOT)** is typing speed once words start.

**Prefill vs decode** explains most of this system. A request first *reads* the
whole prompt, all tokens in parallel, limited by compute. It then *writes* the
answer one token at a time, consulting the whole model for each, limited by
memory bandwidth. They saturate for different reasons and respond to different
fixes.

**Concurrency** is how many requests are in flight at once — the dial a sweep
turns. Not the same as requests per second.

**Percentiles.** p50 is typical, p95/p99 is the bad case. Always look at the
tail; a mean hides the shopper who waited three times as long.

**Prefix caching** lets the engine skip re-reading a prompt prefix it has
already processed. It matters more here than in most applications, for a reason
worth understanding before you measure it — see
[step 6](#step-6--how-many-concurrent-shoppers-end-to-end).

### Why the workload shape matters here

A shopper turn is **heavily prefill-dominated**: it sends a very large prompt and
gets a short reply, several times over. Every model call carries the system
instructions, all tool definitions, the conversation so far, and any catalog
results, while the reply is a few sentences.

Published LLM benchmarks typically use a 1:1 input:output ratio. Those numbers
will not predict this application. A workload that reads far more than it writes
reaches its limits somewhere completely different.

So do not sweep at a tool's default shape. Take the real ratio from your own
deployment first (step 2), then feed it into the sweep (step 5). A sweep at the
wrong shape produces confident numbers describing a workload you do not run, and
the gap between the two is routinely a multiple, not a few percent.

---

## Before you measure anything

Measuring the wrong thing is worse than not measuring. Every wrong conclusion in
this area starts with a deployment that was not doing what the measurer assumed.
Confirm all six.

```bash
# 1. The NIM is serving, at the precision you expect
curl -s http://localhost:8000/v1/models | python3 -m json.tool
docker logs retail-shopping-assistant-nemotron-1 2>&1 | grep -iE "Precision:|quant_algo"
#    want: Precision: fp8   /   quant_algo=FP8

# 2. The catalog is indexed -- an empty one yields confident nonsense
curl -s http://localhost:8010/ready
#    want: {"status":"ready","catalog_id":"fashion_products","products":<n>}

# 3. The app is really using the local NIM. Count requests either side of a
#    query and check the delta equals the reported model_calls.
before=$(docker logs retail-shopping-assistant-nemotron-1 2>&1 | grep -c "POST /v1/chat/completions")
curl -s -X POST http://localhost:8009/query/timing -H 'Content-Type: application/json' \
  -d '{"user_id":1,"query":"Show me black dresses under $100","session_id":"smoke-1"}' \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['token_usage'])"
after=$(docker logs retail-shopping-assistant-nemotron-1 2>&1 | grep -c "POST /v1/chat/completions")
echo "local LLM calls: $((after-before))"

# 4. Monitoring is scraping all three jobs, not just Prometheus itself
cd monitoring && ./dashboard.sh status
#    want: vllm, dcgm and containers all healthy
```

Step 3 is the one people skip, and it is the one that catches a hosted endpoint
quietly serving your "local" benchmark.

Two more checks are specific to this model, and each has silently invalidated
whole days of measurement:

```bash
# 5. Tool calling actually works. If the parser cannot read this model's
#    output, vLLM returns the tool call as plain text, the agent answers
#    without ever searching, and every latency number describes a
#    different, much cheaper application.
curl -s http://localhost:8009/query/timing -X POST -H 'Content-Type: application/json' \
  -d '{"user_id":1,"query":"Show me black dresses under $100","session_id":"tool-1"}' \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print('tools called:', d.get('agent_diagnostics',{}).get('tool_calls'))"
#    want: a non-empty list including a catalog search

# 6. Prefix caching resolved the way you asked. Unset does not mean default-on;
#    vLLM disables it for hybrid attention models unless told otherwise.
curl -s localhost:8000/metrics | grep -o 'enable_prefix_caching="[^"]*"'
curl -s localhost:8000/metrics | grep -o 'mamba_block_size="[^"]*"'
#    want: "True", and a small mamba_block_size, not the full context length
```

---

## The measurement ladder

Six steps. Each one answers a question the next step assumes you have answered.

| Step | Question it answers | Command |
|---|---|---|
| 1 | Are my instruments telling the truth? | `monitoring/dashboard.sh up`, `benchmarks/bench.sh load` |
| 2 | What does one turn cost, and in what shape? | `benchmarks/turn_profile.py` |
| 3 | What moves when I add concurrency? | `benchmarks/bench.sh load` at rising levels |
| 4 | Where does the *model* saturate, generically? | `benchmarks/bench.sh sweep` |
| 5 | Where does it saturate at *my* workload shape? | `benchmarks/bench.sh sweep` + `SWEEP_*` |
| 6 | How many concurrent shoppers, end to end? | `benchmarks/shopper_study.py` |

### Step 1 — Prove the instruments work

```bash
cd monitoring
./dashboard.sh up
```

This checks the NIM is healthy and exposing metrics, starts Prometheus, Grafana
and the exporters, waits for the scrape target to come up and the dashboards to
provision, then prints URLs.

| Service | URL |
|---|---|
| Grafana | http://localhost:6005 |
| Prometheus | http://localhost:9090 |
| Phoenix (agent traces) | http://localhost:6006 |
| Raw NIM metrics | http://localhost:8000/metrics |

Grafana is on **6005** because the app's nginx owns 3000 and Phoenix owns 6006.
Both Prometheus and Grafana bind loopback only; reach them remotely with:

```bash
ssh -N -L 6005:localhost:6005 -L 9090:localhost:9090 <user>@<host>
```

**An idle dashboard looks exactly like a broken one.** Do not conclude anything
until you have driven traffic and seen the panels move:

```bash
cd ../benchmarks
./bench.sh load 8 32      # 32 requests, 8 at a time
```

Then confirm every data source is being collected, not just Prometheus itself.
`dashboard.sh status` lists all three jobs, or query it directly:

```bash
curl -s 'http://localhost:9090/api/v1/targets?state=active' \
  | python3 -c "
import json,sys
for t in json.load(sys.stdin)['data']['activeTargets']:
    print(t['labels'].get('job'), t['health'], t.get('lastError',''))"
#    want: vllm up, dcgm up, containers up
```

If a job is `down`, fix it now. A missing exporter does not announce itself
later; it shows up as a panel reading zero, which is indistinguishable from a
real zero.

### Step 2 — Profile one turn, and learn your workload shape

```bash
python3 benchmarks/turn_profile.py --out /tmp/profile.json
```

This is the highest-value single measurement in the ladder, because it produces
the numbers steps 4 and 5 need as inputs. **Do it before any sweep.**

It measures the same turn from three vantage points and reconciles them, because
no single one is sufficient. The app's `/query/timing` gives per-phase timings
and which model roles were called; the NIM's counters give the true
prefill/decode/queue split; Phoenix spans give per-LLM-call and per-tool-call
granularity. The app cannot see inside the model, the model cannot see
retrieval, and reconciling all three is what makes the unattributed remainder
visible.

What to take away from it:

- **Input and output tokens per call.** Divide per-turn totals by `model_calls`.
  These become `SWEEP_INPUT_LEN` and `SWEEP_OUTPUT_LEN` in step 5.
- **How many model calls a turn makes.** Likely more than you expect, and each
  one re-sends the shared prompt.
- **How much of the prompt never changes** — system instructions plus tool
  schemas. That is your `SWEEP_PREFIX_LEN`, and it predicts what prefix caching
  can win.
- **Where non-model time goes** — catalog search, memory, guardrails.

Two cautions. Set `EXPOSE_AGENT_DIAGNOSTICS=true` first or the tool-call
breakdown comes back empty. And the streaming endpoint's `total_seconds` is the
**sum of overlapping buckets** (`deepagents` already contains `catalog_search`
and the safety checks), so it over-counts and is not wall clock;
`/query/timing` reports a true wall-clock `timings.total`.

### Step 3 — Add concurrency and watch what moves

```bash
cd benchmarks
./bench.sh load 1 20
./bench.sh load 4 40
./bench.sh load 16 80
./bench.sh load 64 160
```

You are not after a number here, you are building intuition about which panels
respond to load and which do not. Watch, in Grafana or via `./bench.sh top`:

- **running vs waiting** requests — the scheduler's own view
- **KV cache usage** — does it climb linearly with concurrency?
- **preemptions** — should be zero; any non-zero means eviction and recompute
- **GPU utilisation** — note how early it saturates, and distrust it after that

The lesson this step teaches is that GPU utilisation is nearly useless as a
capacity signal. It reports that at least one SM had work scheduled during the
sampling window, not that the GPU delivered anything near peak, so it can read
high while the machine still has several times more work to give.

### Step 4 — Find the model's saturation point, generically

```bash
./bench.sh sweep 1,2,4,8,16,32,64
```

Load comes from vLLM's own benchmark, so throughput and latency are the figures
vLLM's maintainers publish rather than a homegrown approximation. While each
level runs, the script separately polls `/metrics` twice a second to capture what
the benchmark cannot see from outside: peak queue depth, peak KV usage, and
whether the scheduler had to **preempt**.

It calls saturation on four signals in order of severity — preemption, then
queueing, then KV pressure above 90%, then a throughput plateau. They are
distinguished because the fixes differ completely: queueing can mean a scheduler
setting is too low, a one-line change, while preemption means you genuinely ran
out of memory. Reporting "saturated" without saying *why* leads to buying
hardware when a config edit would have done.

The sweep needs two things nothing else here does. `./bench.sh sweep` explains
both if they are missing:

```bash
# vLLM's benchmark CLI (large download)
python3 -m venv ~/vllm-bench && ~/vllm-bench/bin/pip install vllm==0.17.1
VLLM_BIN=~/vllm-bench/bin ./bench.sh sweep

# a tokenizer -- the benchmark counts tokens locally (a few MB, not the weights)
hf download <your-model-repo> \
  --include 'tokenizer*' 'config.json' --local-dir ~/model-tokenizer
TOKENIZER=~/model-tokenizer ./bench.sh sweep
```

**Reading a sweep table.** The columns, with illustrative values to show the
format rather than to report a result:

```
 conc  reqs  out tok/s  tot tok/s   req/s  TTFT p99  TPOT p99  peak run  peak wait  peak KV  preempt
    8    24      200.0    16000.0    1.75     3200m     34.0m         8          5     4.0%        0
```

At concurrency 8: 24 requests completed, 200 output tokens/sec (16,000 including
prompt processing), 1.75 req/sec, 99% of requests began replying within
3,200 ms, tokens then arrived every 34 ms, 8 running at peak, **5 waiting in a
queue**, cache peaked 4% full, nothing evicted.

Scan the **peak wait** and **preempt** columns. While both are zero you have
headroom. The first level where `peak wait` leaves zero is where requests start
waiting for a slot rather than for the model.

**Read total tok/s, not output tok/s, for this workload.** Output looks feeble
because each call writes a couple of hundred tokens while reading many
thousands. Judging a prefill-heavy workload by output rate badly understates the
machine.

### Step 5 — Sweep again at your own workload shape

Step 4's defaults are a generic benchmark shape. Now use what step 2 measured:

```bash
SWEEP_INPUT_LEN=<your per-call input tokens> \
SWEEP_OUTPUT_LEN=<your per-call output tokens> \
SWEEP_PREFIX_LEN=<the shared, unchanging part of the prompt> \
  ./bench.sh sweep 1,8,32,64,128
```

`bench.sh` warns if you sweep without these, because the default shape answers a
different question.

`SWEEP_PREFIX_LEN` passes `--random-prefix-len` to the benchmark, giving every
generated prompt the same fixed prefix. Without it the benchmark sends
independent random prompts, which share nothing, so **prefix caching cannot
engage and the sweep measures substantially more prefill work than your
application actually does.**

Expect this step to disagree sharply with step 4. That disagreement is the single
most important reason not to size a deployment from published benchmarks.

**A warning about what this step cannot tell you.** Even with a shared prefix,
this is a *closed-loop* benchmark: it launches a synchronised burst and holds
concurrency fixed. Real shoppers arrive independently. A synchronised burst can
make every request miss the cache simultaneously, so the sweep may fail to
measure caching at all at the concurrencies you care about. Treat step 5's
throughput as a **floor**, and get your caching answer from step 6.

### Step 6 — How many concurrent shoppers, end to end

```bash
# the model's ceiling: nobody pauses, everybody starts together
python3 benchmarks/shopper_study.py --levels 1,2,4,8,16,32 --think 0

# the honest caching number: different conversations, not N copies of one
python3 benchmarks/shopper_study.py --mixed --text-only --levels 4,8,16,32

# closer to real people: they read, and they do not arrive in step
python3 benchmarks/shopper_study.py --levels 8,16,32 --think 20 --stagger 30
```

This drives whole journeys through the real application — agent, tools,
retrieval, cart — and reports **TTFT as the headline latency**, because that is
what a shopper waits for.

It fixes the *work* rather than the time: N shoppers each play one complete
journey and the level ends when the last finishes, so every level does exactly
N × (turns per journey) and the levels are directly comparable. This is the
difference from [`journey_load.py`](../benchmarks/journey_load.py), which holds
each level for a fixed duration.

Read the output like this:

- **Find where p95 TTFT crosses your target.** That crossing, not any maximum,
  is your capacity number. State the target alongside it, always.
- **Check `cache%`.** This is where you learn what prefix caching is really
  worth, because unlike step 5 the arrivals are realistic.
- **Check `app cores`.** If the application tier is cheap, scaling it is wasted
  money. If it is not, you have found a different problem.
- **Any level marked `<-- DISCARD` is not a measurement.** See below.

**Interpreting the cache hit rate.** Two different things can produce one, and
they have opposite consequences for capacity planning. *Lateral* reuse is
shoppers sharing text with each other; if that is what you are seeing, capacity
depends on your users resembling one another. *Longitudinal* reuse is a single
conversation reusing its own earlier work — a turn makes several model calls that
each re-send the system prompt, tool schemas and history, and turn N+1's prompt
contains turn N's prompt as a prefix. To tell them apart, run at **concurrency
1**, where lateral sharing is physically impossible, and compare. Predict the
longitudinal share ahead of time by dividing the unchanging boilerplate from
step 2 by the per-call input length.

**Why `--text-only` exists.** Some journeys attach an image or video. Without an
image embedding service deployed, those turns exercise a fallback path, so
including them measures the fallback rather than the product. The flag skips them
and says which.

**Why the liveness checks exist.** This is the most important thing in this
document. When a model call fails, the application returns canned text — "I
could not complete that shopping request." That text *streams like any other
reply*: it has a first-token time and raises no exception. A harness measuring
latency and counting exceptions cannot tell it from a real answer, and will
cheerfully report a full sweep of completed turns with zero errors against a
model that answered none of them.

`shopper_study.py` therefore applies three independent checks per level — probe
the model server directly before and after, confirm `vllm:prompt_tokens_total`
advanced, and match the application's failure strings in the replies — and
aborts the sweep rather than climbing into levels that would produce the same
confident nonsense. Any harness you point at this stack needs the equivalent.

The tells to recognise in your own data, if you ever suspect it:

- A cache hit rate of **exactly** zero, which is what a missing metric looks like
  rather than a real effect.
- Latencies **clustered on a constant** across different load levels. Real
  latency does not do that; a timeout does.
- Throughput **rising** at the highest level while latency stays flat, which no
  saturated system does.

---

## The scripts, one by one

### `monitoring/dashboard.sh`

Manages the observability stack: `up`, `down`, `status`, `urls`, `logs`. Checks
the NIM is healthy and exposing `vllm:` series *before* standing up a scraper,
then waits for the target to go healthy and the dashboards to provision rather
than returning optimistically. `status` lists all three scrape jobs, because a
missing exporter shows up as a panel reading zero rather than as an error.

### `monitoring/docker_stats_exporter.py`

Exports per-container CPU and memory in Prometheus format, reading the Docker
Engine API over its socket.

This exists because cAdvisor, the usual choice, could not resolve container
metadata on a host using the `overlayfs` storage driver — it reported container
IDs instead of service names across three versions. Without per-container
figures, a run limited by Milvus, the chain server or the retriever is
indistinguishable from one limited by the GPUs, and those call for opposite
responses. It needs no privileged container and mounts nothing but the socket.

### `benchmarks/bench.sh`

Entry point for the three subcommands that need environment plumbing —
resolving the served model name, locating the `vllm` CLI, finding a tokenizer on
disk.

| Command | What it does | Needs |
|---|---|---|
| `load [conc] [n]` | Drive concurrent traffic at the NIM directly | — |
| `sweep [levels]` | Concurrency sweep to find saturation | `vllm` CLI + tokenizer |
| `top [secs]` | Live metrics in the terminal, no browser | — |

Run `top` in a real terminal, not inside a tool session — it is a full-screen
TUI and will wedge a non-interactive shell.

### `benchmarks/loadgen.py`

Concurrent request generator against the NIM's OpenAI-compatible endpoint.
Standard library only, so it runs anywhere. Discovers the served model name from
`/v1/models` rather than hardcoding it, which matters because the name differs
between a hand-built vLLM server and a NIM, and a stale name yields HTTP 404 for
every request. Behind `bench.sh load`.

### `benchmarks/saturate.py`

The sweep. Wraps `vllm bench serve` for load generation while polling `/metrics`
twice a second for what the benchmark cannot see from outside — queue depth, KV
usage, preemptions. Resolves the served model name and passes it explicitly, and
honours `SWEEP_INPUT_LEN`, `SWEEP_OUTPUT_LEN` and `SWEEP_PREFIX_LEN`. Writes
results named by shape as well as concurrency, so runs at different shapes do not
overwrite each other. Behind `bench.sh sweep`.

### `benchmarks/metrics_top.py`

Terminal dashboard for the NIM's metrics, for when you have no browser or no port
forward. Behind `bench.sh top`.

### `benchmarks/turn_profile.py`

Per-turn latency budget through the whole app, reconciling app timings, engine
counters and Phoenix spans. Step 2's tool, and the one that produces the token
shape every later step depends on. **Set `EXPOSE_AGENT_DIAGNOSTICS=true` first**
or the tool-call breakdown — most of the value — comes back empty. The script
warns when it does.

### `benchmarks/shopper_study.py`

The capacity harness. Step 6's tool. Fixed work per level, TTFT headline,
optional think time and staggered arrivals, `--mixed` for independent
conversations, `--text-only` to skip media journeys, and the three liveness
checks described above. Captures GPU, KV, cache hit rate and per-container CPU
from Prometheus for each level, and writes a JSON file per run.

```
usage: shopper_study.py [--levels 1,2,4] [--journey J01] [--mixed]
                        [--think N] [--stagger N] [--text-only]
                        [--settle N] [--out PATH]
```

### `benchmarks/journey_load.py`

Drives real journeys at rising concurrency for a **fixed duration** per level,
and names the limiting factor it hit. Useful for a quick "what breaks first"
answer. Prefer `shopper_study.py` when you need levels to be comparable to each
other, since fixed-duration levels finish different amounts of work.

### `benchmarks/concurrent_shoppers.py`

A narrower question: do concurrent shoppers queue behind each other end to end?
Good for catching application-level serialisation that a model-level sweep cannot
see.

### `tests/evaluation/src/replay.py`

The **correctness** harness, which also records a stopwatch and TTFT.

```bash
python -m tests.evaluation.src.replay --label perf-check     # all, sequentially
python -m tests.evaluation.src.replay --only J01,J13 --label spot
```

Use it to confirm behaviour stayed correct and turns stayed roughly as quick. Do
**not** use it to find a capacity ceiling: concurrency is capped, and every turn
runs different content, so a slow turn may simply have been a harder one. It
saves full transcripts under `tests/evaluation/results/val/<label>/`, which is
the fastest way to see whether the assistant is actually answering or quietly
apologising.

Note the evaluation harness uses a hosted model as its judge, so journey runs are
exposed to the per-IP rate limit described below, independently of where the app
LLM runs.

---

## Pitfalls that produce confident, wrong numbers

Every one of these has invalidated real measurements on this stack.

**A dead model looks like a fast one.** Covered in step 6, and the single most
expensive mistake available here.

**The tool-call parser silently dropping calls.** `pythonic` cannot read this
model's XML-form tool calls, so vLLM returns them as plain assistant text with no
`tool_calls` and the agent answers without ever searching. Every latency number
measured in that state describes a much cheaper application that never used its
tools. Check `agent_diagnostics.tool_calls` is non-empty before believing
anything. Use `qwen3_coder`.

**Sampling slower than the thing you measure.** `dcgm-exporter` defaults to a
30-second collection interval, longer than many runs and far longer than a
prefill burst, so the same stale idle sample is scraped repeatedly and
utilisation reads zero through work that plainly happened. It is set to 1 second
in `monitoring/docker-compose.yaml`.

**Benchmarks that replay themselves.** `vllm bench serve` defaults to `--seed 0`,
so every level generates byte-identical prompts. Those exact repeats produce
prefix cache hits that have nothing to do with the prefix under test, which is
enough to invalidate an entire prefix-caching experiment.

**Closed-loop load is not how users arrive.** A fixed-concurrency burst can make
every request miss the cache simultaneously. A load generator can be wrong not
only about the size of the work but about its *arrival pattern*, and the second
is much easier to miss.

**Gauges read zero at idle.** `vllm:kv_cache_usage_perc` is a gauge and means
nothing unless sampled during load. The sweep polls it twice a second for that
reason.

**Panel windows hide bursts.** A 5-minute average includes the idle time either
side of a burst, so a short burst at a high rate can display as a small fraction
of it. Both figures are correct; check what window a panel averages before
comparing it to anything.

**Env profiles blend rather than switch.** Every `.env.*` profile is a sourced
shell file using `${VAR:-default}`, so an already-exported value wins over the
file's default. Sourcing two profiles in one shell silently mixes them, with
whichever came first winning on overlapping names. Two that bite:
`CATALOG_IMAGE_EMBEDDING_ENABLED` leaking in as `true` changes indexing time and
per-turn latency, so runs you meant to compare are not comparable; and
`TEXT_EMBED_BASE_URL` unset falls back to `integrate.api.nvidia.com`, which fails
with `401` on every batch if your key is scoped elsewhere. Deploy from a clean
shell:

```bash
env -i HOME="$HOME" PATH="$PATH" bash -c 'set -a; . ./.env.local-llm; set +a; docker compose up -d'
```

**Hosted endpoints rate-limit by IP.** Even with the LLM local, catalog search
embeds every query through a hosted endpoint (`TEXT_EMBED_BASE_URL`), and
guardrails add more hosted calls per turn when enabled. Sustained concurrency
trips a per-IP limit and returns HTTP 429, which surfaces as mass application
failure rather than as a rate limit. If load testing suddenly fails everywhere,
suspect that before your code.

**One remote hop stays on the critical path.** The same hosted embedding call
means a fully local deployment would be faster than what you measure here, so
your figures are a floor rather than a best case. `docker-compose-nim-local.yaml`
defines local NIMs for the embedding and safety models, but a 2-GPU box running a
120B model has no room left for them; everything local needs more GPUs.

**The model profile can change under you.** Left unset, the NIM selects a profile
by reading *free* GPU memory at startup, and this image also offers a BF16 profile
it considers runnable — so precision, and therefore every performance number, can
vary run to run depending on what happened to be resident. Pin it for the
duration of any measurement you intend to compare:

```bash
# `list-model-profiles` in the image shows the available set
NIM_MODEL_PROFILE=<id> docker compose -f docker-compose-nim-local.yaml up -d nemotron
```

**A deployment can wedge without anything noticing.** Under sustained concurrency
the NIM can stop dispatching while the container stays up, the process tree stays
intact and the weights stay resident — every endpoint including `/health` timing
out, with no crash, no OOM and no restart. The image ships **no `HEALTHCHECK`**,
so `docker inspect` reports `current health: not tracked`. If you run this under
load, add a healthcheck against `/health` and a restart policy, or a wedged
server will quietly absorb an entire test run.

**An empty dashboard has two causes that look identical.** Either no traffic
(counters are cumulative, panels show rates, idle reads zero) or the `model_name`
filter points at a model that no longer produces data — which happens whenever
you switch between a hand-built vLLM server and a NIM. Check the dropdown before
debugging anything else.

---

## Related documentation

- [`benchmarks/README.md`](../benchmarks/README.md) — a map of the measurement tools
- [`monitoring/README.md`](../monitoring/README.md) — the observability stack itself
- [`docs/OBSERVABILITY.md`](OBSERVABILITY.md) — OpenTelemetry and Phoenix span model
- [`docs/DEPLOYMENT.md`](DEPLOYMENT.md) — deploying the stack, health checks
- [`docs/API.md`](API.md) — the `metrics` stream event, `/query/timing`, token and model usage fields
