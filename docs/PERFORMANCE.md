# Performance

Measuring this application answers three questions: how long a shopper waits
before words start appearing, how many shoppers you can serve at once before
that wait becomes unacceptable, and — when it is slow — whether the model or
your own code is responsible.

This page carries only what is specific to *this* application. General
benchmarking method and the guided run itself live elsewhere:

| For | Go to |
| --- | --- |
| Actually measuring it, step by step | [Notebook 5: Performance Measurement](../notebook/5_Performance_Measurement.ipynb) |
| What the metrics mean — time to first token, inter-token latency, throughput, percentiles | [NVIDIA LLM benchmarking: Metrics](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html) |
| How to design a benchmark at all | [NVIDIA LLM benchmarking: Overview](https://docs.nvidia.com/nim/benchmarking/llm/latest/overview.html) |
| Standing up Prometheus and Grafana | [`monitoring/`](../monitoring/README.md) |
| Traces for a single turn | [Observability](OBSERVABILITY.md) |

Two constraints before anything else.

**You need locally hosted models.** A hosted endpoint publishes no metrics and
rate-limits sustained load, so there is nothing to measure and no way to
measure it. See
[Locally Hosted Models](DEPLOYMENT.md#-locally-hosted-models).

**This document deliberately contains no measured results.** Latency and
throughput depend on the GPU, the model, the precision, the catalog and the
profile in use, so a figure recorded here would be read as a target on hardware
it never described. Measure your own.

## Why this workload is not a standard benchmark

A shopper turn is **heavily prefill-dominated**. Every model call carries the
system instructions, all tool definitions, the conversation so far and any
catalog results, while the reply is a few sentences — and a single turn makes
several such calls.

Published LLM benchmarks typically use a 1:1 input-to-output ratio. Those
numbers will not predict this application, because a workload that reads far
more than it writes reaches its limits somewhere completely different. The gap
is routinely a multiple, not a few percent.

So take the real ratio from your own deployment before sweeping anything, and
feed that shape into the sweep. A sweep at a tool's default shape produces
confident numbers describing a workload you do not run.

The practical consequence for capacity planning: **a capacity number is
meaningless without a latency target attached.** Running more requests at once
finishes more work per second while making each one slower, so "how many
shoppers" is a choice of point on that curve, not a property of the hardware.
Size against time to first token rather than total turn time — on a streaming
interface, that is what the shopper actually experiences.

## Before you measure anything

Measuring the wrong thing is worse than not measuring. Every wrong conclusion
in this area starts with a deployment that was not doing what the measurer
assumed. Confirm all six.

```bash
# 1. The local LLM is serving, at the precision you expect
curl -s http://localhost:8000/v1/models | python3 -m json.tool
docker logs retail-shopping-assistant-local-llm-1 2>&1 | grep -m1 -oE "dtype=[a-z0-9.]+"
#    want: dtype=torch.bfloat16 for the default BF16 checkpoint

# 2. The catalog is indexed -- an empty one yields confident nonsense
curl -s http://localhost:8010/ready
#    want: {"status":"ready","catalog_id":"fashion_products","products":<n>}

# 3. The app is really using the local LLM. Count requests either side of a
#    query and check the delta equals the reported model_calls.
before=$(docker logs retail-shopping-assistant-local-llm-1 2>&1 | grep -c "POST /v1/chat/completions")
curl -s -X POST http://localhost:8009/query/timing -H 'Content-Type: application/json' \
  -d '{"user_id":1,"query":"Show me black dresses under $100","session_id":"smoke-1"}' \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['token_usage'])"
after=$(docker logs retail-shopping-assistant-local-llm-1 2>&1 | grep -c "POST /v1/chat/completions")
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

Check 5 needs `EXPOSE_AGENT_DIAGNOSTICS=true`, or the breakdown comes back
empty.

## Pitfalls that produce confident, wrong numbers

Every one of these has invalidated real measurements on this stack.

**A dead model looks like a fast one.** When a model call fails the application
does not error — it returns canned text such as "I could not complete that
shopping request." Those turns finish fast, so throughput climbs and latency
drops, and the run looks like the best you have ever recorded. Confirm replies
contain real products before believing any number. This is the single most
expensive mistake available here.

**The tool-call parser silently dropping calls.** `pythonic` cannot read this
model's XML-form tool calls, so vLLM returns them as plain assistant text with
no `tool_calls` and the agent answers without ever searching. Every latency
number measured in that state describes a much cheaper application that never
used its tools. Check `agent_diagnostics.tool_calls` is non-empty before
believing anything. Use `qwen3_coder`.

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
nothing unless sampled during load. Poll it while traffic is running rather
than reading it afterwards.

**`vllm:num_requests_waiting` answers "is it the model or my code?"** Tokens in
and out, time to first token, queue depth and KV-cache utilisation all describe
the engine's work, but waiting sequences are the one series that separates a
saturated model from a slow application. A hosted endpoint cannot answer that
question at all, which is the strongest reason to serve the model yourself while
tuning.

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
env -i HOME="$HOME" PATH="$PATH" bash -c 'set -a; . ./.env.local-models; set +a; docker compose up -d'
```

**Hosted endpoints rate-limit by IP.** Guardrails add hosted calls per turn
when enabled, and if you keep text embedding hosted (the comment in
`.env.local-models.example` shows how), catalog search embeds every query
through `TEXT_EMBED_BASE_URL` too. Sustained concurrency trips a per-IP limit
and returns HTTP 429, which surfaces as mass application failure rather than as
a rate limit. If load testing suddenly fails everywhere, suspect that before
your code.

**Decide what the comparison is.** With text embedding local, a run differs
from a hosted `.env.example` run in two models, not one. To attribute a
difference to the model serving the shopper alone, keep embedding hosted; the
hosted hop then stays on the critical path, so figures are a floor rather than
a best case.

**Speculative input execution trades blocked-turn cost for latency.** With
`GUARDRAILS_SPECULATIVE_MAIN_MODEL_ENABLED=true`, a guarded text-only turn starts
its first app-model step while the parallel input rails run. Every tool remains
behind the input allow decision, and media remains sequential. An allowed turn
can save at most the overlap between input-rail latency and that first model
step. A blocked turn runs no tool, but its already-started model request can
still be billed even after cancellation. Keep the default `false` unless that
cost tradeoff is intentional, and compare allowed and blocked traffic mixes as
well as latency percentiles.

**A deployment can wedge without anything noticing.** Under sustained concurrency
the model server can stop dispatching while the container stays up, the process tree stays
intact and the weights stay resident — every endpoint including `/health` timing
out, with no crash, no OOM and no restart. `local-llm` has a healthcheck against
`/health`, so `docker inspect` reports it unhealthy, but `restart: "no"` means
nothing acts on that. Watch the health status during a run, or a wedged server
will quietly absorb an entire test run.

**An empty dashboard has two causes that look identical.** Either no traffic
(counters are cumulative, panels show rates, idle reads zero) or the `model_name`
filter points at a model that no longer produces data — which happens whenever
you switch the served model name. Check the dropdown before
debugging anything else.

---

## Related documentation

- [Notebook 5: Performance Measurement](../notebook/5_Performance_Measurement.ipynb) — the guided run
- [`monitoring/README.md`](../monitoring/README.md) — the observability stack itself
- [Observability](OBSERVABILITY.md) — OpenTelemetry and Phoenix span model
- [Deployment](DEPLOYMENT.md) — deploying the stack, health checks
- [API Reference](API.md) — the `metrics` stream event, `/query/timing`, token and model usage fields
