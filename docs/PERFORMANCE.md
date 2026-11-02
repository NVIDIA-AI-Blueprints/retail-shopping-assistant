# Performance

Measuring this application answers three questions: how long a shopper waits
before words start appearing, how many shoppers you can serve at once before
that wait becomes unacceptable, and — when it is slow — whether the model or
your own code is responsible.

**[Notebook 5: Performance Measurement](../notebook/5_Performance_Measurement.ipynb)
is how you measure it.** It walks one deployment end to end: check the server,
load a trace, smoke test, run the matrix, read the results.

This page carries only the handful of things the notebook assumes you know.

| For | Go to |
| --- | --- |
| What the metrics mean — time to first token, inter-token latency, throughput, percentiles | [NVIDIA LLM benchmarking: Metrics](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html) |
| How to design a benchmark | [NVIDIA LLM benchmarking: Overview](https://docs.nvidia.com/nim/benchmarking/llm/latest/overview.html) |
| Prometheus, Grafana, and reading the panels | [`monitoring/`](../monitoring/README.md) |
| Traces for a single turn | [Observability](OBSERVABILITY.md) |

No measured results appear here. Latency and throughput depend on the GPU, the
model, the precision, the catalog and the profile in use, so a figure recorded
here would be read as a target on hardware it never described. Reference
results will be published separately.

## You need locally hosted models

A hosted endpoint publishes no metrics and rate-limits sustained load, so there
is nothing to measure and no way to measure it. Set up
[Locally Hosted Models](DEPLOYMENT.md#-locally-hosted-models) first.

## Generic benchmark numbers will not predict this

A shopper turn is **heavily prefill-dominated**. Every model call carries the
system instructions, all tool definitions, the conversation so far and any
catalog results, while the reply is a few sentences — and one turn makes
several such calls.

Published LLM benchmarks typically use a 1:1 input-to-output ratio, and a
workload that reads far more than it writes reaches its limits somewhere
completely different. The gap is routinely a multiple, not a few percent. Take
the real ratio from your own deployment before sweeping anything.

The same asymmetry is why **a capacity number is meaningless without a latency
target attached**: running more requests at once finishes more work per second
while making each one slower. Size against total turn time: the reply text
arrives when the turn finishes, since the stream carries progress frames rather
than tokens.

## Traps specific to this application

**A dead model looks like a fast one.** When a model call fails the application
does not error — it returns canned text such as "I could not complete that
shopping request." Those turns finish fast, so throughput climbs and latency
drops, and the run looks like the best you have ever recorded. Confirm replies
contain real products before believing any number. This is the single most
expensive mistake available here.

**`vllm:num_requests_waiting` is the one series that answers "model or code?"**
Everything else describes the engine's work; waiting sequences separate a
saturated model from a slow application. A hosted endpoint cannot answer that
question at all.

**Env profiles blend rather than switch.** Each `.env.*` file is sourced and
uses `${VAR:-default}`, so an already-exported value wins and sourcing two
profiles in one shell silently mixes them. A model URL or feature flag leaking
in from another profile changes per-turn latency, so runs you meant to compare
are not comparable. Measure from a clean shell:

```bash
env -i HOME="$HOME" PATH="$PATH" bash -c 'set -a; . ./.env.local-models; set +a; docker compose up -d'
```

**A wedged model server stays up.** Under sustained concurrency vLLM can stop
dispatching while the container runs and the weights stay resident, with every
endpoint including `/health` timing out. `local-llm` has a healthcheck, but
`restart: "no"` means nothing acts on it. Watch container health during a run.

## Related documentation

- [Notebook 5: Performance Measurement](../notebook/5_Performance_Measurement.ipynb) — the guided run
- [`monitoring/README.md`](../monitoring/README.md) — the observability stack and its gotchas
- [Observability](OBSERVABILITY.md) — OpenTelemetry and Phoenix span model
- [Deployment](DEPLOYMENT.md) — deploying the stack, health checks
- [API Reference](API.md) — the `metrics` stream event, `/query/timing`, token and model usage fields
