# Monitoring

Prometheus + Grafana for the local LLM NIM, plus a per-container resource
exporter. This directory is the **observability stack** only — the tools that
generate load and take measurements live in
[`benchmarks/`](../benchmarks/README.md).

See [`docs/PERFORMANCE.md`](../docs/PERFORMANCE.md) for how to measure this
application and how to read what comes back. This file covers only how the
pieces here fit together.

## Prerequisites

The app must be running with a **local NIM**
(`docker-compose-nim-local.yaml`), since these tools read metrics that a NIM
exposes and a hosted endpoint does not.

## Usage

```bash
./dashboard.sh up        # start the stack, wait for a healthy scrape, print URLs
./dashboard.sh status    # containers, scrape target health, live metric sample
./dashboard.sh urls      # URLs plus the SSH tunnel command for remote access
./dashboard.sh logs      # follow Prometheus and Grafana logs
./dashboard.sh down      # stop (add --volumes to discard history)
```

Needs nothing beyond Python 3 and Docker.

To put traffic through the system so the panels have something to show, see
[`benchmarks/`](../benchmarks/README.md) — `bench.sh load` is the quickest.

## How it is wired

Both containers join the app's `shopping-network` as an external network rather
than using host networking. That lets Prometheus scrape the NIM by service name
at `nemotron:8000` with no host ports involved, and avoids the port collisions
host networking caused (the app's nginx owns 3000, Phoenix owns 6006).

Host bindings are loopback-only, and Grafana defaults to **6005** to stay clear
of both:

| | Port | Override |
|---|---|---|
| Grafana | 6005 | `GRAFANA_PORT` |
| Prometheus | 9090 | — |

Grafana provisions its datasource and dashboards from `grafana/provisioning/`,
so there is nothing to configure by hand and no state worth preserving;
deleting the volumes loses only Prometheus history.

## What gets scraped

| Job | Source | Why it is separate |
|---|---|---|
| `vllm` | the NIM's `/metrics` | engine counters: tokens, queue depth, KV usage, preemptions |
| `dcgm` | `dcgm-exporter` | GPU utilisation, memory, temperature, power |
| `containers` | `docker_stats_exporter.py` | per-container CPU and memory |

The third exists because a run limited by Milvus, the chain server or the
retriever is otherwise indistinguishable from one limited by the GPUs, and
those call for opposite responses.

`dcgm-exporter` is pinned to a **1-second** collection interval. Its 30-second
default is longer than many benchmark runs and far longer than a prefill burst,
so the same stale idle sample gets scraped repeatedly and utilisation reads zero
through work that plainly happened.

## Files

| Path | Purpose |
|---|---|
| `dashboard.sh` | Entry point for the stack: `up`, `down`, `status`, `urls`, `logs` |
| `docker-compose.yaml` | Prometheus, Grafana and the exporters, joined to `shopping-network` |
| `prometheus.yaml` | Scrape config for the three jobs above |
| `grafana/dashboards/vllm-official.json` | vLLM's published dashboard, extended with CPU, GPU and prefix-cache panels |
| `grafana/provisioning/` | Datasource and dashboard auto-provisioning |
| `docker_stats_exporter.py` | Per-container CPU and memory, read from the Docker Engine API |

### Why a custom exporter rather than cAdvisor

cAdvisor is the usual choice and was tried first. On a host using the
`overlayfs` storage driver it could not resolve container metadata, reporting
container IDs instead of service names across three versions — which makes the
per-container series unusable for attributing a bottleneck to a service.
`docker_stats_exporter.py` reads the Engine API directly, needs no privileged
container, and mounts nothing but the socket.

## Gotchas

**An idle dashboard is indistinguishable from a broken one.** Panels show rates
over cumulative counters, so with no traffic everything reads zero. Put some
load through it before concluding anything is wrong.

**The dashboard filters on `model_name`.** Serve a different model and the saved
filter can point at a name that produces no data, leaving every panel silently
blank. Check the dropdown at the top.

**`vllm:kv_cache_usage_perc` is a gauge, not a counter.** It reads 0 at idle and
is only meaningful when sampled during load, which is why the sweep in
`benchmarks/` polls it twice a second rather than reading it afterwards.

**A missing exporter looks like a real zero.** If a job is `down`, the panels it
feeds draw nothing rather than erroring. Check `./dashboard.sh status` lists all
three targets as healthy before trusting a measurement.
