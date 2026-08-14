<!-- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Observability

Two producers send traces to one collector, and the collector decides where they
land. Both are **off by default**: absent an endpoint, no provider is built, no
exporter is created and nothing is instrumented.

```
chain-server ──┬── openinference-langchain ──┐
               │                             ├── otel-collector ── Phoenix
               └── NeMo Relay subscriber ────┘        :4318          :6006
```

The app never names a backend. Changing where traces land is a change to
`otel-collector-config.yaml`, not to any service.

---

## Running it

```bash
docker compose up -d otel-collector phoenix

OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318 \
  docker compose up -d chain-server

# Phoenix
open http://localhost:6006
```

That gives you the LangChain/LangGraph traces. **Relay needs two more things**: a
flag, and an image built with the package.

```bash
INSTALL_RELAY=true docker compose build chain-server

INSTALL_RELAY=true RELAY_ENABLED=true \
  OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318 \
  docker compose up -d chain-server

docker logs chain-server | grep -i relay
# → Relay tracing enabled, exporting to http://otel-collector:4318
```

Then send a turn and look:

```bash
curl -s -X POST localhost:8009/query/stream -H 'Content-Type: application/json' \
  -d '{"query":"show me black dresses in a size 2","user_id":770000111,
       "conversation_id":"relay-smoke-1"}' > /dev/null

# spans arrive batched; give it ~15s
curl -s 'localhost:6006/v1/projects/default/spans?limit=400' \
  | python3 -c "import sys,json;print(sorted({r['name'] for r in json.load(sys.stdin)['data']}))"
```

One turn produces around two dozen spans. Relay's are `turn`-adjacent scopes
(`model`, `tools`), one span per LLM call named for the model, and `mark:` spans
for lifecycle events.

### The settings

| Variable | Default | Meaning |
|---|---|---|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | unset | Collector address. Unset means no tracing at all, from either producer. |
| `OTEL_SERVICE_NAME` | `chain-server` | Service name on every span. |
| `RELAY_ENABLED` | `false` | Attach Relay's middleware and register its subscriber. |
| `INSTALL_RELAY` | `false` | **Build arg.** Whether the image contains `nemo-relay`. |

`RELAY_ENABLED` without `INSTALL_RELAY` logs one warning and serves shoppers
untouched. `RELAY_ENABLED` without an OTLP endpoint logs one warning and exports
nothing. Neither is a failed turn.

---

## Why the Relay dependency is a build arg

`nemo-relay[deepagents]` requires `langgraph>=1.2.9`; `requirements.txt` pins
`1.2.7`. Installing Relay therefore **moves the agent runtime**, and the image
that ships should be the image that was tested. So the default image carries the
tested pin, and `INSTALL_RELAY=true` produces a second image that carries
`langgraph 1.2.11` alongside `nemo-relay 0.7.3`.

The gap is four patch releases inside `1.2.x`, and the unit suite has been
running against `1.2.11` in the local venv all along. That is evidence for the
bump, not proof of it. Raising the pin is a decision to take on its own
evidence.

---

## What each producer actually gives you

They overlap, and the overlap is not wasted — they disagree usefully.

**`openinference-instrumentation-langchain`** traces the LangChain object graph:
`ChatOpenAI`, `LangGraph`, every middleware hook (`TodoListMiddleware.after_model`,
`PatchToolCallsMiddleware.before_agent`), and each tool by name. This is the
faithful record of *what the framework did*.

**NeMo Relay** traces the agent runtime's own lifecycle: scopes for the turn, the
model call and the tool batch; a span per LLM call named for the model
(`azure/openai/gpt-5.2`) carrying token counts and finish reason; and `mark:`
events for runtime milestones. It is framework-shaped rather than
LangChain-shaped, which is the point — it is the same vocabulary the Rust and
Node surfaces emit.

**The `turn` span is ours**, not either library's. It carries the per-turn
diagnostics the rest of this repo is built around:

```
conversation.id                  relay-smoke-1
metadata.termination_reason      completed
metadata.tool_calls              2
metadata.tool_calls_rejected     0
metadata.products_shown          4
metadata.zero_result_scopes      0
metadata.skills                  ["/shopper/product-discovery/SKILL.md"]
metadata.tools                   ["activate_shopper_skills_tool", "search_catalog_tool"]
metadata.diagnostics_json        {...}
```

`conversation.id` is the session key, so a Phoenix session and a durable
conversation are the same set of turns by construction.

### What Relay does not tell you here

Its DeepAgents integration reports skills from a `skills=` keyword argument to
`create_deep_agent`. **This service does not pass one** — skills reach the agent
through the filesystem backend and a gate middleware — so Relay's skill mark
arrives with an empty skill list and `nemo_relay.mark.orphan: true`:

```
mark:DeepAgents Skills Configured
  nemo_relay.mark.metadata.deepagents_kind   skill
  nemo_relay.mark.metadata.phase             configured
  nemo_relay.mark.data.subagents             []
  nemo_relay.mark.data.backend               FilesystemBackend
  nemo_relay.mark.orphan                     true
```

It says skills were configured, once, at agent construction. It does not say
which skill fired on a given turn. **`metadata.skills` on the `turn` span is
still the answer to that**, and remains so.

---

## The guarantee that matters

An observability layer that can quietly change the agent is worse than no
tracing. `add_nemo_relay_integration` is handed the keyword arguments for
`create_deep_agent` and returns them modified, so the runtime **checks the
result rather than trusting it**:

- every argument that went in must come back
- no argument other than `middleware` may be replaced
- our middleware must still come first, so the tool-loop control and the skill
  gate run ahead of anything Relay adds

If any of those fail the runtime logs it and builds the agent from the arguments
it started with. Tracing degrades; the shop does not change.

Verified against `nemo-relay 0.7.3`: all six arguments survive and Relay's
middleware is *appended* to ours.

---

## Retention

Phoenix holds prompts, completions and tool transcripts — more than
`memory_retriever` keeps by design, and it persists in the `phoenix-data`
volume. It is published on all interfaces so a tunnel can reach it; on anything
other than a development box that wants to go back to loopback with access
through the tunnel only.

There is no backfill. Only live turns are captured, so a replayed transcript
correctly shows no LLM spans.
