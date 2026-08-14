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

### The settings

| Variable | Default | Meaning |
|---|---|---|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | unset | Collector address. Unset means no tracing at all, from either producer. |
| `OTEL_SERVICE_NAME` | `chain-server` | Service name on every span. |
| `RELAY_ENABLED` | `false` | Attach Relay's middleware and register its subscriber. |
| `INSTALL_RELAY` | `false` | **Build arg.** Whether the image contains `nemo-relay`. |

`RELAY_ENABLED` without `INSTALL_RELAY` logs one warning and serves shoppers
untouched. `RELAY_ENABLED` without an OTLP endpoint logs one warning and exports
nothing. Neither is a failed turn, and each is said **once**, not once a turn --
the agent is rebuilt every turn, so a warning on that path would otherwise
repeat forever.

---

## Checking that Relay is doing its job

"Traces appeared" is not the same as "Relay is working": the LangChain
instrumentation produces spans on its own, so a broken Relay looks exactly like
a working one until you count. Four checks, cheapest first.

### 1. It started

```bash
docker logs chain-server | grep -i relay
```

| What you see | What it means |
|---|---|
| `Relay tracing enabled, exporting to …` | Subscriber registered. Go to check 2. |
| `nemo-relay is not installed` | Image built without `INSTALL_RELAY=true`. |
| `OTEL_EXPORTER_OTLP_ENDPOINT is not` | Flag on, nowhere to send. |
| `did not preserve our middleware order`<br>`dropped … from the agent arguments` | **A new Relay release changed the agent.** Tracing is off and the shop is running on its original arguments. Do not upgrade past this without reading the wrapper. |
| nothing at all | `RELAY_ENABLED` never reached the container. Check `docker compose config`. |

### 2. Relay's own spans arrive

Send one turn, wait for the batch, then count spans **by producer**. Relay's
spans are the ones named for the model and the `mark:` events; everything else
comes from the LangChain instrumentation.

```bash
curl -s -X POST localhost:8009/query/stream -H 'Content-Type: application/json' \
  -d '{"query":"show me black dresses in a size 2","user_id":770000111,
       "conversation_id":"relay-check"}' > /dev/null
sleep 15

curl -s 'localhost:6006/v1/projects/default/spans?limit=400' | python3 -c "
import sys, json
rows = json.load(sys.stdin)['data']
mine = [r for r in rows if r['attributes'].get('conversation.id') == 'relay-check']
turn = mine[0] if mine else None
relay = [r for r in rows if any(k.startswith('nemo_relay.') for k in r['attributes'])]
print('turn span found :', bool(turn))
print('relay spans     :', len(relay))
print('relay names     :', sorted({r['name'] for r in relay}))
"
```

Healthy output names the model and at least one `mark:`:

```
turn span found : True
relay spans     : 6
relay names     : ['azure/openai/gpt-5.2', 'mark:DeepAgents Skills Configured', ...]
```

Read `relay spans` as a total for the project, not for your turn: Relay's spans
carry no `conversation.id` to filter on, which is the limitation described
below. Roughly six accumulate per turn, so what matters is whether the number
**grew** — not what it equals.

**`relay spans: 0` while the `turn` span is present** is the failure that
matters: the app is exporting and Relay is not. The subscriber did not
register — re-read check 1.

### 3. It costs the shopper nothing

Relay must never sit on the turn's path. Compare latency with the flag on and
off, same query, three turns each:

```bash
for f in false true; do
  RELAY_ENABLED=$f docker compose up -d chain-server >/dev/null 2>&1
  until curl -sf localhost:8009/health >/dev/null; do sleep 2; done
  for i in 1 2 3; do
    curl -s -o /dev/null -w "RELAY_ENABLED=$f  %{time_total}s\n" \
      -X POST localhost:8009/query/stream -H 'Content-Type: application/json' \
      -d "{\"query\":\"show me black dresses\",\"user_id\":77000$i,
           \"conversation_id\":\"lat-$f-$i\"}"
  done
done
```

Turn time is dominated by model round trips and varies by seconds on its own, so
read this as "no new seconds", not as a benchmark. Exports are batched and
off-path; anything that looks like a consistent multi-second penalty is a bug.

### 4. The agent is unchanged

The point of the wrapper is that Relay cannot alter the agent. Prove it rather
than trust it:

```bash
python3 -m pytest tests/unit/chain_server/test_deepagents_observability.py \
  -k "Relay" -q
```

Eleven tests, covering: every argument survives, only `middleware` may change,
our middleware stays first, and any violation falls back to the untouched
arguments. Then the behavioural check — run a replay journey with the flag on
and confirm the assertions still pass:

```bash
RELAY_ENABLED=true python3 -m tests.evaluation.src.replay --label relay-on --only J13
```

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

**NeMo Relay** traces the agent runtime's own lifecycle: one span per LLM call
**named for the model** (`azure/openai/gpt-5.2`) carrying token counts and
finish reason, one per tool execution, and `mark:` events for runtime
milestones. It is framework-shaped rather than LangChain-shaped, which is the
point — it is the same vocabulary the Rust and Node surfaces emit.

Relay's spans are the ones carrying `nemo_relay.*` attributes, and that is the
only reliable way to tell them apart. In particular the spans named `model` and
`tools` are **not** Relay's — those are LangGraph's node names, traced by the
LangChain instrumentation, and they nest under the turn where Relay's do not.

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

### The limitation to know before you rely on it

**Relay's spans do not join the turn's trace.** Measured on `nemo-relay 0.7.3`,
one turn arrives in Phoenix as **seven traces**, not one:

| Trace | Spans |
|---|---|
| the turn | 18 — our `turn` span, `LangGraph`, `ChatOpenAI`, each middleware hook, each tool |
| ×3 | one each, `azure/openai/gpt-5.2` |
| ×2 | one each, a tool |
| ×1 | `mark:DeepAgents Skills Configured` |

Every Relay span is its own root, with `parent_id: None` and no
`conversation.id`. So the token counts Relay reports are real but **unattributed**
— you cannot see them in the context of the turn that spent them, and the trace
list grows about sevenfold. Over a twenty-turn journey that is roughly 140
entries where you wanted 20.

Count it yourself:

```bash
curl -s 'localhost:6006/v1/projects/default/spans?limit=400' | python3 -c "
import sys, json
rows = [r for r in json.load(sys.stdin)['data'] if r['start_time'] > '$(date -u +%Y-%m-%dT%H:%M --date '-5 min')']
print('traces for one turn:', len({r['context']['trace_id'] for r in rows}))
"
```

The cause is scope propagation inside LangGraph's task execution, not our
wiring: Relay offers `capture_propagation_context` and `propagate_scope_to_thread`
for exactly this, but applying them means wrapping LangGraph internals we do not
own. That is a worse trade than living with orphan spans, so this is recorded
rather than worked around. **If a later Relay release fixes it, that count drops
to 1 and this section goes away** — which is why the check above counts traces.

Until then, the turn-level story is told by our own `turn` span, and Relay's
value here is per-call model detail rather than a unified trace.

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
