<!-- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Observability

How to watch a shopper's conversation, and how to find out why a turn did what
it did.

---

## The model: session, turn, span

Three levels, and they map onto things you already know:

| | is | keyed by |
|---|---|---|
| **session** | one conversation | `session.id` — the `conversation_id` the client sends |
| **trace** | one turn | one per request |
| **span** | one step inside a turn | model call, tool call, middleware hook |

A Phoenix session and a durable conversation in `memory_retriever` are the same
set of turns **by construction**, because both are grouped by
`conversation_id`. That is the property that makes a trace worth reading: what
you see in Phoenix is what the shopper actually had.

Traces go to an OpenTelemetry collector, and the collector decides where they
land. The app never names a backend.

```
chain-server ──┬── our turn span ─────────────┐
               ├── openinference-langchain ───┼── otel-collector ── Phoenix
               └── NeMo Relay (optional) ─────┘        :4318          :6006
```

---

## What Relay gives you, and what it costs

Relay is the third and optional producer. The first two — the `turn` span this
service writes, and `openinference-instrumentation-langchain` — are always on
and answer *what happened*. Relay answers *what the agent was thinking while it
happened*: the prompt as sent, the completion as returned, which skill was
selected, which subagent ran, and every tool call with its arguments and result.

The practical difference, on a turn that went wrong:

| question | without Relay | with Relay |
|---|---|---|
| Which tools ran, in what order | yes | yes |
| How long each took | yes | yes |
| **What arguments the tool was called with** | no | **yes** |
| **What the tool returned** | no | **yes** |
| **The exact prompt the model saw** | no | **yes** |
| **Which skill the agent selected and why** | partly | **yes** |

That is the difference between "the search returned nothing" and "the search was
asked for `taxonomy_level_2=dresses` with `primary_color=red`, and the catalogue
has no red dress".

**What it costs.** It is not free and it is not merely additive:

- It sees prompts, completions and cart contents. That is shopper data leaving
  the process; it belongs in a backend you control.
- Installing it moves the agent runtime. `nemo-relay[deepagents]` requires
  `langgraph>=1.2.9` and `requirements.txt` pins `1.2.7`, which is why it is a
  build argument rather than a normal dependency.
- It adds a second OTLP exporter to the process.

**Leave it off for normal running.** Turn it on to study a specific problem,
then turn it off. It is not a monitoring tool — it produces one detailed trace
per turn, not metrics.

### Worked example: a shopper says the assistant found nothing

The shopper asked for red dresses under $100 and was told the shop has none.
Is that true, or did the search go wrong?

**1. Turn Relay on and reproduce the turn.**

```bash
INSTALL_RELAY=true docker compose build chain-server
INSTALL_RELAY=true RELAY_ENABLED=true \
  OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318 \
  docker compose up -d chain-server
docker logs chain-server | grep -i relay
# → Relay tracing enabled, exporting to http://otel-collector:4318
```

**2. Open Phoenix** at `http://localhost:6006` and find the conversation. Each
turn is one `turn` span; Relay's events nest under it, so the whole turn reads
top to bottom in one place.

**3. Open the `search_catalog_tool` call** and read its arguments. This is the
thing you cannot see any other way. You are looking for the difference between
three quite different failures:

- The arguments match what the shopper asked for, and the result is genuinely
  empty → the catalogue has no red dress. The assistant was right.
- The arguments carry a constraint the shopper never gave → the agent invented
  one, and the empty result is its own doing.
- The arguments are right, the result has products, and the reply says
  otherwise → a grounding failure between the tool and the answer.

**4. Turn it back off** when you have your answer:

```bash
docker compose build chain-server && docker compose up -d chain-server
```

### A defect this found, and what it means for trusting a trace

Enabling Relay used to change the agent's behaviour. Its tool wrapper did not
pass the model's arguments to the tool: it encoded them for the span and handed
the tool the decoded copy, and the copy was not the original. The codec tries
`model_dump()` first, which materialises every optional field that was never
set, so a search sent as:

```json
{"requested_product_type": "dress"}
```

arrived at the tool as:

```json
{"requested_product_type": "dress", "price": {"min": null, "max": null}}
```

This service rejects constraints the shopper never gave, so the tool refused its
own call and the turn answered *"I couldn't complete a valid catalog search"*.
Measured on journey J01: two failures with tracing on, five passes with it off,
on the same image, differing only by `RELAY_ENABLED`.

`_relay_must_not_rewrite_arguments` in `deepagents_runtime.py` is the fix. Relay
still records the call; the tool now receives what the model sent. The property
is held by tests, not by care.

Worth stating plainly because it is the general lesson: **an observability layer
that can rewrite what it observes can change the outcome it is reporting on.**
If a turn behaves differently with tracing on, the trace is evidence about a
system that no longer exists. Reproduce with tracing off before believing a
finding, and that is cheap here — one build argument apart.

## Turning it on

```bash
docker compose up -d otel-collector phoenix

OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318 \
  docker compose up -d chain-server

open http://localhost:6006
```

Absent `OTEL_EXPORTER_OTLP_ENDPOINT` nothing is built, exported or
instrumented — that is the default, and it is a real off switch rather than a
quiet no-op.

NeMo Relay is separate and optional; see [Adding NeMo Relay](#adding-nemo-relay).

| Variable | Default | Meaning |
|---|---|---|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | unset | Collector address. Unset means no tracing at all. |
| `OTEL_SERVICE_NAME` | `chain-server` | Service name on every span. |
| `RELAY_ENABLED` | `false` | Also emit NeMo Relay's events. |
| `INSTALL_RELAY` | `false` | **Build arg.** Whether the image contains `nemo-relay`. |

---

## Studying a session

Start here. Most questions — "why did it add the wrong dress", "when did it stop
showing products" — are questions about a *conversation*, not a turn.

### List what you have

```bash
python3 scripts/read_session.py
```

```
  13 turns  demo13-run8
   8 turns  demo20-run11
   3 turns  session-demo
```

### Read one

```bash
python3 scripts/read_session.py demo20-run11
python3 scripts/read_session.py demo20-run11 --replies   # include the answers
```

```
demo20-run11 — 8 turns

1. "show me black dresses in a size 2"
   skill  /shopper/product-discovery/SKILL.md
   tools  activate_shopper_skills_tool, search_catalog_tool
   ->     4 shown, 0 rejected, completed

2. "add the first one to my cart"
   skill  /shopper/cart-management/SKILL.md
   tools  activate_shopper_skills_tool, resolve_conversation_products_tool, add_cart_items_tool
   ->     0 shown, 0 rejected, completed
```

In the Phoenix UI the same thing is the **Sessions** tab: search the
conversation id and the turns are listed in order.

### What to look for, and what it means

Each line is read off the `turn` span, and each field answers a recurring
question:

| Field | Reading it |
|---|---|
| `skill` | **Which skill fired.** A cart question answered under `product-discovery` explains a turn that searched instead of resolving. The commonest root cause in this system is a rule sitting in a skill that never loaded. |
| `tools` | What actually ran. `resolve_conversation_products_tool` absent on an "add the blue one" turn means it searched rather than resolved — a different failure from resolving wrongly. |
| `shown` | Products streamed. `0` on a discovery turn is either an honest "we don't carry that" or a silent failure; the reply tells you which. |
| `rejected` | Tool calls the server refused. Non-zero is not necessarily bad — it is the loop control and the skill gate doing their job — but a turn with several is a turn where the model kept trying something it was not allowed to do. |
| `completed` | Termination reason. Anything else (`timeout`, `max_iterations`) means the reply you are reading is partial. |

### Compare two runs of the same script

The most useful thing this enables. Replay the same journey twice and diff the
shape rather than the prose:

```bash
diff <(python3 scripts/read_session.py demo20-run11) \
     <(python3 scripts/read_session.py demo20-run12)
```

A turn that changed which skill it loaded, or stopped calling the resolver, is
visible immediately — where reading two transcripts side by side is not.

---

## Digging into one turn

When the session view tells you *which* turn went wrong, open that turn's trace.
In Phoenix, click the turn; every span below it is one step.

### The shape of a turn

This is a real three-tool turn, exactly as it appears:

```
turn                                        ← ours: the whole turn, and its diagnostics
  LangGraph                                 ← the agent graph
    PatchToolCallsMiddleware.before_agent
    model  →  ChatOpenAI                    ← round 1: the model decides
    tools  →  activate_shopper_skills_tool  ← round 1: it loads a skill
    model  →  ChatOpenAI                    ← round 2
    tools  →  resolve_conversation_products_tool
    model  →  ChatOpenAI                    ← round 3
    tools  →  add_cart_items_tool
    model  →  ChatOpenAI                    ← round 4: it writes the reply
```

The `model → tools` pairs are the agent loop. **Counting them is the fastest
read of a turn**: four model calls to add one product is normal; nine means it
was struggling, and the tool results between them say why.

### The four questions, and where each is answered

**"Which skill was loaded, and what did the agent do?"** — the `turn` span:

```
conversation.id                  demo20-run11
session.id                       demo20-run11
metadata.skills                  ["/shopper/cart-management/SKILL.md"]
metadata.tools                   ["activate_shopper_skills_tool", "add_cart_items_tool"]
metadata.tool_calls              3
metadata.tool_calls_rejected     0
metadata.products_shown          0
metadata.zero_result_scopes      0
metadata.termination_reason      completed
metadata.diagnostics_json        {...}
```

`metadata.diagnostics_json` is the full record: every tool call in order, with
its arguments, its status, and — when it was refused — the reason. When a turn
is confusing, this is the thing to read.

**"What was the model actually told?"** — any `ChatOpenAI` span. It carries the
whole conversation as sent: `llm.input_messages.N.message.role` and
`.content`, including the system prompt at index 0 and every tool result the
model had in front of it. This is how you tell "the model ignored the rule" from
"the model never saw the rule".

**"What did a tool return?"** — the tool span, `input.value` and `output.value`.
For `search_catalog_tool` that is the scopes it searched and the products that
came back; for `add_cart_items_tool`, exactly what went into the cart.

**"Why was a tool call refused?"** — the rejected call is in
`metadata.diagnostics_json` with a `rejection_reason`
(`duplicate_catalog_scope`, `skill_activation_required`, `repair_scope_changed`).
The refusal is deliberate: those gates exist so a wrong tool call cannot become
a wrong answer.

### Worked example: "it added the wrong product"

1. `read_session.py <convo>` — find the turn, and check `tools`. Is
   `resolve_conversation_products_tool` there?
2. **Absent** → it never tried to resolve a reference; it searched. The question
   becomes why the reference was not recognised, and the `ChatOpenAI` span for
   that round shows what it had to work with.
3. **Present** → open its span. `input.value` is the descriptor the model sent;
   `output.value` is what came back. If the descriptor is wrong, the model
   misread the shopper. If the descriptor is right and the result is wrong, the
   resolver is at fault.
4. Then `add_cart_items_tool`'s `input.value` says what was actually added, and
   whether the size came from the shopper or from nowhere.

That sequence separates *the model chose badly* from *the system resolved
badly* — different bugs with different fixes, and impossible to tell apart from
the reply alone.

### From the command line

Phoenix's UI is usually faster, but a trace can be pulled whole:

```bash
curl -s 'localhost:6006/v1/projects/default/spans?limit=1000' | python3 -c "
import sys, json
SESSION, TURN = 'demo20-run11', 2
rows = json.load(sys.stdin)['data']
turns = sorted((r for r in rows
                if r['attributes'].get('session.id') == SESSION and r['name'] == 'turn'),
               key=lambda r: r['start_time'])
trace = turns[TURN - 1]['context']['trace_id']
for r in sorted((x for x in rows if x['context']['trace_id'] == trace),
                key=lambda x: x['start_time']):
    print(f\"{r['span_kind']:6} {r['name']}\")
"
```

Phoenix caps `limit` at 1000 and rejects anything higher rather than clamping.

---

## What each producer gives you

Three, and only the first is required.

### 1. The `turn` span — ours

About ten lines of `opentelemetry-sdk`, no framework. It carries `session.id`
and the per-turn diagnostics above. **This is what makes sessions work**, and
what `read_session.py` and the Sessions tab read. Neither of the other two
producers is needed for it.

### 2. `openinference-instrumentation-langchain`

Traces the LangChain object graph: `LangGraph`, `model`, `tools`, `ChatOpenAI`,
each middleware hook, each tool by name — nested under the turn. This is what
makes *digging in* possible: the prompts, the tool arguments and results, and
the loop structure all come from here.

Drop it and sessions still work; you lose the inside of every turn.

### 3. NeMo Relay — optional

Per-call detail from the agent runtime: one span per LLM call named for the
model, carrying token counts and finish reason, plus per-tool spans and `mark:`
lifecycle events. See [Adding NeMo Relay](#adding-nemo-relay) for what it does
and does not do on this architecture.

---

## The design, and the decision that is open

### Why three producers and not one

Not by design — by history. The `turn` span was written for this system's own
diagnostics; the LangChain instrumentation came free with the framework; Relay
was added to evaluate it. They overlap, and the overlap is mostly harmless
because they answer different questions.

The uncomfortable part is that **Relay is an agent runtime being used as a
tracing library**, which is not what it is for.

### What Relay actually is

A Rust runtime with Python, Node and Rust bindings. Its unit is the **scope**:
open one, and everything inside nests under it and inherits its causality.
Around that it provides lifecycle events, middleware, and exporters.

Relay's DeepAgents integration hooks exactly one seam — the middleware. Our
model and tool calls execute inside LangGraph, **outside Relay's scope stack**,
so Relay sees isolated calls and cannot tell they belong to a turn, let alone to
a conversation. That is not a defect in Relay; nothing told it.

### The runtime-native alternative, and the evidence for it

Relay's own model fits this problem exactly: **a scope can be the session**, and
it survives across turns. Verified on 0.7.3:

```
session   trace 01a0024ca0e47c  parent ROOT          session=sess-proof
turn-1    trace 01a0024ca0e47c  parent b498b6eeb4ce
turn-2    trace 01a0024ca0e47c  parent b498b6eeb4ce   ← a fresh context, same trace
```

`capture_propagation_context()` returns a small JSON token
(`{"version":1,"parent_uuid":"…"}`); `create_scope_stack_from_propagation()`
rehydrates it in a later request. The root scope had already **closed** before
turn 2 ran — causality is by uuid, not by an open span — so there is no dangling
root to manage.

The shape that follows:

1. First turn of a conversation opens a `session` scope and captures the token.
2. The token is stored against the conversation — `memory_retriever` is already
   keyed by `conversation_id`.
3. Every later turn rehydrates it and opens a `turn` scope inside.
4. Model and tool calls are recorded through `nemo_relay.llm.call` and
   `nemo_relay.tools.call`.

That yields **one trace per conversation**, everything nested, session
attributes on the root — and both other producers become droppable.

**What it costs:** instrumenting our own boundaries rather than setting a flag,
and owning that instrumentation. Roughly a day including tests.

**What is unverified**, and would be checked first:

- **Concurrent conversations.** Rehydrating a stack per request must not leak
  between shoppers. `use_scope_stack` is context-local so it should hold, but
  that wants proving with parallel turns, not assuming.
- **Whether Phoenix groups on it.** Its Sessions view keys on a bare
  `session.id`; Relay emits `metadata.session.id`. `attribute_mappings` did not
  remap it on 0.7.3 — tested twice, and it silently dropped the `inherit` field.
  One clean trace per conversation may be better than a session list, but the
  Sessions tab specifically could stay empty.

**Status: not decided.** What ships today is the three-producer arrangement
above, with Relay off by default.

---

## Adding NeMo Relay

Optional, off by default, and it needs two things: a flag, and an image built
with the package.

```bash
INSTALL_RELAY=true docker compose build chain-server

INSTALL_RELAY=true RELAY_ENABLED=true \
  OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318 \
  docker compose up -d chain-server

docker logs chain-server | grep -i relay
# → Relay tracing enabled, exporting to http://otel-collector:4318
```

`RELAY_ENABLED` without the package logs one warning and serves shoppers
untouched. `RELAY_ENABLED` without an endpoint logs one warning and exports
nothing. Each is said **once**, not once a turn — the agent is rebuilt every
turn, so a warning on that path would otherwise repeat forever.

### Why the dependency is a build arg

`nemo-relay[deepagents]` requires `langgraph>=1.2.9`; `requirements.txt` pins
`1.2.7`. Installing Relay therefore **moves the agent runtime**, and the image
that ships should be the image that was tested. The default image keeps the
tested pin; `INSTALL_RELAY=true` produces a second image carrying
`langgraph 1.2.11` and `nemo-relay 0.7.3`.

The gap is four patch releases inside `1.2.x` and the unit suite has been
running against `1.2.11` locally throughout. That is evidence for the bump, not
proof of it.

### Checking that Relay is doing its job

Traces appearing does not mean Relay is working — the LangChain instrumentation
produces them alone, so a broken Relay looks exactly like a working one until
you count.

**1. It started.** `docker logs chain-server | grep -i relay`

| What you see | What it means |
|---|---|
| `Relay tracing enabled, exporting to …` | Subscriber registered. |
| `nemo-relay is not installed` | Image built without `INSTALL_RELAY=true`. |
| `OTEL_EXPORTER_OTLP_ENDPOINT is not` | Flag on, nowhere to send. |
| `did not preserve our middleware order`<br>`dropped … from the agent arguments` | **A new Relay release changed the agent.** Tracing is off and the shop is running on its original arguments. Do not upgrade past this without reading the wrapper. |
| nothing at all | `RELAY_ENABLED` never reached the container. Check `docker compose config`. |

**2. Its spans arrive.** Send a turn, wait for the batch, count by producer —
Relay's spans are the ones carrying `nemo_relay.*` attributes:

```bash
curl -s -X POST localhost:8009/query/stream -H 'Content-Type: application/json' \
  -d '{"query":"show me black dresses in a size 2","user_id":770000111,
       "conversation_id":"relay-check"}' > /dev/null
sleep 15

curl -s 'localhost:6006/v1/projects/default/spans?limit=1000' | python3 -c "
import sys, json
rows = json.load(sys.stdin)['data']
turn  = [r for r in rows if r['attributes'].get('conversation.id') == 'relay-check']
relay = [r for r in rows if any(k.startswith('nemo_relay.') for k in r['attributes'])]
print('turn span found :', bool(turn))
print('relay spans     :', len(relay))
print('relay names     :', sorted({r['name'] for r in relay}))
"
```

Read `relay spans` as a project total, not a per-turn count: only the
`relay-turn` scope carries a conversation, so what matters is whether the number
**grew**. Roughly six accumulate per turn. **Zero while the `turn` span is
present** is the failure that matters — the app is exporting and Relay is not.

**3. It costs the shopper nothing.** Compare turn latency with the flag on and
off. Turn time is dominated by model round trips and varies by seconds on its
own, so read it as "no new seconds", not as a benchmark. Exports are batched and
off-path; a consistent multi-second penalty is a bug.

**4. The agent is unchanged.**

```bash
python3 -m pytest tests/unit/chain_server/test_deepagents_observability.py -k Relay -q
RELAY_ENABLED=true python3 -m tests.evaluation.src.replay --label relay-on --only J13
```

### The guarantee

An observability layer that can quietly change the agent is worse than no
tracing. `add_nemo_relay_integration` is handed the arguments for
`create_deep_agent` and returns them modified, so the runtime **checks the
result rather than trusting it**:

- every argument that went in must come back
- no argument other than `middleware` may be replaced
- our middleware must still come first, so the tool-loop control and the skill
  gate run ahead of anything Relay adds

Any violation is logged and the agent is built from the arguments it started
with. Tracing degrades; the shop does not change. Verified against
`nemo-relay 0.7.3`: all six arguments survive and Relay's middleware is
*appended* to ours.

### What Relay does not give you here

**Its spans do not join the turn's trace.** One turn arrives as about seven
traces: the turn itself, plus one root per Relay LLM call, tool call and mark.
The runtime opens a `relay-turn` scope per turn carrying the conversation id, so
Relay has a per-turn record — but its model and tool events do not nest under
it, because they are emitted where the scope stack does not follow. Relay ships
`propagate_scope_to_thread` for that, but it would have to be applied inside
Relay's own middleware rather than in code we own.

**It cannot tell you which skill fired.** Its DeepAgents integration reports
skills from a `skills=` argument to `create_deep_agent` that this service does
not pass — skills reach the agent through the filesystem backend and a gate
middleware. So its mark arrives with an empty list and `orphan: true`.
`metadata.skills` on the `turn` span remains the answer.

Both of these follow from the design section above: Relay is being used outside
its own runtime.

---

## Retention

Phoenix holds prompts, completions and tool transcripts — more than
`memory_retriever` keeps by design — and it persists in the `phoenix-data`
volume. It is published on all interfaces so a tunnel can reach it; on anything
other than a development box that wants to go back to loopback with access
through the tunnel only.

There is no backfill. Only live turns are captured, so a replayed transcript
correctly shows no LLM spans.
