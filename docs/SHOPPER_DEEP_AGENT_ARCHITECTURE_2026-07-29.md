# Shopper Deep Agent Architecture — 2026-07-29

This document contains two dated views:

1. the shopper-serving architecture as built at commit `f6fe646`; and
2. the durable cross-turn context plan agreed on 2026-07-30 after analyzing the
   failed comparison turn.

The planned sections are explicitly marked and are not an implementation
claim. The maintained detailed reference is
[Shopper Agent Architecture](SHOPPER_AGENT_ARCHITECTURE.md).

Status: the source audit is complete, but the fresh three-turn live gate failed
on its final comparison turn. The failure exposed a general cross-turn
continuity gap: Deep Agents can summarize a long request-local graph run, but
that summary is not carried into the next shopper request. The agreed plan
below addresses that general gap without adding an event-specific state
machine. No durable summary or cross-turn weather receipt described below is
built yet, so this remains a design record rather than a feature-readiness
claim.

## Decision

One Deep Agent is the only semantic procedure and tool-use authority for a
shopper turn. It interprets the conversation, selects skills, chooses eligible
tools, observes their results, and decides when the procedure is complete.

Deterministic runtime code surrounds that agent. It validates skill
composition, tool grants, typed arguments, execution limits, evidence
boundaries, scoped response postconditions, redaction, replay, and finalization.
It does not route shopper language into an intent-specific workflow.

The final response stage is hybrid:

```text
tool evidence
    → deterministic evidence and response-path selection
    → optional tools-disabled model editor
    → scoped postconditions, fallback, redaction, and rendering
```

The editor may improve wording within the supplied evidence. On the protected
event path, it may also select one shopper-authored venue quote and one or two
allowlisted styling-adjustment codes. It cannot select skills, call tools,
change tool results, or reopen the Deep Agent.

## Serving Flow

```text
UI / API
    ↓
durable turn start + profile snapshot
    ↓
one request-scoped Deep Agent graph
    ↓
bounded grounding and redaction
    ↓
durable finalize
    ↓
SSE response to the UI
```

Inside the one Deep Agent graph:

```text
activate skills
    → inject complete selected skill files
    → expose only their tool-grant union
    → choose one tool
    → observe its server-authored or validated result
    → choose another tool or answer
```

This is a single semantic procedure/tool authority, not a single model call.

## One Turn, Step by Step

1. FastAPI accepts `/query/stream` and gives the request to
   `DeepAgentsRuntime`.
2. The runtime starts a durable memory turn before guardrail, model, or tool
   work. An exact finalized retry replays without repeating that work.
3. Input guardrails run when configured. The media-perception boundary is
   invoked every turn, but model-backed media analysis runs only when
   applicable. The runtime loads the cached catalog capability contract.
4. The runtime creates one request-scoped Deep Agent and invokes that graph
   once under the shared execution deadline.
5. The first model step can call only `activate_shopper_skills_tool`. The model
   selects the smallest applicable skill set and, for `event-context`, the next
   relevant event question.
6. Deterministic activation code accepts only a valid skill composition. It
   then injects the complete selected `SKILL.md` files and exposes only the
   union of their declared tool grants.
7. The same Deep Agent performs the procedure sequentially. It may call one
   shopping tool in a model step, inspect the server-authored or validated
   result in the next step, and then call another eligible tool or answer.
8. `ToolLoopControlMiddleware` bounds catalog-schema repair and closes tools for
   final synthesis. Other deterministic runtime boundaries own search
   count/deduplication, detail-read limits, graph recursion, and the shared
   deadline. None decides what the shopper meant.
9. After the graph answers, deterministic code collects current tool evidence
   and chooses the safe grounding path. A separate tools-disabled model editor
   may rewrite the answer within that evidence boundary.
10. Scoped postconditions cover empty output, protected-event JSON,
    candidate retention, weather-language stripping, and redaction.
    Deterministic fallback handles editor timeout or error and failures of
    those checks. General rewritten prose is not comprehensively fact-validated.
    Weather attribution and optional output guardrails are then applied.
11. Memory durably finalizes the turn before products, content, and metrics are
    emitted to the UI.

## Ownership

| Component | Owns | Does not own |
| --- | --- | --- |
| Deep Agent | Shopper-language interpretation, skill selection, eligible tool selection, comparison and styling judgment | Product, weather, cart, policy, or memory facts |
| Shopper skills | Complete procedures, composition metadata, and declared tool grants | Execution authorization or external facts |
| Activation middleware | First-step activation, skill injection, and model-visible grant union | Shopper-intent routing rules |
| Tool-loop middleware | Bounded catalog-schema repair and post-tool synthesis closure | A second plan, completion review, or all execution limits |
| Tools and services | Catalog, weather, cart, policy, and durable-memory facts or mutations | Styling judgment |
| Grounding stage | Evidence isolation, bounded response-only judgment, scoped postconditions, redaction, and fallback | Tool use or semantic procedure correction |
| UI | Representative-shopper selection and response rendering | Profile authority, agent routing, or weather resolution |

The two application-supplied Deep Agent middleware components are:

1. `ToolLoopControlMiddleware`
2. `ShopperSkillActivationMiddleware`

Shopper skills are injected by the activation middleware. General-purpose
subagents are disabled. The legacy `planner.py` and `graph.py` paths are not
part of the serving request flow.

## Persistence and Lifetime Boundaries — As Built

The word "context" currently refers to several different lifetimes. The
distinction is architectural: durable conversation memory is not the same
thing as a Deep Agents graph checkpoint or its automatic summarization.

| Lifetime and owner | Persisted contents | Use on a later turn |
| --- | --- | --- |
| Memory-service installation | Five immutable representative-shopper records | Resolves the profile selected for a conversation |
| Conversation | Ordered shopper/assistant turns, the nullable profile binding enforced through those turns, structured event envelopes, the bounded product-reference projection, and its last finalized turn | Reconstructs recent dialogue and resolves historically presented products |
| Cart owner | Current cart rows with stable cart-line IDs and the cart-mutation idempotency ledger | Supplies the authoritative current cart; the bundled UI creates a new cart identity with a new conversation |
| Finalized request | Request and attempt identity, request/finalize digests, sequence, status, termination reason, catalog revision, assistant text, product/image response artifacts, diagnostics, and selected skill names | Exactly replays the same finalized request and exposes the immediately previous skill names as a non-authorizing hint |
| Chain-server process and request | Full Deep Agents messages, tool calls and results, model reasoning, current-turn evidence maps, and the LangGraph checkpoint | Not durable conversation state; the checkpoint is request-scoped and deleted after successful finalization |

The product ledger is already the correct durable grounding mechanism for
historical product references. Finalized presented-product events rebuild a
bounded, same-conversation product-reference index. One unique exact match can
be restored after a chain-server restart or on another worker. Missing,
ambiguous, or stale-catalog references still require clarification or a fresh
search.

### Current next-turn hydration contract

At turn start, the memory boundary currently supplies:

- a versioned rolling-summary storage pair, currently empty because compaction
  is not connected;
- bounded, context-eligible raw shopper/assistant turns;
- the selected representative-shopper context, or no context for Guest;
- the current authoritative cart;
- the historical product-reference index; and
- the immediately previous selected skill names as a continuity hint.

The raw turns are bounded first by the memory service's turn limit and again by
the chain server's character limit. Blocked and abandoned turns are durable for
audit or replay semantics but are excluded from the model context.

The serving path does **not** currently populate or rehydrate:

- a conversation-level semantic summary;
- the complete prior tool transcript or model reasoning;
- normalized event location, venue, or date state;
- prior weather tool evidence or a record that a forecast was already fetched;
- active anchors or effective preferences, although reserved projection lanes
  exist; or
- raw uploaded media.

Deep Agents 0.6.12 includes automatic summarization middleware. In this
application it can compact one long graph execution, but the graph thread is
keyed by both `conversation_id` and `request_id`. A new shopper turn receives a
new request ID, and a successfully finalized turn deletes its checkpoint.
Deep Agents' internal summary therefore does not cross the durable turn
boundary. The legacy `summarizer.py` path is not part of the serving runtime.

## Durable Cross-Turn Context Plan — Agreed 2026-07-30

Implementation status on 2026-07-30: Slice 1 is built. Migration 8, the memory
wire contract, finalize-time compare-and-swap, and strictly post-watermark raw
reads establish the durable summary boundary. The runtime intentionally neither
generates nor renders summary text yet. Slices 2–5 remain planned.

The failed third turn should be corrected at the general continuity boundary,
not with a new weather or comparison workflow. The serving Deep Agent remains
request-scoped. The memory service gains a durable rolling conversation summary
and a bounded projection of selectively reusable typed evidence.

The planned next-turn context is:

```text
rolling semantic summary
    + non-overlapping bounded raw-turn tail
    + selected shopper profile
    + authoritative current cart
    + bounded historical product-reference index
    + immediately previous selected-skill hint
    + small bounded set of valid typed receipts
```

This does not turn the summary into a planner, router, grant, or evidence
source. It gives the Deep Agent semantic continuity. Authoritative state and
typed evidence remain separately grounded.

The planned persistence additions are deliberately small:

| Lifetime and owner | Planned addition | Explicitly not added |
| --- | --- | --- |
| Conversation projection | Rolling summary text, its through-sequence watermark, and a bounded set of active typed receipts | Full tool history, model reasoning, or a conversation-long graph checkpoint |
| Finalized request | Source identity for any receipt promoted by that request and an atomic summary/projection update under the existing attempt fence | A second copy of the rolling summary or raw provider output |
| Request-local Deep Agent | No new durable state; it is hydrated from memory at the start of each request | Cross-request authorization or ownership of profile, cart, catalog, or receipt truth |

### Rolling summary contract

The memory-owned conversation projection now stores:

- `summary_text`: one bounded semantic summary;
- `summary_through_sequence`: the last context-eligible turn represented by
  that summary; and
- the existing projection version used for compare-and-swap finalization.

No second independent version was necessary.

The non-overlap invariant is:

```text
summary_text                 covers eligible turns ≤ summary_through_sequence
bounded raw-turn tail        contains eligible turns > summary_through_sequence
```

No eligible turn is intentionally supplied in both forms, and no turn is
dropped merely because summarization failed. The compactor receives only the
prior summary and the oldest raw turns being folded into it. It does not
receive or reproduce the complete tool transcript.

Compaction should occur only when the raw tail reaches its configured bound,
not after every shopper message. A successful compaction advances the sequence
watermark atomically with durable turn finalization. If compaction fails, the
old summary and watermark remain authoritative and the unsummarized turns
remain raw; the runtime must not advance the watermark or silently discard
them.

The rolling summary may preserve semantic facts such as:

- the shopper is assembling a semi-formal wedding outfit;
- the event is in New York on an outdoor patio next week; and
- the shopper is now comparing two previously shown dresses.

It cannot prove a price, product property, forecast value, availability result,
cart mutation, or policy fact. It also cannot authorize a tool or replace the
fresh skill-selection step.

### Selective typed receipts, not historical tool transcripts

Persisting every catalog, cart, and weather output would reproduce the
unbounded-context problem and retain stale facts. The plan promotes only
compact, server-validated results that are useful across turns.

| Tool-result class | Cross-turn treatment |
| --- | --- |
| Catalog search | Do not persist the search transcript. Continue to project deduplicated presented products into the existing bounded product ledger. |
| Product details | Resolve through the ledger and fetch current details when needed. A catalog-revision-bound detail receipt may be added only if later measurements justify it. |
| Cart reads and mutations | Use the authoritative current cart. Keep mutation records for idempotency, not as model-visible historical evidence. |
| Weather forecast | Add one compact typed receipt for a successful normalized forecast scope. |
| Availability and promotions | Recheck because these facts are volatile; do not carry their old result as current evidence. |
| Tool failures | Keep the sanitized failure on its request for replay and diagnostics, but do not promote it as reusable factual evidence. |

A proposed `weather_forecast.v1` receipt contains only:

- a stable receipt type and identifier;
- source turn and tool identity;
- provider and fetch time;
- provider-resolved location;
- exact requested start and end dates;
- normalized daily conditions already accepted by
  `WeatherForecastEvidence`;
- required provider attribution; and
- a configured validity boundary.

The receipt never contains the API key, prepared URL, raw provider body, raw
exception, or an unvalidated model-authored forecast.

A weather receipt is reusable only when its schema validates, its exact
location/date scope still applies, and its configured freshness boundary has
not expired. A changed location or date is a different scope. Newer success for
the same normalized scope supersedes the older receipt. The projection retains
only a small configured number of unexpired scopes and never injects expired
or superseded receipts into the model context.

The bounded active-receipt projection, rather than an append-only copy of every
tool response, is the model-facing source. A separate short audit retention
policy may exist if operationally required, but it is not part of the agent's
context and is not required by this feature.

### Grounding relationship

The three context forms have deliberately different authority:

| Context form | Purpose | Authority |
| --- | --- | --- |
| Rolling summary | Older conversational meaning and goals | Semantic continuity only |
| Raw-turn tail | Recent exact shopper and assistant wording | Reference resolution input, not proof of external facts |
| Profile, cart, product ledger, and valid receipts | Current state and validated external evidence | Deterministic grounding within each artifact's declared scope |

For example, the summary can establish that the shopper is comparing the lacy
gown and satin dress for the same wedding. The product ledger resolves those
names to catalog identities, current detail calls establish their verified
attributes, and a still-valid weather receipt establishes the previously
observed forecast. Assistant prose saying that rain was expected is not a
substitute for the receipt.

### Planned request lifecycle

With this addition, one request proceeds as follows:

1. Memory durably starts the request and returns the summary plus its sequence
   watermark, only the later bounded raw turns, profile context, cart,
   product-reference projection, prior-skill hint, and valid receipt
   projection.
2. The chain server renders those sections separately in one bounded
   current-turn context. It does not merge summary prose into authoritative
   evidence.
3. One request-scoped Deep Agent selects skills and performs the turn. Its
   built-in summarization may still compact a long graph execution, but it
   remains request-local.
4. Grounding accepts current validated tool results and applicable prior typed
   receipts. The rolling summary alone cannot support an external-fact claim.
5. Before durable finalization, the chain server may prepare a bounded summary
   compaction and any validated receipt promotion produced by this request.
6. Memory atomically finalizes the turn, exact-replay output, product events,
   projection version, accepted summary advancement, and bounded receipt
   updates under the existing attempt fence.
7. Successful finalization deletes the request checkpoint. An exact retry
   replays the finalized output without repeating compaction, tool work, or
   receipt promotion.

This keeps `request_id` as the idempotent execution identity and
`conversation_id` as the durable continuity identity. It does not require a
shared conversation-long LangGraph checkpoint.

### Architectural consequence for event guidance

No new event-specific action enum, event-state machine, comparison skill,
intent router, or completion reviewer is part of this plan. Event and product
continuity live in the generic summary and existing product ledger. Exact
forecast reuse lives in a scoped weather receipt. The event-context and
outfit-styling skills continue to own semantic judgment inside the one Deep
Agent.

The existing `event_context_next_question` field remains part of the
`f6fe646` as-built snapshot until an implementation slice deliberately changes
it. This plan does not extend that field into broader per-turn state.

### Clean implementation slices

Each slice stops after its focused proof and ships code and documentation at
the same breadth:

1. **Durable summary boundary — built:** migration 8 adds the projection fields
   and memory wire contract, including versioned atomic updates and the
   non-overlap retrieval invariant. Focused offline tests prove accepted
   advancement, exact retry, conflict rollback, migration preservation, and
   strictly later context-eligible raw reads without calling a hosted model.
2. **Rolling compaction and hydration:** add the bounded summarizer call and
   render summary and raw tail as separate prompt sections. Prove that
   compaction sees only the prior summary plus the turns being folded, and that
   a failed call never advances the watermark.
3. **Weather receipt projection:** promote only validated
   `WeatherForecastEvidence`, enforce exact scope, freshness, supersession, and
   cap behavior, and expose only applicable receipts to grounding. Keep the
   product ledger and current cart unchanged.
4. **Focused conversation proof:** run the one three-turn live fixture without
   Judge and require search, then weather, then historical resolution plus two
   detail reads with no weather refresh on the comparison turn. Inspect
   transcript, quality, call count, and timing before any broad suite.
5. **Final regression gate:** only after the focused feature passes, run the
   broader offline suite and any explicitly approved live evaluation cohort.

The implementation should not resume the paused per-turn event-action
correction. That direction treated request-local context loss at the event
skill boundary instead of correcting the generic cross-turn continuity
boundary.

## Deliberately Not Present

The serving path has no:

- intent router or keyword-based shopper workflow;
- product-comparison skill;
- city-to-ZIP rule table or representative-ZIP substitution;
- post-answer completion reviewer;
- operation plan that can replace the agent's procedure;
- correction path that discards an answer and reopens tools;
- second serving graph for styling, weather, or comparison.

Named locations such as `NYC`, `Cancun`, or `Paris` go directly through the
Visual Crossing adapter. A bounded model-authored qualifier may preserve and
clarify the shopper's phrase, such as `NYC, NY`; deterministic code does not
invent a ZIP.

## The Three-Turn Live Architecture Gate

The focused fixture is
[`conv_prior_product_weather.yaml`](../tests/integration/conversations/event_context_comparison/conv_prior_product_weather.yaml).
It is one conversation with three turns, not three independent end-to-end
tests.

| Turn | Shopper action | Activation result | Exact business-tool path | Must not happen |
| --- | --- | --- | --- | --- |
| 1 | States a semi-formal wedding shopping occasion | `outfit-styling` + `event-context`; ask `event_location` | One catalog search | Weather, product details, or ZIP disclosure |
| 2 | Supplies `NYC`, outdoor patio, and next week | Same skills; ask `none` | One weather call | A repeated catalog search or repeated event question |
| 3 | Compares the lacy gown with the hem satin dress | Same skills; ask `none` | One batched historical resolution, then two detail reads | A new search or weather refresh |

The deterministic fixture validator checks exact skill selection, event-question
choice, tool order and counts, required evidence, the categorical redacted
weather trace, stable product names, and forbidden response fragments. Those
assertions are test oracles only; they do not participate in serving decisions.

### What a passing run proves

- one Deep Agent can carry the event-and-product thread across three turns;
- event context adds weather without replacing outfit styling;
- context fulfillment does not cause product rediscovery;
- natural prior-product references resolve through durable memory;
- both products are refreshed from current catalog details before comparison;
- the observed weather-call diagnostic contains redacted arguments and only
  the expected categorical trace;
- the measured call count and timing for this exact path.

### What it does not prove

- broad regression safety or stochastic repeatability;
- Guest mode or every representative profile;
- every location, date, saved-ZIP, or weather-failure path;
- UI behavior, guardrails, or load behavior;
- semantic styling quality across a wider conversation set.

## Why This Small Live Test Is Still Expensive

The last passing working-tree run used 14 application-model calls and 192,753
tokens across only three turns:

| Turn | Deep Agent sequence | Final editor | Model calls |
| --- | --- | --- | ---: |
| 1 | activation → search → synthesis | grounding editor | 4 |
| 2 | activation → weather → answer | protected event editor | 4 |
| 3 | activation → resolver → detail → detail → comparison | grounding editor | 6 |

It also used one catalog text embedding and one Visual Crossing request. No
catalog repair or duplicate-call recovery was triggered.

The cost is therefore mainly sequential agent reasoning with repeated context,
plus one final editor call per turn. It is not caused by the deterministic
fixture validator, which runs only after all hosted calls finish. Reducing those
model calls is a separate performance slice; removing the validator would not
materially reduce this run's cost and would only hide the failure.

## Minimum Validation Policy

During feature iteration:

1. Run the one focused three-turn live fixture first, once, without Judge,
   Challenger, guardrails, or the broad suite.
2. If it fails, archive the transcript and stop. Diagnose that one artifact; do
   not automatically rerun or patch in a loop.
3. If it passes, inspect the transcript and timing, then run only these three
   small offline boundaries before a code commit:
   - capability boundary: `event-context` adds only weather to the
     `outfit-styling` grant union;
   - single-agent loop boundary: a weather result does not hide or block later
     business tools;
   - oracle boundary: the committed three-turn fixture and standalone
     diagnostic validator reject incomplete or incorrect sequences.
4. Run the 1,000+ unit suite and wider live cohorts once at the final
   merge/release gate, after the focused feature works.
5. Use Judge separately when semantic response quality needs scoring. Judge is
   not required to prove this tool architecture.

Focused offline commands:

```bash
python skills/retail-test-runner/scripts/run_retail_tests.py unit \
  --pytest-args -q \
  unit/chain_server/test_tool_policy.py::test_event_context_adds_only_weather_to_outfit_styling

python skills/retail-test-runner/scripts/run_retail_tests.py unit \
  --pytest-args -q \
  unit/chain_server/test_skill_activation.py::test_weather_result_does_not_hide_or_block_later_business_tools

python skills/retail-test-runner/scripts/run_retail_tests.py unit \
  --pytest-args -q \
  unit/integration/test_event_context_comparison_dataset.py::test_event_context_comparison_dataset_is_one_three_turn_gate \
  unit/test_diagnostic_validation.py::test_standalone_validator_rejects_wrong_or_incomplete_sequence \
  unit/test_diagnostic_validation.py::test_standalone_validator_checks_weather_outcome_and_response
```

These commands are documented here; this documentation slice runs the live
gate first and does not run the broad suite.

## Source of Truth

- Serving orchestration:
  [`deepagents_runtime.py`](../chain_server/src/deepagents_runtime.py)
- Skill activation and grant visibility:
  [`skill_activation.py`](../chain_server/src/skill_activation.py)
- Tool-loop limits:
  [`tool_loop_control.py`](../chain_server/src/tool_loop_control.py)
- Immutable tool policy:
  [`tool_policy.py`](../chain_server/src/tool_policy.py)
- Shopper procedures:
  [`chain_server/skills/shopper/`](../chain_server/skills/shopper/)
- Durable-turn client:
  [`conversation_memory.py`](../chain_server/src/conversation_memory.py)
- Representative-shopper client:
  [`shopper_profiles.py`](../chain_server/src/shopper_profiles.py)
- Prior-product resolver:
  [`conversation_products.py`](../chain_server/src/conversation_products.py)
- Cart adapters and memory-service authority:
  [`commerce_tools.py`](../chain_server/src/commerce_tools.py) and
  [`memory_retriever/src/`](../memory_retriever/src/)
- Named-place weather adapter:
  [`weather.py`](../chain_server/src/weather.py)
- Three-turn fixture:
  [`conv_prior_product_weather.yaml`](../tests/integration/conversations/event_context_comparison/conv_prior_product_weather.yaml)
- Judge-free diagnostic oracle:
  [`diagnostic_validation.py`](../tests/integration/diagnostic_validation.py)

## Current Qualification

The previous passing archive was produced from the working tree immediately
before commit `f6fe646`; its metadata identifies base commit `c00c448` with
`git_dirty=true`. The committed fixture is byte-identical to that archive, but
a fresh focused run was required to tie the proof to the committed serving
source.

That one fresh run was performed on 2026-07-29 with serving source unchanged at
`f6fe646`; only this documentation slice was uncommitted. It **failed** the
standalone diagnostic gate on the third shopper turn:

```text
required: resolver → detail → detail
observed: weather → resolver → detail → detail
```

The first two turns followed the required search and weather paths. The third
turn correctly resolved and read both products, but unnecessarily refreshed
weather and repeated the forecast in the comparison response. The run used 15
application-model calls, 215,681 tokens, one text embedding, and two weather
requests over 70.75 seconds. No Judge call was made, and no automatic retry or
serving-code patch followed.

Canonical local evidence:

- transcript:
  `~/exec-briefs/retail-shopping-assistant/quality/shopping/targeted/event_context/post_answer_completion_slice3/runs/f6fe646_live_failed_2026_07_29/results/conv_prior_product_weather.yaml`
- run summary:
  `~/exec-briefs/retail-shopping-assistant/quality/shopping/targeted/event_context/post_answer_completion_slice3/runs/f6fe646_live_failed_2026_07_29/summary.md`
- comparison with the prior passing working-tree run:
  `~/exec-briefs/retail-shopping-assistant/quality/shopping/targeted/event_context/post_answer_completion_slice3/comparisons/single_authority_current_wip__to__f6fe646_live_failed_2026_07_29.md`
