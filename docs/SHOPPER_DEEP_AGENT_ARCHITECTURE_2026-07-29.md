# Shopper Deep Agent Architecture — 2026-07-29

This is a dated snapshot of the shopper-serving architecture at commit
`f6fe646`. It records the narrow architecture correction and the minimum proof
needed before broader regression testing. The maintained detailed reference is
[Shopper Agent Architecture](SHOPPER_AGENT_ARCHITECTURE.md).

Status: the source audit is complete, but the fresh three-turn live gate failed
on its final comparison turn. This document records the architecture and that
failure; it is not a feature-readiness claim.

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
