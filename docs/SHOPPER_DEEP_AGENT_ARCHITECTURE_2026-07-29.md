# Shopper Deep Agent Architecture — 2026-07-29

This document contains two dated views:

1. the shopper-serving architecture as built at commit `f6fe646`; and
2. the durable cross-turn context plan agreed on 2026-07-30 after analyzing the
   failed comparison turn, with implementation status recorded by slice.

The addendum distinguishes built slices from the remaining plan. The maintained
detailed reference is
[Shopper Agent Architecture](SHOPPER_AGENT_ARCHITECTURE.md).

Status: the source audit is complete. The first fresh three-turn live gate on
2026-07-29 failed on its final comparison turn and exposed a general cross-turn
continuity gap: Deep Agents can summarize a long request-local graph run, but
that summary is not carried into the next shopper request. The agreed
correction below addresses that general gap without adding an event-specific
state machine. Its durable summary boundary, rolling compaction, and bounded
cross-turn weather-receipt projection are now built. The weather continuity
correction adds one typed current scope plus a source-identity-bound semantic
resolver, durable pending-question binding, and response-contract-v5 lane
containing the exact completed turns referenced by that scope. Both scope components are
resolved atomically through `retain`, `set`, or `clear`; deterministic
compilation accepts `set` only from current-turn authority, and the forecast
tool accepts no location/date arguments. Prior raw turns and the rolling
summary therefore guide semantics without directly authorizing a provider
request.

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
   relevant event question. The same closed date-authority parser used by the
   tool gate removes `event_date` from this turn's activation enum only when the
   current shopper turn contains an accepted bounded date. Prior raw-turn dates
   remain available to the semantic agent and weather tool but cannot narrow
   activation. The model may also bind one listed valid weather receipt only
   for the unchanged exact event scope and only when that question is `none`.
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
   plus only the one explicitly bound valid receipt and chooses the safe
   grounding path. A separate tools-disabled model editor may rewrite the
   answer within that evidence boundary.
10. Scoped postconditions cover empty output, protected-event JSON,
    candidate retention, weather-language stripping, and redaction.
    Deterministic fallback handles editor timeout or error and failures of
    those checks. General rewritten prose is not comprehensively fact-validated.
    Weather attribution and optional output guardrails are then applied.
11. Memory durably finalizes the turn before products, content, and metrics are
    emitted to the UI. A paired successful weather call/result may be promoted
    in that same atomic transaction.

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
| Conversation | Ordered shopper/assistant turns, the nullable profile binding enforced through those turns, structured event envelopes, the rolling-summary projection, the bounded product-reference projection, one typed current weather-planning scope with a pending location/date binding, at most four valid typed weather receipts, and its last finalized turn | Reconstructs recent dialogue, resolves historically presented products, carries current location/date authority without an event registry, derives an at-most-three exact scope-source read lane across compaction, and keeps short-lived exact-scope forecast evidence server-side for explicit binding |
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

- a versioned rolling semantic summary;
- bounded, context-eligible raw shopper/assistant turns;
- the selected representative-shopper context, or no context for Guest;
- the current authoritative cart;
- the historical product-reference index; and
- the immediately previous selected skill names as a continuity hint; and
- one typed current weather-planning scope with optional pending
  `event_location`/`event_date`; and
- at most three exact completed turns referenced by that scope in a
  lane isolated to subject resolution and protected styling provenance; and
- a bounded list of validated, unexpired `weather_forecast.v1` receipts.

The raw turns are bounded first by the memory service's turn limit and again by
the chain server's character limit. Blocked and abandoned turns are durable for
audit or replay semantics but are excluded from the model context.

The additive summary, receipt, current-weather-scope, and scope-source lanes use
negotiated turn-start response contracts. The current chain requests contract 5 as its
maximum, and memory returns the highest version it supports. An unversioned
caller receives the exact earlier response shape and a bounded raw tail read
from sequence zero, so an older chain remains usable after memory deploys first
or after chain rollback even if a summary watermark has advanced. Conversely,
the current chain treats a missing contract marker from older memory as
contract 1. That legacy contract may contain abandoned turns without assistant
text; the chain filters them before model context and applies the strict
memory-owned eligibility invariant only to contract 2 and later. Contract 3
remains supported with the legacy scope transition and
without the v4-only pending question or either source field. Contract 4 marks
atomic scope-finalize write capability. Contract 5 adds only the exact
scope-source read lane, required even when empty and absent from earlier
contracts. It derives from existing rows and needs no migration. Deploy memory
first, then chain; a new chain negotiating only v3
fails closed for an atomic update. Fresh and upgraded SQLite
schemas give the additive non-null projection columns equivalent database
defaults, allowing the memory binary to roll back without making old projection
inserts fail. Migration 11 stores pending state in its own defaulted lane so the
core scope JSON remains strict pre-v4-readable.

The serving path does **not** populate or rehydrate the complete prior tool
transcript or model reasoning, a normalized event state machine, active anchors
or effective preferences, or raw uploaded media. A receipt is the sole bounded
prior-weather evidence form; assistant forecast prose and prior weather tool
messages remain non-evidence.

Deep Agents 0.6.12 includes automatic summarization middleware. In this
application it can compact one long graph execution, but the graph thread is
keyed by both `conversation_id` and `request_id`. A new shopper turn receives a
new request ID, and a successfully finalized turn deletes its checkpoint.
Deep Agents' internal summary therefore does not cross the durable turn
boundary. The legacy `summarizer.py` path is not part of the serving runtime.

### Current weather scope correction — serving boundary built 2026-07-30

Memory migration 10 adds one singleton `CurrentWeatherScope`, not a list of
event anchors. It contains only a monotonic revision, optional location and
normalized date-window components. Memory, not the model, stamps each newly set
component with its source turn ID and sequence. Migration 11 stores the optional
pending `event_location`/`event_date` binding in defaulted
`current_weather_pending_json`. It extracts any pending fields embedded by the
pre-split v4 work only when the complete source-bound binding is present;
incomplete unsourceable pending fields are dropped, leaving
`current_weather_scope_json` parseable by strict pre-v4 models. Memory
rehydrates the pending binding only when its stored scope
revision matches the core revision, so a rollback-era core mutation makes stale
pending state inert.

Contract 3 retains the legacy `continue`/`replace` transition for rolling
compatibility. Contract 4 uses one atomic finalize resolution: it copies the
expected scope revision and explicitly chooses `retain`, `set`, or `clear` for
both location and date. A value is present exactly for `set`; retaining a
missing component fails validation.

Changing location or date invalidates older receipts. A receipt promoted in the
same finalize must exactly match the resulting complete scope. Venue, occasion,
products, styling claims, and forecast evidence remain in their existing
semantic/evidence lanes and never enter this scope. The same singleton can
represent an event request or ordinary weather-appropriate dressing.

Before normal skill activation, response contract 5 supplies the exact
completed shopper/assistant turns named by the current scope's location, date,
and pending-question source identities in a separate lane. A request-local
tools-disabled resolver compares that lane with the current query. It must make exactly one
forced typed
`WeatherScopeResolverDecision` control call. It is not a registered business
tool, not a separate weather agent, and not a subagent. Rolling summary and
other recent turns may provide semantics but cannot supply authority. The
source lane is independent of compaction and raw-tail eviction; contract 4
retains the bounded-tail fallback during rolling deployment. Missing source
turns, timeout, malformed output, structural invalidity, or an unclear
relation fails closed.

The chain constructs this call under one aggregate dynamic-input budget,
configured by `weather.scope_resolver_max_input_chars` (16,384 by default;
1,024–65,536 allowed). It first preserves the exact payload shape when it fits.
On overflow, the current query, typed scope, trusted date, and every source
sequence remain exact while semantic text receives deterministic marked
head-and-tail excerpts. Oldest optional recent turns are removed before the
optional summary. If the mandatory payload still cannot fit, the chain creates
no model, records zero resolver calls, and returns `unclear/not_addressed` so
prior retention fails closed while current-turn facts survive. This projection
is request-local: contract v5, memory limits, durable text, and replay are
unchanged.

The isolated resolver owns only orthogonal `subject_relation` and
`pending_disposition` controls; it never duplicates location/date extraction.
The schema accepts exactly six outcomes:

| Subject relation | Pending disposition | Meaning |
| --- | --- | --- |
| `same_subject` | `not_addressed` | Continue or revise the subject without consuming a pending binding. |
| `same_subject` | `answered` | Complete the exact pending binding. |
| `new_subject` | `not_addressed` | Replace prior subject authority. |
| `unchanged` | `not_addressed` | Leave an unanswered question silent. |
| `unchanged` | `resume_requested` | Re-render the exact unanswered question without a scope write when no current-turn scope facts are supplied. |
| `unclear` | `not_addressed` | Fail closed for prior authority. |

The handle is required exactly for `answered` and `resume_requested`. These
semantic choices remain model-owned; the server has no phrase matcher, intent
router, or rules table for recognizing an answer or a resumption.
Normal activation still owns skill selection, the
one shopper-facing question, and the single atomic scope selection.
Deterministic compilation admits `set` only from current-turn location/date authority. When
the accepted response asks for a missing location or date, memory stores that
question as the pending binding with its originating turn ID and sequence even
if no authority value otherwise changes. A resolver-approved `same_subject`
relation authorizes ordinary same-subject retains, while `new_subject` clears
them. Completing a pending component while retaining its stored counterpart
requires `same_subject/answered`, the exact opaque pending source-turn handle,
and activation setting the component it names. A reply that also changes or
withdraws the opposite component is `same_subject/not_addressed`. Runtime
forwards the handle only when activation itself proposes the counterpart
`retain`; the compiler never rewrites either component action, and memory
atomically verifies that exact live retain shape. A current-turn counterpart
`set` or `clear` uses the ordinary scope-revision boundary with no completion
handle. Preserving an otherwise unanswered
pending binding during another same-subject update likewise requires an exact
server-authored source handle; without it memory binds the question to the
current finalized turn. `unchanged/not_addressed` suppresses an accidental
repeat of the exact live pending question. Exact-handle
`unchanged/resume_requested` re-renders that question without creating a scope
resolution or rotating its original binding when activation supplies no
current-turn scope facts. Any validated current-turn `set` still survives. When the
isolated resolver is unavailable or unclear, the pure authority compiler clears
every proposed retained component without erasing a validated current-turn
`set`, receipt/refresh reuse is rejected, and prior-dependent weather is
blocked. Self-contained incomplete proposals use normal missing-component
handling; a complete `set`/`set` replacement may require weather. The binding
also records that the question was already asked; intervening product work is
instructed not to repeat it. This lets “conference on Sunday next week” clear NYC while
keeping its own date and asking location, then lets “Seattle” complete that
same scope without importing an unrelated subject.

The resulting provider tool is zero-argument: it reads the effective typed
scope and cannot rescan recent prose or accept a model-authored fallback date.
The same singleton supports non-event requests such as weather-appropriate
dressing in Denver without an event or venue anchor.

A scope resolution that produces a complete scope requires the forecast tool
call. For an unchanged complete scope, activation sets `weather_refresh=true`
only for an explicit shopper request for fresh evidence; comparisons and other
turns leave it false and weather is blocked. Required tool choice is both
requested from the model and checked on its response, so prose cannot bypass
the pending evidence step. Diagnostics retain the submitted question while
recording the separately accepted question boundary.

Receipt binding is mutually exclusive with a scope update and requires exact
equality to the effective location/date scope. The scope resolution and any
matching receipt promotion finalize atomically. The rolling summary and bounded
raw turns remain semantic context, never weather-tool authority.

Optional receipt-promotion conflict codes survive the real HTTP client and
trigger one finalize retry without promotion or a model rerun. A scope
revision, resolution, or status conflict is authoritative: runtime discards the
draft/product output and terminalizes the durable turn as failed without the
disputed scope update.

### Serving-boundary validation — 2026-07-31

Focused offline validation passes a 458-test backend subset covering resolver
parsing, atomic scope compilation, source-bound pending questions,
capability-scoped activation enforcement, receipt/finalize recovery, memory
application, diagnostics, and the live-fixture contract. The complete Python
unit suite also passes 1,596 tests.

The activation boundary removes every event-context-only question, scope,
refresh, and receipt field before nested validation when `event-context` was
not selected. This is a grant boundary, not shopper-language routing: those
ungranted controls cannot mutate weather state, expose the forecast tool, or
consume an activation repair. Semantic activation selects `event-context` only
when physical context is part of the current styling subject, not because
weather could hypothetically improve an otherwise location-independent answer.

Three final Judge-free three-turn live fixtures passed every diagnostic
expectation. The activation-boundary fixture created a source-bound pending
rainy-day location question, ran an intervening pink-skirt catalog search with
weather controls inert and no repeated question, then resolved Seattle plus
Friday next week and made one forecast. The cross-subject fixture proved:

1. NYC/Friday next week established a complete scope and made one qualified
   forecast.
2. A newly introduced outdoor conference/Sunday next week cleared NYC, retained
   its current-turn date, made no weather call, and asked for location.
3. The location-only answer “Seattle” completed the pending location binding
   and made one Seattle/Sunday next week forecast.

Across the three cross-subject turns, the pre-split resolver decisions were
`null`, `new_subject`, and `answers_pending`, respectively; the current typed
equivalents are no result, `new_subject/not_addressed`, and
`same_subject/answered`. The second turn proves that the semantic relation
clears activation's attempted NYC retention while preserving the conference's
current-turn date. The third proves the exact source-bound pending-answer path;
its retained, already-normalized date is
correctly recorded as an `exact_date` provider request. No Judge or Challenger ran for any focused
fixture. Their local traces are under
`~/exec-briefs/retail-shopping-assistant/quality/successful-live-traces/`; the
cross-subject comparison is under
`~/exec-briefs/retail-shopping-assistant/quality/weather_scope_continuity/`;
these focused runs are not replacements for the full shopping evaluation.

The third fixture starts with an older wedding that is missing location,
introduces a conference that is also missing location, and then answers the
same question with `Seattle`. It proves that the server-issued completion
handle binds the answer to the conference's pending source and retains only its
date; the older wedding date cannot leak through the shared or memory
validation boundary.

The subsequent full shopping cohort completed all 48 turns and 48 explicit
GPT-5.2 Judge calls at 3.9583/5 and 18.32 seconds average; 42/48 turns scored at
least 4. All 9 diagnostic scenarios passed, with zero activation rejection,
generic activation fallback, agent timeout, grounding timeout, HTTP error, or
Judge error. Relative to the immediately preceding relation-only run, quality
changed by -0.0417, mean / median / p95 / maximum latency changed by +1.66 /
+0.75 / +1.34 / +8.68 seconds, and one additional turn scored at least 4;
paired movement was 7 improved, 33 tied, and 8 regressed. Relative to the
immediately prior WIP baseline, quality is +0.2292 and mean latency is 3.03 seconds
faster. This broad cohort made zero weather calls and produced five inert
`unchanged` resolver decisions. It is broad shopping-regression evidence; the
three focused fixtures are the direct evidence for exact pending completion.
The canonical run and comparisons remain local under
`~/exec-briefs/retail-shopping-assistant/quality/shopping/`.

A separate live comparison regression attempt is preserved locally under
`~/exec-briefs/retail-shopping-assistant/quality/event_context_comparison/failed-live-traces/`.
Its weather-context turn reached Visual Crossing but received a typed timeout,
so no receipt was promoted. The following comparison correctly resolved both
historical products and fetched both detail records, but the prior
unchanged-scope auto-refresh consumed enough latency for the turn to reach the
45-second execution deadline. The correction does not retry the provider or
add an intent router: activation now declares `weather_refresh=true` only for
an explicit shopper request on an unchanged complete scope, and the runtime
hides and dispatch-blocks weather otherwise. Focused tests prove both the
comparison block and explicit-refresh path. The focused and full evaluations
reported above supersede that interim regression state.

## Durable Cross-Turn Context Plan — Agreed 2026-07-30

Implementation status on 2026-07-30: Slices 1 through 3 are built. Migration 8, the
memory wire contract, finalize-time compare-and-swap, and strictly
post-watermark raw reads establish the durable summary boundary. Memory returns
a newest raw prompt tail plus a separate oldest exact compaction prefix. The
runtime renders summary, raw discussion, and historical products separately
and may compact only a validated contiguous boundary in that oldest prefix
after a successful response. Migration
9 and the shared receipt contract add the fourth typed-evidence lane. The
focused live proof and bounded regression gate are complete, and the full
judged shopping evaluation has completed with the result reported above.

The failed third turn should be corrected at the general continuity boundary,
not with a new weather or comparison workflow. The serving Deep Agent remains
request-scoped. The memory service gains a durable rolling conversation summary
and a bounded projection of selectively reusable typed evidence.

The current next-turn context is:

```text
rolling semantic summary
    + non-overlapping bounded raw-turn tail
    + exact current-weather-scope source turns
      (isolated subject/provenance lane; may overlap)
    + selected shopper profile
    + authoritative current cart
    + bounded historical product-reference index
    + immediately previous selected-skill hint
    + small bounded set of valid typed receipts
```

This does not turn the summary into a planner, router, grant, or evidence
source. It gives the Deep Agent semantic continuity. Authoritative state and
typed evidence remain separately grounded.

The persistence additions are deliberately small:

| Lifetime and owner | Built addition | Explicitly not added |
| --- | --- | --- |
| Conversation projection | Rolling summary text, its through-sequence watermark, a bounded set of active typed receipts, and one current weather scope whose existing pointers derive the response-only source-turn lane | Full tool history, model reasoning, duplicated source rows, or a conversation-long graph checkpoint |
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
prior summary and the largest fitting contiguous prefix of the oldest raw turns
that can be folded while retaining the configured newest suffix. If the first
turn alone exceeds the input budget, that call receives explicit deterministic
head-and-tail excerpts of the turn, marked as a bounded input projection. The
durable transcript and exact replay are not altered. The compactor does not
receive or reproduce the complete tool transcript.

The source-turn lane is outside this summary/raw non-overlap invariant. It may
copy one to three exact completed turns from either side of the watermark
because its sole purpose is dereferencing the current scope's durable
identities. It is not compactor input or general Deep Agent context.

Compaction should occur only when the raw tail reaches its configured bound,
not after every shopper message. A successful compaction advances the sequence
watermark atomically with durable turn finalization. If compaction fails, the
old summary and watermark remain authoritative and the unsummarized turns
remain raw; the runtime must not advance the watermark or silently discard
them. Memory accepts the selected watermark only when it equals a turn boundary
in the freshly re-read offered oldest prefix.

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

A `weather_forecast.v1` receipt contains only:

- a stable receipt type and identifier;
- source turn and tool identity;
- provider and fetch time;
- provider-resolved location;
- exact requested start and end dates;
- normalized daily conditions already accepted by
  `WeatherForecastEvidence`;
- required provider attribution; and
- a configured validity boundary.

Saved-area scope stores only the `confirmed_saved_zip` kind; the
profile-bound conversation supplies its identity without persisting ZIP digits
in the receipt. Explicit-location scope stores the exact shopper location plus
its optional qualifier. Exact requested start/end dates complete the scope
identity.

The receipt never contains the API key, prepared URL, raw provider body, raw
exception, or an unvalidated model-authored forecast.

A weather receipt is reusable only when its schema validates, its exact
location/date scope still applies, and its configured freshness boundary has
not expired. The default TTL is 3,600 seconds, the configured hard maximum is
21,600 seconds, and the active projection cap is four. A changed location or
date is a different scope. Newer success for the same normalized scope
supersedes the older receipt. Memory never returns expired or superseded
receipts.

Promotion requires one successful current-turn AI weather call paired with the
same tool-call ID's validated success result. It commits only with completed
turn finalization. A failed, blocked, timed-out, cancelled, or malformed turn
cannot promote a receipt.

The bounded active-receipt projection, rather than an append-only copy of every
tool response, is the server-side source. Its minimal receipt index is rendered
separately from the rolling summary, raw transcript, and product ledger for
activation selection.

Freshness and expiry are evaluated atomically at durable turn start. That
accepted receipt set is the validity snapshot for the full in-flight request;
there is deliberately no second wall-clock expiry check mid-turn. The
pre-activation model receives only receipt ID/type, shopper location/date scope,
and `valid_until`. It never receives normalized forecast evidence. Full evidence
stays server-side and becomes grounding input only after explicit binding.

### Grounding relationship

The context forms have deliberately different authority:

| Context form | Purpose | Authority |
| --- | --- | --- |
| Rolling summary | Older conversational meaning and goals | Semantic continuity only |
| Raw-turn tail | Recent exact shopper and assistant wording | Reference resolution input, not proof of external facts |
| Profile, cart, and product ledger | Current state and validated external evidence | Deterministic grounding within each artifact's declared scope |
| Active weather receipts | Server-side evidence plus a minimal ID/type/scope/expiry index shown for activation selection | No evidence is model-visible and no grounding authority exists until one valid exact-scope receipt is explicitly bound |

For example, the summary can establish that the shopper is comparing the lacy
gown and satin dress for the same wedding. The product ledger resolves those
names to catalog identities, current detail calls establish their verified
attributes, and one explicitly bound still-valid weather receipt establishes
the previously observed exact-scope forecast. An unbound receipt or assistant
prose saying that rain was expected is not a substitute. Current successful
weather takes precedence over the receipt.

### Current request lifecycle

With this addition, one request proceeds as follows:

1. Memory durably starts the request and returns the summary plus its sequence
   watermark, only the later bounded raw turns, profile context, cart,
   product-reference projection, prior-skill hint, valid receipt projection,
   current weather scope, and its exact source turns in an isolated lane used
   only for subject resolution and protected styling provenance.
   Memory returns exact durable text; chain hydration replaces recognized
   assistant forecast prose with a fixed redaction before either consumer sees
   the lane.
2. The chain server renders those sections separately in one bounded
   current-turn context. It does not merge summary prose or the scope-source
   lane into general discussion or authoritative evidence.
3. One request-scoped Deep Agent selects skills and performs the turn. Its
   built-in summarization may still compact a long graph execution, but it
   remains request-local.
4. The isolated source-identity-bound resolver reads only the dedicated source
   lane plus semantic context. It and activation may atomically update the
   singleton with current-turn authority through explicit
   location/date `retain`/`set`/`clear` actions and a typed pending binding.
   They may instead bind exactly one listed receipt
   only with `event-context`, `event_context_next_question=none`, no scope
   update, and equality to the effective scope. Unbound receipts never ground
   and a bound receipt blocks a new weather call. Skill selection, subject
   continuity, venue, materiality, and intent remain model-owned; exact scope
   compilation and tool binding are deterministic.
   Grounding accepts current validated tool results first, then only the bound
   receipt. The rolling summary alone cannot support an external-fact claim.
5. Before durable finalization, the chain server may prepare a bounded summary
   compaction and any validated receipt promotion produced by this request.
6. Memory atomically finalizes the turn, exact-replay output, product events,
   projection version, accepted summary advancement, and bounded receipt
   updates under the existing attempt fence.
7. Successful finalization deletes the request checkpoint. An exact retry
   replays the finalized output without repeating compaction, tool work, or
   receipt promotion.

When current product resolution/details are also present, a bound receipt
guides comparison styling silently. Exact forecast facts and the prior
canonical forecast block are not repeated. Only a current successful weather
result produces a canonical block.

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

The existing `event_context_next_question` field remains the narrow typed
follow-up boundary; it is not broader per-turn state. Slice 3 now narrows its
per-turn enum when the already-shared date-authority parser accepts a bounded
date in the current shopper turn. It deliberately ignores prior raw-turn dates
for enum narrowing, preventing Wedding A's date from suppressing a safe
question for a newly introduced Wedding B. Those dates remain available to the
semantic agent and the weather tool's separate current-and-recent authority
gate. The correction does not add a workflow state machine or move semantic
event judgment out of the one Deep Agent.

### Clean implementation slices

Each slice stops after its focused proof and ships code and documentation at
the same breadth:

1. **Durable summary boundary — built:** migration 8 adds the projection fields
   and memory wire contract, including versioned atomic updates and the
   non-overlap retrieval invariant. Focused offline tests prove accepted
   advancement, exact retry, conflict rollback, migration preservation, and
   strictly later context-eligible raw reads without calling a hosted model.
2. **Rolling compaction and hydration — built:** memory returns a bounded oldest
   exact prefix independently of the newest prompt tail. A tools-disabled
   direct model call runs only after a successful response when configured
   thresholds are met, sees only the prior summary plus the largest fitting
   contiguous portion of that prefix, and returns one closed JSON field. A
   single oversized oldest turn uses marked head-and-tail excerpts only for
   compactor input; its durable text remains exact. Summary, exact raw
   discussion, and the product ledger render separately. Timeout, error,
   malformed output, cancellation, and a summary-only finalization conflict never
   change the response or silently advance the watermark.
3. **Weather receipt projection — built:** migration 9 and the shared
   `weather_forecast.v1` contract promote only paired validated success,
   enforce exact scope, TTL, same-scope supersession, and the four-receipt cap,
   and hydrate a separate activation/grounding lane. Activation explicitly
   binds one exact-scope receipt; unbound receipts remain non-evidence. The
   product ledger and current cart are unchanged.
4. **Focused conversation proof:** run the one three-turn live fixture without
   Judge and require search, then weather, then historical resolution plus two
   detail reads, explicit receipt binding, no weather refresh, and no repeated
   canonical forecast facts on the comparison turn. Inspect
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
| 1 | Explicitly asks `Show me dress options for a semi-formal wedding` | `outfit-styling` + `event-context`; ask `event_location` | One catalog search | Weather, product details, or ZIP disclosure |
| 2 | Supplies `NYC`, outdoor patio, and next week | Same skills; ask `none` | One weather call; receipt status `promotion_prepared` | A repeated catalog search or repeated event question |
| 3 | Compares the lacy gown with the hem satin dress | Same skills; ask `none`; bind the turn-2 exact-scope receipt | One batched historical resolution, then two detail reads; receipt status `bound` | A new search, weather refresh, or any repeated forecast condition, temperature, precipitation, resolved place, attribution, or uncertainty fact |

The deterministic fixture validator checks exact skill selection, event-question
choice, tool order and counts, required evidence, the categorical redacted
weather trace, categorical receipt lifecycle status, stable product names, and
forbidden response fragments. Receipt IDs, location/date scope, and receipt
evidence are not diagnostic oracles. Those
assertions are test oracles only; they do not participate in serving decisions.
The explicit natural dress request isolates receipt continuity from ambiguity
about whether the first turn asked to shop.

The focused offline boundary includes four activation-schema cases: an
accepted current-turn bounded date removes `event_date`, bare current-turn
`next week` shapes the schema, missing date authority keeps `event_date`, and a
prior date for Wedding A cannot narrow activation after the current turn
introduces Wedding B. The cross-event focused subset passed 168 tests with 1
expected xfail in 2.69 seconds. The final bounded offline gate passed 506 tests
with 1 expected xfail and 1 unrelated Starlette deprecation warning in 6.33
seconds. Ruff passed all changed Python.

### What a passing run proves

- one Deep Agent can carry the event-and-product thread across three turns;
- event context adds weather without replacing outfit styling;
- context fulfillment does not cause product rediscovery;
- natural prior-product references resolve through durable memory;
- both products are refreshed from current catalog details before comparison;
- the comparison binds the valid exact-scope receipt, uses it silently for
  styling, and makes no second provider call;
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

The final Slice 3 focused run used 14 application-model calls and 195,821
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
- Typed weather receipt contract:
  [`weather_receipts.py`](../shared/weather_receipts.py)
- Atomic receipt projection:
  [`conversations.py`](../memory_retriever/src/conversations.py)
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

On 2026-07-30, the first Slice 3 isolation run retained the former ambiguous
occasion wording. It made no turn-1 search, while turn 2 successfully prepared
the weather receipt and turn 3 bound it; historical resolution then correctly
failed closed because no product ledger existed. The fixture was narrowed to
the explicit natural dress request rather than changing receipt architecture.

The second one-shot Judge-free run proved turn-1 search and product-ledger
creation, but it also **failed** before receipt promotion. On turn 2 the
shopper supplied `NYC, on an outdoor patio next week`; the weather authority
parser accepted that bounded range, yet activation selected `event_date`,
asked for an exact day, made no weather call, and prepared no receipt. Turn 3
correctly resolved and read both products but repeated the same contradictory
date question. The run used 13 application-model calls, 177,264 tokens, one
text embedding, zero weather-provider calls, and 47.61 seconds. No Judge,
automatic rerun, full live evaluation, or broad unit suite followed. The
shared-parser activation-schema correction above was then implemented and
qualified with the fresh focused gate below.

That fresh full focused gate then passed all 3 diagnostic turns. Judge scored
the turns 5/5, 4/5, and 5/5, for 4.67/5 overall. The run averaged 24.50 seconds
per turn and used 14 application-model calls, 195,821 tokens, one text
embedding, and one Visual Crossing request. Turn 3 explicitly bound the
turn-2 receipt, ran the historical resolver followed by exactly two detail
reads, made zero weather and catalog-search calls, and repeated no forecast
facts. The canonical quality-and-timing comparison is
`~/exec-briefs/retail-shopping-assistant/quality/shopping/targeted/durable_context/slice3_weather_receipts/comparisons/pre_receipt_f6fe646__to__current_wip.md`.
