# Shopper Agent Architecture

This document is the short architectural map of the serving shopper agent. It
shows where product truth is published, how skills control model behavior, and
which tools connect the agent to application services.

For the source-audited serving snapshot and minimum focused validation policy
as of 2026-07-29, see
[Shopper Deep Agent Architecture — 2026-07-29](SHOPPER_DEEP_AGENT_ARCHITECTURE_2026-07-29.md).

## Architecture at a Glance

![Shopper Deep Agent architecture](images/shopper-agent-architecture.svg)

[Open the full-size SVG](images/shopper-agent-architecture.svg).

For the concise executive flow, worked example, and prioritized follow-up work,
see the [Shopper Agent Leadership Note](SHOPPER_AGENT_LEADERSHIP_NOTE.md).

## Architectural Boundaries

| Boundary | Owns | Does not own |
| --- | --- | --- |
| Published catalog | Product records, taxonomy, filter values, field roles, prices, details, and retrieval results | Shopper intent, styling judgment, cart state, or inventory |
| Deep Agents runtime | Semantic intent, skill selection, deterministic selected-skill tool grants, tool selection, styling judgment, and one server-resolved representative-shopper soft-guidance block | Product facts, policy facts, cart truth, or profile ownership |
| Memory service | Immutable representative-shopper registry and conversation binding, typed three-field shopper snapshots, ordered durable shopper/assistant turns, exact finalized replay, a versioned rolling summary, separate newest raw-turn tail and oldest exact compaction prefix, a capped projection of short-lived typed weather receipts, presented-product events and compact index, deterministic same-conversation reference resolution, authoritative cart state, and atomic mutation replay records | Catalog facts, model reasoning, learned preferences, sentiment, active anchors, summary semantics, raw provider data, or cross-conversation memory |
| Graph checkpointer | Request-scoped working graph/tool state within one chain-server process | Durable transcript storage, cross-turn shopper memory, cross-replica context, or product-ref authorization |
| Event-weather boundary | Closed location/venue/date question ordering and weather authority validation, exact shopper location provenance, optional exact-prefix region/country qualifiers, direct Visual Crossing resolution, server-owned bare-range and exact-weekday `next week` normalization, normalized current-turn forecast evidence, explicit exact-scope receipt binding, sanitized typed failures, redacted raw data plus categorical tracing, deterministic location disclosure, attribution, and uncertainty | Product facts or constraints, a rewritten shopper place, an unstated ZIP, inferred beach/outdoor/indoor/terrain setting, unbound prior forecast authority, provider-plan rights, or a public weather API |

A serving turn has one semantic procedure authority: the Deep Agent that
selects the skills, reads their complete instructions, and uses their granted
tools. The runtime does not add a post-answer semantic completion reviewer,
operation plan, or correction loop that can discard a candidate and reopen
tools. The separate tools-disabled grounding editor cannot select skills,
authorize evidence, or invoke tools; deterministic rendering remains
authoritative when it cannot safely preserve the tool evidence.

A model-authored semantic query is an internal **ranking preference**, not a
product fact or shopper-facing explanation. Only catalog tool evidence can
establish catalog facts. Conversation memory may guide judgment, but it cannot
override current tool results. Caller-supplied persona objects are not injected
into model context. The UI may send only one server-published profile ID;
memory resolves and binds it at turn start. The runtime renders the returned
type, behavior, and ZIP once as soft guidance. That context never grants a
skill/tool or establishes budget, constraints, cart intent, or product facts.

The provider-neutral `get_weather_forecast_tool` is registered read-only and
granted only by `event-context`, which composes only with `outfit-styling`.
`WEATHER_ENABLED=false` remains the default, and disabled startup and health
checks make no provider request. The request-bound wrapper accepts one
model-visible attempt on an eligible turn with bounded shopper-authored date
authority; an invalid schema consumes the attempt. The call itself additionally
requires one permitted location authority. Within a valid call,
`max_provider_attempts: 2` permits one internal retry only for timeout or HTTP
5xx. HTTP 400 maps to generic `weather_request_invalid`; other 4xx,
connection, and response-validation failures are not retried. Saved ZIP reaches
the weather client only through the
narrow deterministic current/recent confirmation gate, with both `location`
and `location_query` omitted. Otherwise the wrapper accepts one bounded exact
named-place, address, or postal-code phrase from shopper-authored text in
`location`. For an abbreviation or ambiguous name, `location_query` is
required: it must preserve that exact phrase as its first component and append
only one or two comma-separated region/country qualifiers. Keep
`location="NYC"` and use `location_query="NYC, NY"`; `Springfield, TX` is a
valid explicit regional assumption. It never adds an unstated ZIP or numeric
component and is omitted only when `location` is already sufficiently
qualified.
Semantic equivalence remains model-owned rather than deterministic proof and
is correctable through the disclosed provider resolution.
The adapter sends the bounded named place directly to Visual Crossing
Timeline, using `location_query` only in the qualifier-preserving form above;
it does not synthesize a ZIP or call a separate geocoder. Visual Crossing's
`resolvedAddress` becomes the reversible `resolved_location` assumption for
shopper-provided mode.
An explicit place, negation, uncertainty, or override rejects saved mode.
Modal lowercase `may be` is uncertainty, while calendar `May 5` remains valid.
An exact shopper phrase `<weekday> next week` requires a matching lowercase
`weekday` and is normalized from one captured UTC date to that exact day inside
the next Monday-through-Sunday window. Bare `next week` omits `weekday` and
normalizes to the full range. Missing, mismatched, mixed, negated, or
superseded weekday authority fails closed. Without a bounded shopper-authored
date signal, the skill gate hides and execution-blocks weather for that turn
and final response handling may ask only the question chosen by the accepted
activation. The server never derives a date question from weather being enabled
or date authority being absent. An unambiguous single-day phrase such as
`tomorrow` is resolved by the model against that same anchor into an exact ISO
date; other ambiguous or unresolved relative dates can yield `event_date` only
under the activation contract's enabled-and-material rule. The provider-resolved place is
disclosed only for explicit
shopper location, making the query assumption reversible, and omitted for
saved mode. No weather-specific FastAPI, SSE, or UI contract is added.

A successful same-ID call/result pair may be promoted at completed
finalization into a `weather_forecast.v1` receipt in the versioned conversation
projection. The default 3,600-second TTL is capped at 21,600 seconds; memory
prunes expiry, replaces older success for the same exact location/date scope,
and returns at most four. Failure, saved ZIP digits, raw provider
request/response data, the prepared provider endpoint URL, key, and exception
never enter the receipt; the pinned public attribution URL remains. The runtime hydrates receipts in a
separate lane from summary, raw turns, and the product ledger. Activation may
bind one only with `event-context`, `event_context_next_question=none`, and an
unchanged exact event scope. Unbound receipts never ground, a bound receipt
blocks another weather call, and changed or uncertain scope requires fresh
evidence. Current successful weather takes precedence.

Memory evaluates expiry atomically at durable turn start. The returned active
set is a validity snapshot for the in-flight request, with no second wall-clock
check mid-turn. The pre-activation model sees only each receipt's ID/type,
shopper location/date scope, and `valid_until`; it never sees forecast evidence.
Normalized evidence remains server-side and enters grounding only after
explicit binding.

## 1. Published Catalog Data Foundation

The catalog starts from two operator-published files:

- [Product JSONL](../shared/data/enriched_products.jsonl) supplies product rows
  and all observed values.
- [Field-role sidecar](../shared/data/enriched_products.schema.yaml) identifies
  record fields, ordered taxonomy fields, and whether each field supports hard
  filtering, semantic retrieval, or product details. It does not duplicate
  catalog values.

One validated snapshot supplies capabilities, product details, and the data
indexed in Milvus. `GET /capabilities` publishes the exact current taxonomy,
filter values, numeric ranges, field coverage, and supported retrieval modes.
The chain server caches the first successful contract and uses it both to
generate `search_catalog_tool` inputs and to validate every structured search.

The shopper model owns semantic catalog mapping. `search_catalog_tool` exposes
`semantic_query`, `shopper_guidance`, `requested_product_type`, `taxonomy`,
`required_constraints`, `scope_complete`, and optional `search_mode`. Catalog
capabilities generate its exact taxonomy values, hard-filter properties and
enum values, typed numeric ranges, and search modes. It exposes no
model-authored taxonomy relationship, clarification branch, or catalog-absence
result. The model decides how shopper language maps to the requested role,
semantic direction, advertised taxonomy, and constraints. Deterministic runtime
code does not compare that free-form role with shopper prose or classify it as
exact, umbrella, open, alternative, or parent scope. It validates only the
submitted structural shape and capability-owned values before retrieval:
text-versus-image requirements, one-category cardinality,
category/subcategory ownership, hard-filter fields and values, retrieval mode,
turn limits, and duplicate hard scopes. The catalog service contains no
chat/completion LLM and does not interpret shopper language, expand queries, or
choose filters. Its only model inference is text or image embedding generation;
candidate fusion, hard filtering, COSINE relevance normalization,
deduplication, and final ordering are deterministic.

Each search covers at most one catalog category and one focused requested role.
Every text search carries `requested_product_type`: the shortest product noun or
true umbrella chosen by the model from the current turn or direct antecedent,
excluding color, material, fit, occasion, weather, and style modifiers. This
field records model-authored provenance, not taxonomy or ranking text, and is
`null` only for image-only search. The model owns alternative, comparison,
ordering, negation, open-role, and parent-category semantics; the runtime does
not extract or validate those meanings from shopper prose. A typed selection of
multiple advertised subcategories from one category executes once with a
selection-wide candidate window. Rank-preserving selection keeps one returned
candidate per selected subcategory when available and trims to the configured
result count.

The `semantic_query` remains independent soft ranking direction. Capability
validation enforces exact advertised category and subcategory values and their
ownership relationship without asserting what they mean relative to
`requested_product_type`. Duplicate identity is normalized taxonomy plus hard
constraints, not semantic wording. When the model submits a category-only text
search, structured evidence records the requested role and advertised category
separately. That evidence is neutral: it does not claim that the requested role
is unavailable, that the category is its proven parent, or that a different
taxonomy relationship exists. Returned products retain their actual catalog
categories. If the model cannot select a faithful advertised scope, the
assistant asks one concise clarification directly without a tool call or a
catalog-absence claim.

The detailed contracts and implementation live in:

- [Catalog architecture](CATALOG_REFACTOR_PLAN.md)
- [Commerce contracts](COMMERCE_CONTRACTS.md)
- [Catalog loader](../catalog_retriever/src/catalog.py)
- [Capability publisher](../catalog_retriever/src/capabilities.py)
- [Catalog retriever](../catalog_retriever/src/retriever.py)
- [Chain capability cache](../chain_server/src/catalog_capabilities.py)
- [Model-visible catalog search schema and tool construction](../chain_server/src/deepagents_runtime.py)
- [Reusable model-visible catalog search rules](../chain_server/src/catalog_scope.py)

## 2. One Shopper Turn

1. The runtime scopes the request and starts a durable memory-service turn before
   guardrail, model, or tool work. That transaction returns bounded
   model-context-eligible raw turns, a compact historical-product index, the
   prior turn's selected skill names, the authoritative cart, and an opaque
   execution `attempt_id`. Blocked turns stay durable for exact replay and audit
   but are excluded from both the service projection and chain prompt formatter.
   The raw turns replace the legacy rolling context blob. LangGraph working state
   is isolated to this request under a collision-safe pair of conversation ID and
   request ID.
2. The first model step can call only `activate_shopper_skills_tool`. It selects
   the smallest registered skill set for the current intent. The prior turn's
   selected skill names, persisted with the prior terminal turn, are included
   only as a read-only continuity signal for this fresh semantic decision. They
   do not force routing, inject the old skill, or authorize commerce tools. The
   conversation still establishes intent: a
   terse item-only follow-up inside an active outfit-building or style-led
   single-piece thread remains an `outfit-styling` task. Whenever
   `event-context` is selected, this same call must also bind
   `event_context_next_question` from the current and recent shopper
   conversation: `event_location` only when destination is missing and material,
   `event_venue` only after destination is established when venue or setting is
   missing and material, `event_date` only after destination and any material
   venue are established when live weather is enabled and material and a
   bounded date is neither established nor explicitly unavailable, and `none`
   otherwise. Before constructing that typed enum, the runtime applies the
   same closed shopper-authored weather-date authority parser used by the tool
   gate to the current shopper turn only. Once that turn contains an accepted
   bounded date, including bare `next week`, `event_date` is absent and cannot
   become a contradictory follow-up. Prior raw-turn dates cannot narrow
   activation because the model owns event identity. The weather tool's
   separate eligibility boundary may still use bounded current and recent
   shopper authority. An
   explicitly shopper-stated outdoor patio, beach, garden,
   rooftop, or open-air setting makes enabled live weather material; with
   destination and that setting but no bounded date, select `event_date`.
   Skill selection, location, venue, materiality, and intent remain model-owned
   semantic guidance. The dynamic enum is typed argument consistency, not an
   intent router or keyword routing layer. The field is omitted without
   `event-context`. An invalid
   composition or next-question value returns its typed reason to the
   activation model for one correction. A second invalid selection ends the graph with a
   deterministic clarification and exposes no shopping tool. Multiple
   activation calls in one model response execute none and produce the generic
   shopping-task clarification immediately.
   For example, `Cancun` with no established setting selects `event_venue`;
   after the shopper says `on the beach`, enabled live weather is material and
   the next turn selects `event_date` when the date remains neither established
   nor explicitly unavailable.
3. The runtime validates the selection and injects the complete selected
   `SKILL.md` files. Each file declares `role`, optional `exclusive_group`, and
   `tools_granted`; only the union of those grants becomes model-visible on the
   next step. Modifier skills add their grants to that union and cannot revoke
   tools granted by a primary or standalone skill. Activation therefore cannot
   be bypassed or batched with shopping work. Each skill's required frontmatter also supplies declarative
   `response_guidance`: reviewed shopper-facing fallback framing for a search
   call whose tool evidence contains no `shopper_guidance`.
4. The model chooses only from the selected skills' granted tools. Every
   app-owned shopping dispatch independently rechecks both the frontmatter grant
   union and the immutable execution policy before its handler can run. Before
   a detail, availability, or cart operation uses a product from an earlier
   turn, the selected discovery, styling, or cart skill may call the typed
   historical resolver once with one or more exact descriptors from the compact
   index. Its 0/1/many result requires clarification for zero or many; only one
   match enters request-local product evidence. Resolution is deterministic and
   makes no catalog, embedding, or separate model call. An established
   two-product styling comparison remains inside `outfit-styling`; it does not
   activate another skill or repeat catalog search. The model submits both prior
   products in that one resolver call, then reads each unique ref through the
   scalar detail tool in separate model steps. The default two-read cap covers
   one pair. Any required zero/many resolution clarifies without a substitute
   search, and weather is optional additional evidence rather than a replacement
   for product facts. This sequence is semantic skill procedure, not a
   deterministic comparison-intent or pair-completeness gate. Before
   retrieval, every text search includes one nonempty, product-agnostic
   `shopper_guidance` sentence authored under the active skill; image-only search
   uses empty guidance. The model may submit one category-only scope when that is
   its faithful semantic choice. The resulting evidence records the requested
   role and searched category separately without asserting a parent relation or
   absence. If the model cannot choose a faithful advertised scope, it asks one
   concise clarification directly and makes no search-tool call. Search requests
   pass through the capability-derived schema and deterministic structural
   validation. Tool calls are sequential and duplicate
   taxonomy-plus-hard-constraint scopes are rejected.

   The turn receives one structural catalog-repair opportunity total. The
   runtime does not normalize shopper wording into a repair key, lock a semantic
   scope, compare a repaired noun with conversation text, or reject a correction
   because its semantic mapping changed. The isolated repair receives the
   capability-derived typed
   `search_catalog_tool`, compact server-generated Catalog capabilities, the
   current shopper message, bounded sanitized validator feedback, and the
   complete active shopper-skill instructions. Only `search_catalog_tool` is
   available and parallel calls are disabled. The repair may submit one
   corrected search or return no tool call to signal that clarification is
   needed. That no-tool response is branch/control state: the server marks it,
   discards the model prose, and emits `Could you clarify the product type or
   requirement you want me to use?`. Grounded evidence already collected in the
   turn is preserved.

   Echoed rejected arguments are stripped, and native Pydantic feedback is
   reduced to rejected top-level field names. Independently valid finite
   structural fields—advertised `required_constraints`, `scope_complete`, and
   `search_mode`—may be preserved across the repair and any restoration is
   recorded in bounded `restored_fields` diagnostics. Taxonomy and
   `requested_product_type` remain model-authored and are validated afresh; the
   runtime never rewrites them. A malformed or nonempty
   `unadvertised_requirements` lane fails closed without semantic provenance
   review. After the one repair has been used, a later invalid catalog call
   closes to synthesis. A successful partial repaired search may continue with
   later valid work, but no second repair is available in that turn. The
   configured successful-search cap remains three. A successful or zero-result
   search that consumes the final slot records `SEARCH_BUDGET_EXHAUSTED`; the
   next model step removes only `search_catalog_tool`. Product-detail,
   availability, and cart tools plus honest partial synthesis remain available.
5. Tool-role messages are the evidence boundary. For completed successful
   search-only turns, the runtime runs one final tools-disabled synthesis under
   the active skill, then grounds that draft against tool-role evidence. If the
   draft or editor is unavailable, pre-retrieval `shopper_guidance` and static
   skill `response_guidance` feed deterministic fallback. When the requested
   outcome depends on a material, fit, comfort, durability, care, weather, or
   other functional property absent from evidence, grounding must disclose the
   gap and present candidates as the closest catalog or styling direction, not
   as proven suitable. Deterministic fallback makes the same generic
   unverified-property disclosure. Before fallback
   guidance is serialized, a narrow scrub replaces documented unsupported
   outdoor/weather guarantee terms with neutral selected-role guidance. It does
   not change the semantic query, taxonomy, constraints, or executed search.
   Covered forms include outdoor-surface or outdoor-walking claims and
   constructions such as "handle rain," "work well for outdoor surfaces," or
   "stay secure for outdoor walking," plus `wet conditions` and "works well in
   wet weather/conditions."
   Results, filters, and the assistant draft are not copied into guidance after
   retrieval.
   Deterministic fallback code separately renders every returned candidate with name,
   price, category, and only the confirmed filters from that candidate's search.
   Multi-role output groups each guidance sentence with the products from its
   originating search and deduplicates candidates by `product_ref`, not display
   name. Mixed-outcome turns preserve successful product groups when another
   scope has an unsupported requirement. A fixed unsupported-requirement
   response applies only when that rejection is the sole current-turn business-
   tool outcome. Incomplete successful evidence gets a
   neutral offer to continue with the next requested piece or search scope.
   Zero-result evidence
   retains its exact search scope and cannot establish broader absence. Other
   tool-backed drafts pass through the grounding boundary before becoming
   shopper-facing text. When all current-turn business calls are rejected
   catalog searches and no current product evidence exists, a fixed retry
   response bypasses model-based editing so prior evidence cannot be recast as
   results from the rejected search.
6. The runtime finalizes the durable turn as `completed`, `blocked`, or `failed`
   on every terminal path. An exact retry of a finalized request replays stored
   assistant text, products, retrieved images, and internal diagnostics without
   another model/tool turn or finalize call. Public query responses return an
   empty diagnostics object unless a trusted operator/evaluation deployment
   explicitly sets `EXPOSE_AGENT_DIAGNOSTICS=true`. A start failure runs no agent work. A
   finalize must echo the current attempt token. Only the latest-sequence
   abandoned turn can reopen, and doing so rotates that token; a late finalize
   from the stale attempt is rejected and becomes a safe superseded-attempt
   response with no stale products or images. Other finalize failures do not
   replace an already grounded response; diagnostics record
   `memory_finalize_error`, and the request checkpoint is preserved.
   Memory-service operations are transactional, and database sessions are
   request-scoped and returned to the SQLAlchemy pool after every request.

   The Deep Agents graph and grounding editor share one configurable 45-second
   model-stage deadline. The editor receives only the remaining time. A graph
   timeout cancels the graph, captures bounded partial graph messages, clears
   products and images that were not delivered, and finalizes as failed with
   `agent_timeout`. A grounding timeout finalizes as failed with
   `grounding_timeout`; search-only evidence uses deterministic catalog
   rendering, protected context-only evidence uses deterministic event
   assembly, and current product-detail evidence uses a deterministic verified-
   detail renderer containing only current names, prices, categories, and listed
   fields, followed by a typed weather outcome when present. It accepts only a
   current tool-role result named `get_product_details_tool` that begins with the
   server's canonical successful-detail marker. It preserves facts, not a
   model-authored comparison judgment. Other non-search turns receive a fixed
   retry/cart-check response instead of the unverified draft. Ordinary
   editor errors and empty or whitespace-only output use the same evidence split
   and finalize as failed with `grounding_error`; invalid protected context-only
   output instead falls back deterministically. Only a successful durable
   finalize permits checkpoint deletion and admission of the next conversation turn. An
   already-started synchronous tool operation may finish while graph
   cancellation propagates; cart idempotency and the timeout response's
   cart-check guidance cover that narrow interval.

   Finalization derives one `candidate_set_presented` event only from the
   ordered product cards in the terminal replay output, then rebuilds the compact
   product-reference projection in the same transaction. After that durable
   commit succeeds, the runtime deletes the request-scoped MemorySaver thread.
   `CHECKPOINT_STORE=memory` is the only supported mode. The checkpointer remains
   process-local but is no longer shopper memory; it survives only a finalize
   failure and otherwise ends with the request.

The durable transcript contains raw shopper/assistant text, selected skill
names, bounded replay output, and ordered event envelopes. It does not contain
raw media, model reasoning, or the full graph/tool transcript. Product-card
output populates durable `candidate_set_presented` events and a bounded compact
product-reference index. The projection keeps the newest complete candidate sets within 16,384
serialized characters. A typed batch resolver matches exact product ref,
display name, category, turn, candidate set, and one-based position within the
current conversation. It is enforced at most once per turn and returns
`resolved`, `ambiguous`, or `not_found`; only a unique result becomes
request-local evidence. Active anchors and effective preferences remain
reserved. Fuzzy/embedding lookup, preference or sentiment extraction,
cross-conversation lookup, and stale-catalog-revision handling are not included.

Separately, migration 5 creates `shopper_profiles`. Startup validates and
immutably bootstraps exactly five eval-derived rows from shared memory-service
configuration. The memory service exposes read-only list/get routes, and the
chain server provides the typed list proxy used by the UI. No profile write
route exists. The bundled UI gates a new session on an explicit dropdown choice
of Guest mode or one of the five rows. Selecting another mode remounts the chat
surface and rotates the browser identities; it does not restore a
profile-specific cart or transcript. Reset keeps the selected mode while
starting a clean conversation.

Migration 6 adds nullable `conversation_turns.shopper_profile_id` with
`ON DELETE RESTRICT` and `ON UPDATE RESTRICT`; `NULL` is the explicit Guest
binding. Within the existing `BEGIN IMMEDIATE` turn-start transaction, memory
resolves a selected row, rejects an unknown ID without inserting a turn, and
prevents Guest-to-profile, profile-to-Guest, or profile-to-profile reuse of one
conversation. The selected ID participates in exact request identity. A valid
start or finalized replay returns:

```text
SHOPPER CONTEXT (server-resolved; soft guidance only):
shopper_type: <type>
behavior: <exact behavior>
saved_zipcode: <five-digit ZIP>
END SHOPPER CONTEXT
```

The block and its profile-specific precedence/non-authority prompt rules are
absent for Guest. The block contains no profile ID or display name and is not
written into raw shopper/assistant text or repeated in recent history.
Explicit current-turn instructions win, followed by explicit recent
conversation preferences. Static profile behavior is third-priority interaction
guidance only. Saved ZIP is neither proof of current/event location nor a
weather or product requirement, shopping or shipping destination, or
availability signal. The `event-context` modifier may use it only as a
tentative event-location candidate beside `outfit-styling`; a shopper-stated
destination or venue wins. When ZIP is the only location clue, the helper asks
whether to plan around the shopper's usual area or elsewhere rather than
requesting a city from scratch. The server releases that ZIP to weather only
for a current location-neutral statement explicitly naming `my`/`the`
usual/home area, a bare affirmative immediately after its usual/home-area
question, or an immediate strict date-only follow-up to an accepted
confirmation. Any explicit current place, address, postal code, question,
negation, uncertainty, or location override rejects saved mode. An exact
shopper-authored place may instead authorize explicit-location weather mode
without a representative ZIP. The modal phrase `may be` is uncertainty, while
calendar `May 5` remains valid. The final response editor receives only whether
that candidate exists, never ZIP digits. There are no shopper-type-specific
code branches. Registry bootstrap and turn-start serialization both require
the behavior summary to remain one trimmed line. Guest request digests preserve
their pre-profile canonical shape so older finalized Guest turns still replay
after migration 6.

Migration 8 adds `summary_text` and `summary_through_sequence` to the
conversation projection. They are a versioned pair updated only through an
optional compare-and-swap payload on turn finalization. Memory returns only
context-eligible raw turns strictly after the watermark, so accepted summary
coverage and the raw tail do not overlap. A stale projection version or invalid
boundary rolls back the complete finalization.

Those additive lanes are available only through opt-in turn-start response
contract 2. An unversioned caller receives the exact earlier response shape and
a bounded raw tail from sequence zero. The current chain treats a missing
contract marker as version 1 and suppresses optional summary and receipt writes
for that turn. Equivalent database defaults on fresh and upgraded SQLite
schemas preserve old memory inserts during rollback.

The serving runtime now renders four separate lanes: semantic summary, exact
newest raw discussion, the historical product index, and active typed weather
receipts. Memory also returns
the total unsummarized count and a separate oldest exact prefix of up to four
turns. Only that prefix can feed the compactor; the newest prompt tail never
sets a watermark. With default policy, six unsummarized turns trigger one
tools-disabled direct model call after the successful response and before
finalization, at least two turns remain raw, output is capped at 4,096
characters, and the call has an independent five-second timeout. Its closed
one-key JSON result becomes visible only to the next request.

The compactor input contains only the prior summary and exact offered prefix.
It excludes the current request, representative profile/ZIP, cart, product
ledger, receipt projection, media, tool messages, diagnostics, and request
identity. Canonical
prior forecast blocks are redacted. An input that cannot fit without truncating
a durable turn fails open, as do timeout, error, malformed output, blocked or
failed turns, and cancellation; memory retains every raw turn and the previous
watermark. A summary-only compare-and-swap conflict gets one finalize retry
without the update and no model rerun.

Migration 9 adds `active_receipts_json` to that same versioned conversation
projection. Receipt promotion, expiry pruning, same-scope supersession, the
four-receipt cap, summary advancement, product-index rebuild, replay output,
and turn status share the existing atomic finalize transaction and attempt
fence. Turn start exposes only validated, unexpired active receipts. The chain
server does not blend them into the compactor or transcript formatter; it
renders a separate minimal receipt index for activation. Full evidence remains
server-side until one receipt is bound for grounding.

The resolved agent dependency boundary remains `deepagents==0.6.12`,
`langchain==1.3.11`, `langgraph==1.2.7`, and `langgraph-sdk==0.4.2`.
Services that resolve `orjson` pin `3.11.5`, the last upstream release limited
to the project's Apache-2.0/MIT policy. Redis checkpoint packages are not
installed.

Final-response extraction ignores tool messages, assistant tool-call messages,
and internal activation markers. If no shopper-facing answer remains, the
runtime returns a safe retry response and records `incomplete_agent_response`.
Operator diagnostics also include bounded `catalog_scope_outcomes` for
zero-result scopes. Standard Compose keeps the unauthenticated memory API on the
private service network and host loopback; it is not a public application API.

Product refs used for detail, availability, and cart-add calls live only in the
current request's evidence set. Current-turn search adds them directly; a
unique durable historical resolution can add an earlier presented product.
Ambiguous or missing references never authorize a downstream tool. The current
slice records catalog revision metadata but does not reject stale revisions.

Cart transaction safety is owned by the memory service. Adds use catalog
`product_id`; removes and quantity updates use opaque `cart_line_id`. Each
mutation commits with its owner-scoped idempotency record, so an identical retry
replays and conflicting key reuse cannot change the cart.

The serving implementation is split across the
[Deep Agents runtime](../chain_server/src/deepagents_runtime.py),
[conversation-memory client](../chain_server/src/conversation_memory.py),
[conversation-product boundary](../chain_server/src/conversation_products.py),
[shopper-profile read boundary](../chain_server/src/shopper_profiles.py),
[shopper tool policy](../chain_server/src/tool_policy.py),
[skill activation boundary](../chain_server/src/skill_activation.py), and
[tool-loop controller](../chain_server/src/tool_loop_control.py). The durable
SQLite boundary is implemented by the memory service's
[conversation API](../memory_retriever/src/conversations.py),
[shopper-profile registry](../memory_retriever/src/shopper_profiles.py),
[product-reference resolver](../memory_retriever/src/product_references.py),
[models](../memory_retriever/src/models.py), and
[migrations](../memory_retriever/src/migrations.py).

## 3. Skills and Their Tools

Six shopper skills are registered. Their frontmatter grants are an
authorization boundary: the model sees only the union for the current selected
skills, and dispatch checks the same grant against an independent immutable
policy. Tool schemas and wrappers still enforce refs, filters, service state,
and mutation preconditions.

| Skill | Role | Use | Granted tools |
| --- | --- | --- | --- |
| [`product-discovery`](../chain_server/skills/shopper/product-discovery/SKILL.md) | `primary`, `product_procedure` | Non-styling search, browse, filters, and product facts | `search_catalog_tool`, `get_product_details_tool`, `check_product_availability_tool`, `check_active_promotions_tool`, `resolve_conversation_products_tool` |
| [`outfit-styling`](../chain_server/skills/shopper/outfit-styling/SKILL.md) | `primary`, `product_procedure` | Build, complete, compare, balance, or refine a look | `search_catalog_tool`, `get_product_details_tool`, `check_product_availability_tool`, `check_active_promotions_tool`, `resolve_conversation_products_tool` |
| [`event-context`](../chain_server/skills/shopper/event-context/SKILL.md) | `modifier` | Use stated event destination/venue context, fetch qualified current event weather, bind a valid exact-scope receipt, or resolve material missing context before styling branches; treat saved ZIP as tentative | `get_weather_forecast_tool`; combine only with `outfit-styling` |
| [`budget-shopping`](../chain_server/skills/shopper/budget-shopping/SKILL.md) | `modifier` | Add budget procedure when the shopper states a price ceiling or bundle budget | None; combine with the applicable product or cart skill |
| [`cart-management`](../chain_server/skills/shopper/cart-management/SKILL.md) | `standalone` | Cart reads, adds, removals, and quantity changes, alone or beside product work | `get_cart_tool`, `view_cart_total_tool`, `add_cart_items_tool`, `remove_cart_item_tool`, `update_cart_items_tool`, `resolve_conversation_products_tool` |
| [`store-policy-answers`](../chain_server/skills/shopper/store-policy-answers/SKILL.md) | `standalone` | Returns, shipping, sizing, payment, price matching, and gift cards | `get_store_policy_tool` |

`product-discovery` and `outfit-styling` are mutually exclusive primary
procedures. `budget-shopping` modifies the applicable primary only when the
shopper states a budget. `event-context` modifies only `outfit-styling`. It is
selected whenever destination/venue context is stated or the response would
otherwise ask about or branch on a missing destination/venue. On explicit
plan-before-products turns, missing material context produces exactly two short
sentences and may include only the activation-selected location, venue, or date
question. With context complete, it produces one short paragraph and asks no
further event-context question. It gives explicit event context precedence over saved
ZIP and alone grants the read-only weather tool. Ordinary shop-now occasion
requests that do not explicitly ask for a complete look or name multiple roles
run one search for one grounded requested or core role. If location is still
missing and materially changes the next recommendation, activation may select
only event location alongside the results; with a saved-ZIP candidate, that
means asking whether the event is in the shopper's usual area or elsewhere
rather than a bare destination question. Once destination is established,
activation may instead select one material venue/setting question. Destination
never implies beach, outdoor/indoor setting, or terrain. A genuine multi-intent turn
activates every needed skill once; for example, event styling under a budget
with an explicit add request uses `outfit-styling`, `event-context`,
`budget-shopping`, and `cart-management`.

Event context composes additively with outfit styling. Activation owns only the
typed next-question boundary plus optional binding of one currently listed
receipt. Receipt binding is valid only with
`event_context_next_question=none` for the exact same event location/date
scope; a correction, uncertainty, or refresh request omits it. The server never
infers a question from enabled weather or missing context. Before activation,
the tool gate's closed shopper-authored weather-date authority parser also
shapes the next-question enum from the current shopper turn only. An accepted
current-turn bounded date removes `event_date`; a date only in prior raw turns
does not. This keeps typed arguments consistent without deciding skills, event
materiality, location, venue, or shopper intent. The weather tool may still use
bounded current and recent shopper authority under its separate eligibility
gate. The event helper may hide and
execution-block only `get_weather_forecast_tool` when location, venue, or
bounded date authority is not sufficient for a qualified forecast or when a
valid receipt is bound. It never
hides or denies catalog search, product details, historical-product
resolution, availability, promotions, cart, or policy tools, and it never
closes the overall tool loop. Product, cart, and policy work remains owned by
the selected primary and standalone skills.

The successful event-context activation result, the catalog-search tool
description, and the outfit-styling skill repeat a compact model-visible
boundary. A reply that only supplies the destination, venue, or date requested
in the prior response is context fulfillment: retain established candidates
and do not repeat non-weather product work. Explicit same-turn comparison,
refinement, replacement, search, check, cart, or policy work continues through
the normal selected skill. This is semantic procedure guidance, not a
deterministic intent router or execution gate; tool grants and dispatch
authorization remain unchanged.

Protected event decision rendering is selected from typed evidence, not from an
activation action. It applies only when event context is active, there is no
current non-weather business-tool activity, and a current typed weather outcome
or explicitly bound valid receipt exists. Missing-location/venue or an empty
draft skips the decision editor. A separate prior-candidate fallback uses
deterministic event assembly only when the draft is empty. Product comparison
with current resolution/detail activity remains on ordinary grounding; a bound
receipt may guide styling but does not repeat its exact facts. Other protected
weather-evidence turns give a narrow tools-disabled decision editor
only bounded shopper-authored event text and the server-owned deterministic
weather styling direction; the server renders exact prior names, the accepted
question, and a current typed weather failure or current canonical success
block. Any current
non-weather business-tool activity or evidence uses the normal grounding path,
so comparisons, product details, cart work, and policy answers cannot be
replaced by a context-only weather response. After successful weather, that same
activity prevents the response postprocessor from restoring unrelated names
from the historical-product index.

A forecast gets one model-visible attempt on an eligible date-bearing turn and
requires one bounded exact named-place, address, or postal-code phrase or the
narrow saved-area confirmation gate. Consuming the weather attempt closes only
the weather capability for that turn; it does not close other granted tools.
`location` retains that exact authority phrase;
for an abbreviation or ambiguous place, `location_query` is required: it must
preserve that phrase as its first component and append only one or two
comma-separated region/country qualifiers. Keep `location="NYC"` and use
`location_query="NYC, NY"`; do not rewrite the authority phrase or add an
unstated ZIP or numeric component. Omit the query only when `location` is
already sufficiently qualified. Semantic equivalence remains model-owned and
correctable through provider resolution.
The adapter sends that bounded named place directly to Visual Crossing
Timeline and uses no representative-ZIP table or separate geocoder. Within a
valid tool call it may retry only a timeout or HTTP 5xx once; HTTP 400 remains
a generic invalid-request outcome rather than location proof.
The exact phrase `<weekday> next week` requires a matching lowercase `weekday`
and is resolved server-side from one UTC anchor to that day inside the next
Monday-through-Sunday window. Bare `next week` omits `weekday` and resolves to
the full range. Missing, mismatched, mixed, negated, or superseded weekday
authority fails closed. An unambiguous single-day phrase such as
`tomorrow` is resolved by the model against that same anchor into an exact ISO
date; for another ambiguous or unresolved relative date, only accepted
`event_date` may authorize clarification. A
schema-invalid invocation consumes the attempt. An explicit destination
prevents fallback to saved ZIP. Current successful evidence has precedence;
otherwise only one explicitly bound, unexpired exact-scope receipt can ground
reuse. Every unbound receipt is non-evidence, and changed or uncertain event
scope requires fresh weather. The provider-resolved place is included and
disclosed for shopper-provided location, making the query assumption
reversible, but omitted in saved-ZIP mode.
Prior durable assistant forecast summaries are redacted from graph and
grounding-editor recent discussion while remaining stored and exactly
replayable, prior weather tool messages are excluded from prior evidence, and
the complete grounding-editor prompt replaces the selected profile's saved ZIP
before the editor call.
Current successful rendering appends one exact canonical block with every validated
daily date, condition, available low/high temperature, precipitation
probability/types, Visual Crossing attribution, and forecast uncertainty.
Normal grounding removes weather-domain fact language or fact-shaped
dates/values that lack current weather evidence while preserving ordinary
grounded styling language. Protected context-only rendering accepts no
shopper-facing editor prose. Its decision must be exact two-key JSON with one
exact shopper-authored venue quote and one or two distinct allowlisted
adjustment codes; malformed, ungrounded, extra-key, duplicate, or unknown-code
output falls back. The server maps valid codes to fixed phrases and
deterministically assembles exact newest prior names, the fixed venue sentence,
its weather direction, only the accepted question, and a current typed failure
or current canonical success block. Product comparison using a bound receipt
strips exact forecast facts and does not repeat that earlier block. Weather remains styling context rather
than product-performance proof or an implicit catalog constraint.
Search-bearing event-context turns must preserve at least one exact returned
product in final text; if the editor omits every candidate, the deterministic
grounded catalog renderer restores them.
The styling skill owns fashion procedure and clarification: anchors, color,
proportion, silhouette, formality, occasion, texture, and concise explanation.
The catalog publishes the advertised taxonomy and filter contract. The
capability-aware runtime validates and maps submitted constraints before the
catalog service performs deterministic retrieval and filtering. Cart reads and
mutations are available to styling only through a co-active `cart-management`
skill.

Slice 0 proves selected-skill tool authorization, not explicit mutation intent.
Selecting `cart-management` currently grants its cart tools; deterministic refs
and service preconditions still apply, but a server-owned current-turn intent
authorization object is a later slice.

[trends-current.md](../chain_server/skills/shopper/trends-current.md) is a
read-only seasonal reference used by `outfit-styling`; it is not a registered
skill and is not catalog truth.

## 4. Tool Ownership

| Tool group | Tools | Source of truth |
| --- | --- | --- |
| Catalog | `search_catalog_tool`, `get_product_details_tool` | Active published catalog snapshot |
| Conversation products | `resolve_conversation_products_tool` | Durable same-conversation `candidate_set_presented` events in the memory service |
| Availability boundary | `check_product_availability_tool` | Known-ref, category-aware no-I/O application stub; live inventory remains unavailable |
| Promotions boundary | `check_active_promotions_tool` | Global no-I/O application stub; currently reports no active promotion configured through the assistant |
| Cart | `get_cart_tool`, `view_cart_total_tool`, `add_cart_items_tool`, `remove_cart_item_tool`, `update_cart_items_tool` | Memory service cart |
| Policy | `get_store_policy_tool` | Operator-managed policy YAML |
| Event weather | `get_weather_forecast_tool` | Request-bound provider-neutral client plus an explicitly bound valid `weather_forecast.v1` receipt; Visual Crossing is the first adapter |

`activate_shopper_skills_tool` is an internal control tool, not a commerce
tool. It is forced at turn start, selects static behavior instructions, and
binds `event_context_next_question` when that modifier is selected. It may also
bind one currently valid exact-scope `weather_receipt_id` only when that
question is `none`. The
accepted value is the only event-context follow-up boundary passed to final
response handling. The selected grant union remains additive; the event helper
can gate only its own weather capability and cannot narrow product, cart, or
policy grants. The control tool does not read or mutate catalog, cart, or
policy state.

Raw weather arguments and output are redacted from diagnostics and failed-turn
partial graph capture. The weather call record retains only categorical
`request_shape`, `location_source`, `provider_input`, and `outcome`, with no
place, ZIP, date, resolved place, URL, body, or exception.
Receipt diagnostics add only a categorical lifecycle value; they never expose
the receipt ID, scope, or evidence.
Saved profile ZIP is recursively scrubbed from diagnostic string keys and
values. Final forecast summaries remain durable
assistant text, but recognized summaries are redacted from later graph and
grounding-editor recent discussion.
Before provider calls are enabled for shoppers, the operator must confirm that
the selected Visual Crossing plan permits the intended attribution, display,
storage, and sharing, including durable summaries and downstream app-model and
output-guardrail processing.

For exact input schemas, risk classes, and failure behavior, use the
[Shopper Agent Tool Registry](SHOPPER_AGENT_TOOL_REGISTRY.md). For skill
selection and tuning details, use the
[Shopper Agent Skill Registry](SHOPPER_AGENT_SKILL_REGISTRY.md).
