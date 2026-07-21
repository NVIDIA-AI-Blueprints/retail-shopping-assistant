# Project Status

Updated: 2026-07-21

## Current Milestone

The current working tree extends the shopper-serving Deep Agent architecture:

- a single memory-service SQLite replica now starts each turn durably before
  guardrail/model/tool work, returns bounded finalized raw turns plus the
  authoritative cart, and finalizes every completed, blocked, or failed
  outcome. An exact retry of a finalized request replays its stored response
  without another model turn. Finalized ordered product cards create durable
  `candidate_set_presented` events and a compact reference index. One exact
  typed batch resolver returns 0/1/many same-conversation matches; only a unique
  match becomes request-local evidence, while zero or many require
  clarification. MemorySaver is request-scoped with a collision-safe pair of
  conversation ID and request ID, deleted after successful durable
  finalization, and preserved on finalize failure. The compact reference index
  is capped at 16,384 characters and resolution is enforced at most once per
  turn;
- five shopper skills are registered for product discovery, outfit styling,
  cart management, budget shopping, and controlled store-policy answers. A
  required first model step selects the smallest applicable set each turn; the
  runtime then injects the complete selected files. Their frontmatter declares
  `role`, optional `exclusive_group`, and `tools_granted`; only the selected
  grant union becomes model-visible, and dispatch rechecks it against an
  independent immutable policy before invoking a shopping handler. Registry
  tools, frontmatter grant pairs, and policy pairs must match exactly at startup.
  The previous turn's selected names are a read-only continuity signal
  for that fresh semantic decision; they neither force routing nor unlock tools.
  Terse item-only follow-ups inside an active outfit-building or style-led
  single-piece thread still select `outfit-styling` from conversation context.
  Outfit styling is a focused fashion procedure for anchors, clarification,
  color, proportion, silhouette, formality, occasion, texture, and response
  judgment. It retains search, details, availability, and same-conversation
  product-resolution grants; catalog transport and validation remain with the
  catalog boundary, while cart work requires the cart skill. Product discovery
  and cart management also grant the resolver for their own historical product
  references.
  Each skill also declares product-agnostic `response_guidance` in
  frontmatter as a fallback. Each catalog call supplies required pre-retrieval
  `shopper_guidance` authored under the active skill. Completed successful
  search-only turns receive one tools-disabled synthesis under that skill and
  then the grounding editor; deterministic candidate formatting remains the
  fail-closed fallback when synthesis or editing cannot produce an answer. Selection and
  response metadata are regenerated from current files rather than retained in
  the request checkpoint;
- the runtime has a ten-tool shopper registry plus one internal skill
  activation control tool. A turn receives only the tools granted by its
  selected skills. The shopping tools cover cart quantity update, controlled
  policy lookup, a category-aware no-I/O availability stub, and deterministic
  durable same-conversation product resolution;
- memory-service schema migrations, turn start/finalize/replay, bounded
  recent-turn reads, and cart snapshots use transactional SQLite operations;
  stale active turns are recovered during startup and atomically at the next
  start. Only the latest abandoned sequence can reopen: it retains its request
  ID for cart idempotency but rotates a service-issued attempt token. A stale
  finalize is rejected and converted to a safe response without stale products;
  other finalize outages preserve the grounded response and add
  `memory_finalize_error` while retaining the request checkpoint. Successful
  finalization atomically records presented products and then deletes the
  checkpoint. Cancelled runtime turns are finalized before
  cancellation propagates; database sessions remain request-scoped and are
  always returned to the SQLAlchemy pool after successful and failed API
  requests;
- dependency resolution retains `deepagents==0.6.12`, `langchain==1.3.11`,
  `langgraph==1.2.7`, and `langgraph-sdk==0.4.2`. The services that resolve
  `orjson` pin `3.11.5`, the last release limited to the Apache-2.0/MIT policy;
  Redis checkpoint packages remain absent;
- caller-supplied persona data is not injected into model context. Persona
  support remains deferred until it has a typed, bounded schema, authenticated
  ownership, and input-safety validation; and
- both response paths expose additive agent diagnostics for activated skill
  files, ordered tool calls and arguments, rejected/duplicate calls, bounded
  current-turn product evidence from successful catalog search/detail results,
  bounded no-direct/zero-result catalog scope outcomes, and final termination.
  Failed graph messages are captured from the current checkpoint before cleanup
  deletes it. Pre-activation and same-batch shopping calls are execution-blocked
  and reported as `skill_activation_required`; post-activation calls outside
  the selected grant union are blocked as `skill_tool_not_granted` before their
  handlers run.

The schema-driven catalog refactor is implemented:

- the active feed is a 205-product JSONL catalog with a field-role sidecar;
- all 205 current product names existed in the prior 218-row CSV; the migration
  removes 12 unique legacy names, collapses one duplicate row, and primarily
  improves how existing products can be discovered and verified;
- startup builds one validated snapshot for indexing, capabilities, filtering,
  product details, and index-rebuild fingerprinting;
- raw Milvus COSINE scores are normalized to `[0, 1]` relevance scores before
  applying the configured similarity threshold;
- catalog values and category-specific field availability are derived from the
  active rows rather than hard-coded in application logic;
- the catalog publishes fields, values, ranges, coverage, taxonomy scopes, and
  retrieval modes through `GET /capabilities`;
- the chain server caches the first successful capability response for its
  process lifetime, reuses the full object for deterministic validation, and
  sends the LLM a compact projection rather than refetching the API each turn;
- every agent catalog call requires one `semantic_query`, one product-agnostic
  `shopper_guidance`, a nullable `requested_product_type`, one `taxonomy_status`,
  a `taxonomy` envelope, `required_constraints`, and `scope_complete`. The
  strict runtime model requires nonempty `shopper_guidance` for every search
  except `image_only` and `no_direct_catalog_match`, which require it to be
  empty. The
  agent-facing structural transport schema is separate from the strict runtime
  semantic model, so cross-field failures reach capability-aware validation and
  produce capability-derived feedback. For text searches,
  `requested_product_type` is the shortest product noun or true umbrella from
  the current turn or direct antecedent. Color, material, fit, occasion,
  weather, and style modifiers stay out of it. It is null only for image-only
  search and is neither taxonomy nor ranking text. Literal validation may bind
  the longest exact advertised suffix in a modifier-bearing model phrase, such
  as `waterproof boots` to `boots`, but never applies that shortcut to explicit
  alternatives containing `and`, `or`, `/`, or `&`; `closed shoes or boots`
  remains model-owned alternative or umbrella reasoning. The
  allowed `taxonomy.category` and `taxonomy.subcategory` values are generated
  from the cached catalog capabilities rather than application taxonomy. Each
  call accepts at most one category. `agent_selected_type` selects exactly one
  advertised subcategory only when the shopper named no type for that role;
  alternatives, confirmations, comparisons, and follow-ups count as named
  types. For an open role, the runtime derives duplicate
  `requested_product_type` provenance from the selected subcategory and retains
  `agent_selected_type`. Invalid open-role provenance is rejected rather than
  silently reinterpreted. The model owns `taxonomy_status`; runtime never
  semantically rewrites it. Capability-owned exact category/subcategory
  relationships validate the submitted status and selection. For a malformed
  agent-selected open-role call, deterministic validation stops
  before retrieval and reports the exact eligible subcategories from the
  current capability contract. The model operating under the active skill still
  chooses the semantic role; validation does not choose it for the model.
  Executable text search
  requires at least one taxonomy value; image-only and no-direct no-retrieval
  requests use empty arrays;
- the optional search-mode enum is generated from the catalog's advertised
  retrieval modes, and an explicit unknown or unsupported mode stops before
  retrieval instead of becoming an automatic default;
- the chain maps those generic taxonomy roles to the catalog-advertised field
  names, infers owning categories for a valid subcategory-only selection,
  rejects incompatible category/subcategory selections, and applies the result
  as deterministic hard filters;
- `no_direct_catalog_match` is selected from the requested product type alone,
  carries empty taxonomy and no hard constraints, and performs no retrieval. An
  unsupported modifier does not erase an advertised type, while subjective
  style remains semantic direction;
- duplicate search identity is normalized taxonomy plus hard constraints, not
  semantic wording. Repeating that identity is stopped even when the query is
  paraphrased. A shopper-named product scope also executes at most once per
  turn, so an adjacent taxonomy cannot replace a successful first search;
- repair accounting uses the full normalized `requested_product_type` phrase,
  with server-derived keys that keep distinct advertised siblings separate.
  Each scope receives one total repair. A schema correction or a fresh
  constraint-provenance review can consume that shared budget; constraint
  feedback returned by an in-flight schema repair closes the loop for synthesis
  rather than opening another repair. The repair is an isolated model phase:
  its concise, schema-generic system prompt replaces the base runtime prompt,
  while the skill gate retains the complete active shopper-skill instructions.
  Only `search_catalog_tool` is exposed and forced, and parallel calls are
  disabled. Active-skill responses containing more than one shopping tool call
  are rejected before execution. Its messages contain only the current shopper
  message plus bounded, sanitized validator feedback in a separate Human data
  message.
  Echoed rejected arguments are stripped. Native Pydantic feedback is reduced
  to rejected top-level field names, unbounded requested-scope text is not
  replayed, and invalid AI/tool history and full conversation history are absent.
  For native tool-transport failures, the scope is locked only when current or
  recent shopper text grounds it; an ungrounded model-generated scope may be
  corrected. A rejected change to a locked scope is removed before execution
  and recorded in `agent_diagnostics` as `repair_scope_changed`.
  A strict request failure with an independently valid constraint object
  snapshots its capability-validated advertised `required_constraints`
  privately. Isolated repair feedback supplies that exact finite constraint
  object, including an explicit empty object, while excluding free-form rejected
  arguments. The repaired call must preserve it exactly. Before it reaches
  tool execution, runtime restores only independently valid finite fields from
  the rejected call: a validated taxonomy relation, canonical advertised
  `required_constraints` (including an empty object), valid `scope_complete`
  and explicit `search_mode`, and a singleton exact/agent-selected
  `requested_product_type`. This is value preservation, not shopper-language
  interpretation: the model still owns every semantic correction required to
  make the rejected field valid. Accepted product-phrase
  normalization retains the same lock, and list-valued filters compare
  canonically; omitted optional defaults equal explicit empty values. A
  no-direct repair may clear constraints only while remaining no-direct; a
  repair that changes to retrieval must preserve the original advertised
  constraints.
  Native enum failures on `agent_selected_type` include the matching provenance
  rule in that same feedback, so one repair can correct both transport and
  relation errors. Shopper-named advertised subtypes repair to exact provenance;
  named umbrellas and alternatives repair to member provenance. A terminal
  no-direct result after repair keeps its specific
  not-advertised shopper response.
  Native failures confined to `required_constraints` include only finite,
  validated taxonomy status and selection in repair feedback; free-form scope,
  query, and guidance remain excluded while scope is compared privately.
  Independently valid locked fields are restored before execution and reported
  in diagnostics as bounded `restored_fields` names without values. Any drift
  outside that finite restoration boundary still closes before execution and
  is classified as `repair_scope_changed`, `repair_relation_changed`, or
  `repair_constraints_changed`.
  A native schema-invalid call with any nonempty
  `unadvertised_requirements` lane closes without repair. Schema-valid,
  genuinely open `agent_selected_type` requests retain the bounded review for
  proposed inferred requirements.
  Every unadvertised requirement on a shopper-stated product scope fails closed
  before retrieval, including a synonym rather than the shopper's exact
  wording. The bounded constraint review is reserved for a proposed inferred
  requirement on a genuinely open `agent_selected_type` role when its shared
  repair budget remains. That review freezes requested type, taxonomy status,
  taxonomy, completion state, `search_mode`, and all advertised hard
  constraints. Within that preserved hard scope, it may correct only the soft
  `semantic_query`, the reviewed unadvertised-requirement lane, and its
  associated guidance; it either copies the shopper's shortest exact wording or
  removes the inferred requirement. Exact wording fails closed. Removal scrubs
  product-attribute guidance. If a
  runtime semantic open-role schema repair removes its proposed inferred
  requirement, runtime replaces the submitted pre-search guidance with neutral
  generic guidance for the selected role. Unresolved provenance after review
  also fails closed. A successful partial search may continue with another
  valid role and its own single repair opportunity. A
  second invalid call in the same scope, completed scope, unsupported
  requirement, or deterministic stop closes the tool loop; the configured turn
  cap remains three searches. A successful or zero-result search that consumes
  the final configured slot records `SEARCH_BUDGET_EXHAUSTED`; the next model
  step removes only `search_catalog_tool`. Product details, availability, cart
  work, and honest partial synthesis remain available. Successful search
  evidence preserves
  the model-authored semantic query as independent internal ranking direction
  and the pre-retrieval `shopper_guidance` as product-agnostic response framing.
  Before that guidance becomes deterministic shopper-facing evidence, a narrow
  scrub replaces documented unsupported outdoor/weather guarantee terms with
  neutral selected-role guidance. Search semantics, taxonomy, hard constraints,
  and retrieval are unchanged. Covered forms include outdoor-surface or
  outdoor-walking claims and constructions such as "handle rain," "work well
  for outdoor surfaces," or "stay secure for outdoor walking," plus `wet
  conditions` and "works well in wet weather/conditions."
  Deterministic code groups each pre-retrieval guidance sentence with the
  products from its originating search, then renders every returned name, price,
  category, and per-search filter group separately. Candidate groups deduplicate
  by `product_ref`, not display name. Mixed-outcome turns keep successful groups
  when another scope has no direct match or an unsupported requirement.
  A fixed no-direct or unsupported canned response is used only when that
  rejection is the sole current-turn business-tool outcome.
  Incomplete successful evidence receives a neutral continuation note.
  Zero-result evidence retains the exact advertised taxonomy and filter scope
  and cannot support broader absence claims. Turns containing only rejected
  catalog searches and no current product evidence return a fixed retry response
  before model-based editing, so prior evidence cannot become claimed results;
- final-response extraction excludes tool messages, assistant tool-call
  messages, and internal activation markers. A completed graph with no
  shopper-facing answer returns a safe retry response and records
  `incomplete_agent_response`;
- multi-search response assembly keeps each search's confirmed filters attached
  to that search's returned products instead of flattening filters across one
  global candidate list.

The catalog service performs embeddings, filtering, and deterministic retrieval
only. It makes no chat/completion call or shopper-language interpretation.

The concise serving map is
[Shopper Agent Architecture](docs/SHOPPER_AGENT_ARCHITECTURE.md); catalog detail
lives in [Schema-Driven Catalog Architecture](docs/CATALOG_REFACTOR_PLAN.md).

## Verification

The newest focused gate is recorded first. Older Slice 0, Slice 2, and Slice 3
results remain below as comparison points; generated quality and timing
artifacts stay in the required local archive rather than versioned source.

- Focused Slice 5 durable-product gate: 77 passed and 1 existing expected
  failure. Coverage includes server-derived `candidate_set_presented` events,
  compact projection rebuilds, exact 0/1/many typed resolution, strict
  clarification, skill grants and immutable policy, request-local evidence, and
  restart-safe memory-service lookup. The 14 focused runtime lifecycle tests
  also pass for collision-safe conversation/request checkpoint isolation,
  deletion only after successful durable finalization, and preservation on
  finalize failure.
  No full offline suite or live/Judge evaluation has been run for this slice.
- Focused Slice 4 durable-turn gate: 74 passed in 2.85 seconds. Coverage joins
  the new start/finalize/replay and crash-recovery boundary, existing memory
  cart behavior, the typed chain-server client, and serving lifecycle tests.
  Ruff, Compose configuration validation, and whitespace checks pass. The test
  run emitted one pre-existing `StarletteDeprecationWarning`. No live or
  48-turn evaluation was run for this storage-only slice.
- Focused category-aware availability slice: 19 passed. Coverage includes
  general availability, apparel size, footwear size through the category
  fallback, one-size accessories, unknown conversation refs, unchanged tool
  registration, and unchanged skill-policy grants. The runtime test emitted
  the pre-existing `StarletteDeprecationWarning`. No live or 48-turn evaluation
  was run for this narrow deterministic stub change.
- Focused Slice 2 skill/authorization gate after restarting the local runtime:
  34 passed and 2 strict expected failures in 2.89 seconds, with one pre-existing
  `StarletteDeprecationWarning`. At that slice, the expected failures tracked
  constraint provenance and not-yet-built historical-reference resolution.
- Targeted `conv_5.yaml` styling continuity gate: all five turns activated only
  `outfit-styling`, exposed only the current `skill_names` activation field,
  completed without fallback, and were judged 3.2/5
  (scores: 4, 3, 3, 2, 4). Mean turn latency was 3.615 seconds. The critical
  "What bottoms go well with that?" turn retained the beige-top anchor, searched
  the complete advertised bottoms scope, and scored 3/5. That scope currently
  contains only skirts; the Judge's request for trousers, jeans, and shorts is
  a Golden/catalog mismatch rather than a taxonomy omission. The shoe follow-up
  scored 2/5 because its products did not coordinate well with the established
  beige-top-and-skirt look even though the catalog advertises neutral footwear.
- An earlier 3.4/5 result is invalid as evidence for this commit: the chain
  process predated the commit and emitted the removed `intents` activation
  field. The clean result is 0.4 points above the matching committed Slice 0
  scenario (2.8/5 and 7.717 seconds mean), flat with Staging quality (3.2/5)
  while faster than its 11.064-second mean, and 0.6 points below the older
  pre-Slice-0 good WIP scenario (3.8/5 and 5.892 seconds mean). The current
  48-turn Slice 3 comparison is recorded below.

- Historical Slice 3 full offline unit suite: 861 passed, 2 strict expected
  failures, and one pre-existing `StarletteDeprecationWarning` in 7.90 seconds.
  At that point the expected failures tracked constraint provenance and
  not-yet-built historical-reference resolution.
- Focused Slice 3 cart transaction suite: 78 passed with the same pre-existing
  warning. It covers caller-stable request IDs, adapter propagation, exact
  add/remove/update replay, conflicting-key rejection, stable cart-line removal,
  rollback of every mutation path, and legacy quantity-ledger migration.
- Directly affected legacy cart-adapter compatibility coverage: 12 passed,
  including opaque-line resolution before remove and bulk add/remove dispatch.
- Focused checkpoint coverage: 8 passed, including default and explicit memory
  mode, fail-fast rejection of every other store, and failed-thread cleanup.
- Focused Slice 0 policy and activation coverage in the full suite verifies
  exact registry/frontmatter/policy agreement, forced activation-only binding,
  complete selected-file injection, model visibility for each selected grant
  union, direct-dispatch allow/deny behavior, current-request isolation,
  same-batch rejection, and preservation of non-shopping `read_file` access.
- The tests-only Slice 1 boundary originally froze five cases. Browse-only cart
  rejection passes through Slice 0, and add/remove replay now pass through the
  Slice 3 transaction boundary. The minimal Slice 5 resolver now covers exact
  historical 0/1/many matching and clarification; invented filter provenance
  remains separate work.
- Agent observability coverage verifies current-turn skill activation, model-issued
  tool-call order and arguments, rejection/duplicate classification, pending
  calls, additive API/SSE propagation, snapshot-before-delete failure handling,
  and product evidence scope, provenance, size bounds, and truncation.
- Focused evaluator coverage: 62 collected/passed in the full suite, including strict evidence allowlisting,
  aggregate-size rejection, legacy defaults, and turn-scoped Judge propagation.
- Focused shopper-skill registry coverage validates all five `role` and
  `tools_granted` declarations.
- Focused runtime, commerce-adapter, and memory-service modules: 165
  collected/passed in the full suite.
- Current capability API response: 147,792 bytes; compact LLM projection:
  5,769 bytes, down from 12,962 bytes before this change while retaining
  taxonomy keys, every current enum value, and semantic/filter roles.
- UI production build: compiled successfully.
- UI lint: 0 errors and 3 pre-existing warnings.
- Docker Compose configuration: valid.
- Fresh no-cache chain-server and memory-retriever Docker images: built
  successfully. The chain image contains MemorySaver and no Redis/Valkey
  checkpoint package; the memory image publishes the keyed quantity-update
  contract.
- Direct local smoke coverage passed for opaque non-reused cart-line IDs,
  exact idempotency replay, conflicting-key rejection, delete/add/retry safety,
  disabled policy content, persona omission, service health, and Compose
  configuration.
- The current Slice 3 fixed shopping cohort completed all 48 Judge decisions at
  3.4167/5, exactly matching the preserved Slice 0 WIP average. Score-4-or-better
  coverage increased from 29/48 to 33/48. Distribution changed from
  `2: 12, 3: 7, 4: 26, 5: 3` to `1: 4, 2: 7, 3: 4, 4: 31, 5: 2`.
- Timing is qualified: 46/48 requests produced timing records, with mean / median /
  p95 / maximum of 13.810s / 12.898s / 25.037s / 30.761s versus
  4.513s / 3.704s / 9.135s / 17.257s in the preserved Slice 0 WIP. Two target
  requests hit the explicit 60-second timeout. Per-turn memory stayed near
  0.007s and the added cart transaction path was not exercised by this
  read-only fixed suite; the observed latency and timeout regression is in the
  hosted Deep Agents path and cannot be attributed to the Slice 3 cart boundary.
  The run is preserved as `current_wip`; it was not patched or rerun.
- The clean Slice 0 working tree was evaluated as
  `slice0_clean_candidate_20260720` and promoted to the canonical local
  `current_wip` archive. It completed all 48 shopper turns with 48 Judge results
  and 48 timing records. Judge average was 3.4167/5; distribution was 2: 12,
  3: 7, 4: 26, and 5: 3. Twenty-nine of 48 turns (60.42%) scored at least 4.
- Mean / median / p95 / maximum latency was 4.513s / 3.704s / 9.135s /
  17.257s. All 48 turns included agent diagnostics and terminated `completed`;
  no request/graph error, generic fallback, partial graph-message loss,
  ungranted-tool dispatch, or Judge failure was observed.
- The run recorded 26 bounded catalog rejections: 16 invalid requests, six
  unsupported constraints, three constraint reviews, and one stop-tool-use
  outcome. One valid catalog search still completed on every turn.
- Against the preserved immediately preceding good WIP, Judge average moved
  from 3.4375 to 3.4167 (-1 Judge point across 48 turns) and
  score-4-or-better coverage moved from 30/48 to 29/48. Mean and median latency
  improved from 4.718s / 4.257s to 4.513s / 3.704s; p95 was effectively flat
  at 9.085s / 9.135s, while maximum latency moved from 11.978s to 17.257s.
- Against `previous_committed` / Staging, Judge average improved from 3.2083 to
  3.4167 and score-4-or-better coverage rose from 23/48 to 29/48. Mean / median /
  p95 / maximum latency improved from 14.922s / 8.863s / 57.367s / 94.609s to
  4.513s / 3.704s / 9.135s / 17.257s.
- A targeted live two-turn flow activated only `product-discovery` for browse,
  then only `cart-management` for an explicit add using the prior
  `PRODUCT_REF`; the add completed successfully. The fixed suite contains no
  cart mutation, so this flow and the offline allow/deny tests provide the cart
  binding evidence.
- Two earlier Slice 0 cohorts are excluded from quality comparison and retained
  only as local diagnostics: one hosted completion emitted an anomalous 32,781
  output tokens with no graph message, and one hosted request never returned
  before the 120-second client timeout. Replaying the affected turn completed
  normally; the clean cohort followed a service restart and separated target
  collection from judging.

The Staging comparison remains qualified because the catalog and broader dirty
WIP treatment changed; the adjacent good-WIP comparison is the closest Slice 0
signal. Catalog-sensitive Golden drift
still applies: existing beige skirts and pastel tops contradict older absence
answers, so raw scores are not a standalone catalog-quality verdict.

The preserved Slice 0 full run covers the deterministic search-only renderer, noun-only
product-type provenance, one-repair search boundary, finite preservation of
independently valid repair fields, review-blocker fixes, and turn-scoped Judge
evidence. Search-only answers stayed inside names, prices, roles, and confirmed
filter evidence. The remaining score gap is therefore not a reason to weaken
the evidence boundary.

## Remaining Quality Risk

The store-policy YAML contains explicit operator placeholders and defaults to
`configured: false`; policy lookup fails closed until an operator replaces
them and explicitly enables the file. Durable shopper/assistant turns now live
in the single-replica memory-service SQLite database on the Compose
`memory-data` volume. Transcript retention/TTL and a shared multi-writer memory
store remain open production decisions. Finalized product-card output now
creates durable presented-product events and a compact reference index. Active
anchors and preferences remain reserved and unused.

Deep Agents graph checkpoints are request-scoped and process-local. They use a
collision-safe pair of conversation ID and request ID, disappear on chain-server
restart, and are not shared across replicas. Successful durable finalization deletes them; a
finalize failure deliberately preserves the incomplete request checkpoint. The
checkpoint is no longer shopper memory, so the production scale decision is a
shared multi-writer replacement for the single SQLite memory service rather
than a shared graph-history store. Repository-wide allowlist compliance remains
a separate audit because the pre-existing dependency set contains additional
license notices unrelated to checkpointing.
Cart reads now expose an opaque, non-reusable `cart_line_id` as
`CART_LINE_ID`, including an idempotent migration for existing databases.
Adds persist the catalog `product_id`; remove and absolute-quantity update use
the cart-line ID. All three mutation paths share one owner-scoped replay ledger
and commit the cart change plus replay record atomically. Identical retries
replay, while conflicting key reuse fails without mutation. Mutation records
and their stored responses currently persist for the SQLite database lifetime;
retention and cleanup policy remain a follow-up. Variant-level cart identity is
also deferred.
Product availability is currently a deterministic application stub, not live
inventory. For a known conversation product ref it reports general
availability, echoes a requested size for apparel and footwear, and treats
other product categories as one-size. Unknown or expired refs require a fresh
catalog search. Live stock counts and variant inventory remain future work.

`PRODUCT_REF` authorization exists only in request-local evidence. Current-turn
search adds it directly; one unique durable same-conversation resolution can
restore an earlier presented product after restart or on another worker.
Missing or ambiguous references never authorize a downstream tool. Matching is
exact, catalog revision is not yet enforced, and catalog replacement can still
require a fresh search.
Persona support remains intentionally unavailable until a trusted profile
source and typed validation contract are defined.

Judge product evidence is capped at 24 records and 32,000 serialized characters
per turn. `product_evidence_truncated` makes omissions explicit; a truncated
turn may still require a narrower rerun when the omitted fact is material to the
score.

Mandatory activation adds one bounded app-model step to every Deep Agents turn.
It guarantees that shopping cannot run before a complete skill file is loaded,
then limits model visibility and dispatch to the selected `tools_granted` union.
The semantic choice among registered skills remains model-selected and is
explicit in `agent_diagnostics.skill_files_read`. Slice 0 does not yet prove
explicit current-turn mutation intent: selecting `cart-management` grants its
mutators, while server-owned intent authorization remains a later slice.

The corrected contract guarantees validation and enforcement of the scope and
must-haves the agent supplies. It cannot prove that a language model copied
every strict phrase from shopper language into `required_constraints`. For
example, if the agent omitted “only cotton,” deterministic validation would
have no omitted value to reject. Guaranteeing that separately would require
another interpretation/review model call or fixed natural-language rules; both
were intentionally excluded to avoid added latency and hard-coded language
logic. The prompt and required tool fields are the chosen minimal boundary.

`product-discovery` still contains transport-oriented catalog instructions.
That separate cleanup is intentionally deferred so this slice can isolate the
effect of simplifying `outfit-styling`.

The runtime rejects duplicate taxonomy values, partially incompatible
category/subcategory sets, and taxonomy fields that are not scalar enum hard
filters.

In the preserved Slice 0 run, nine turns scored 2. Most are safe no-match or
unenforceable answers where older Goldens prefer speculative adjacent products.
One observed gap motivated Slice 5: after a denim-skirt request correctly failed
closed, the next turn's “that skirt” resolved to an older maxi-skirt candidate.
The new durable resolver deterministically returns zero, one, or many exact
matches and authorizes only one, so missing or ambiguous references now require
clarification. Focused tests cover that boundary; a full live/Judge rerun has
not yet been performed.

The raw Judge score should not be described as an unqualified catalog-quality
gain until catalog-dependent Goldens are reconciled with the active inventory.
