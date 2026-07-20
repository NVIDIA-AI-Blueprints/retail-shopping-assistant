# Project Status

Updated: 2026-07-20

## Current Milestone

The current working tree extends the shopper-serving Deep Agent architecture:

- conversation checkpointing uses process-local MemorySaver.
  `CHECKPOINT_STORE=memory` is the only supported value; checkpoints disappear
  on restart and are not shared across replicas. A compliant production shared
  backend remains an open decision. The conversation-scoped product-ref cache
  is separate process-local state;
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
  Each skill also declares product-agnostic `response_guidance` in
  frontmatter as a fallback. Each catalog call supplies required pre-retrieval
  `shopper_guidance` authored under the active skill. Completed successful
  search-only turns display its safe form before deterministic candidate facts,
  without a response-editor or final-synthesis model call. Selection and
  response metadata are regenerated from current files rather than retained in
  the conversation checkpoint;
- the runtime has a nine-tool shopper registry plus one internal skill
  activation control tool. A turn receives only the tools granted by its
  selected skills. The shopping tools cover cart quantity update, controlled
  policy lookup, and an honest availability stub;
- memory-service database sessions are request-scoped and always returned to
  the SQLAlchemy pool after successful and failed API requests;
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

- Full offline unit suite: 854 passed with one pre-existing
  `StarletteDeprecationWarning` in 6.11 seconds.
- Focused checkpoint coverage: 8 passed, including default and explicit memory
  mode, fail-fast rejection of every other store, and failed-thread cleanup.
- Focused Slice 0 policy and activation coverage in the full suite verifies
  exact registry/frontmatter/policy agreement, forced activation-only binding,
  complete selected-file injection, model visibility for each selected grant
  union, direct-dispatch allow/deny behavior, current-request isolation,
  same-batch rejection, and preservation of non-shopping `read_file` access.
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

The current full run covers the deterministic search-only renderer, noun-only
product-type provenance, one-repair search boundary, finite preservation of
independently valid repair fields, review-blocker fixes, and turn-scoped Judge
evidence. Search-only answers stayed inside names, prices, roles, and confirmed
filter evidence. The remaining score gap is therefore not a reason to weaken
the evidence boundary.

## Remaining Quality Risk

The store-policy YAML contains explicit operator placeholders and defaults to
`configured: false`; policy lookup fails closed until an operator replaces
them and explicitly enables the file. Deep Agents graph
checkpoints are process-local, disappear on restart, and are not shared across
replicas. Successful thread histories have no TTL or capacity bound and remain
in process heap until restart. Selecting an Apache-2.0/MIT-compatible production
shared backend remains open. Repository-wide allowlist compliance remains a
separate audit because the pre-existing dependency set contains additional
license notices unrelated to checkpointing.
Cart reads now expose an opaque, non-reusable `cart_line_id` as
`CART_LINE_ID`, including an idempotent migration for existing databases.
Quantity updates use one absolute-value `PUT`; their idempotency ledger and cart
mutation commit atomically, so identical retries replay and conflicting key
reuse fails without mutation. The legacy add/remove paths still store products
by display name and do not enforce their generated idempotency keys server-side.
Quantity-update idempotency rows and their stored responses currently persist
for the SQLite database lifetime; retention and cleanup policy remain a
follow-up.
Product availability remains `unknown` until a live inventory/variant service
is integrated.

Same-conversation `PRODUCT_REF` evidence is held in a bounded process-local
cache and is valid only for the active catalog snapshot. It is separate from
the process-local graph checkpoint. A restart, another replica, cache eviction,
or catalog replacement therefore requires a fresh search before details or cart
adds.
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

The runtime rejects duplicate taxonomy values, partially incompatible
category/subcategory sets, and taxonomy fields that are not scalar enum hard
filters.

Nine turns still scored 2. Most are safe no-match or unenforceable answers where
older Goldens prefer speculative adjacent products. One concrete conversation
gap remains: after a denim-skirt request correctly failed closed, the next turn's
“that skirt” resolved to an older maxi-skirt candidate instead of triggering a
clarification. Products and filter evidence stayed grounded, but ambiguous
long-conversation referent resolution remains a follow-up. It should be solved
with explicit conversation-state semantics, not taxonomy inference or weaker
catalog boundaries.

The raw Judge score should not be described as an unqualified catalog-quality
gain until catalog-dependent Goldens are reconciled with the active inventory.
