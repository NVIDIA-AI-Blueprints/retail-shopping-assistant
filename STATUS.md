# Project Status

Updated: 2026-07-28

## Current Milestone

The current working tree extends the shopper-serving Deep Agent architecture:

- local app-code mode now keeps React browser API requests under a scoped
  `/local-api` prefix on port `3000` and forwards them through the development
  proxy to the chain server without Create React App's package-proxy Host
  restriction. Remote browsers therefore load representative shoppers and chat
  through the single forwarded UI port instead of failing into Guest-only mode
  or `Invalid Host header` when port `8009` is not separately exposed.
  Development responses use `Cache-Control: no-store` so tunneled browsers do
  not retain an obsolete direct-port bundle. The
  picker makes one delayed automatic retry after an unavailable profile load,
  then retries on browser-online or tab-focus events without polling for the
  app lifetime or interrupting an active Guest conversation;
- the chain server now has a dormant, provider-neutral weather forecast
  contract with a Visual Crossing Timeline adapter and a directly constructible
  `get_weather_forecast_tool` wrapper. It accepts only an exact five-digit US
  ZIP plus local today, one ISO date, or a complete inclusive ISO range of at
  most 15 days; returns bounded normalized daily evidence with attribution;
  rejects non-live provider sources; and maps transport/provider failures into
  sanitized typed outcomes. `WEATHER_ENABLED=false` is the default, the key
  remains an indirect `WEATHER_API_KEY` environment reference scoped only to
  the chain server, and startup/health paths perform no provider request. The
  wrapper is deliberately absent from the runtime registry, tool policy, skill
  grants, prompts, request context, FastAPI, UI, and shopper-profile behavior;
- the memory-service SQLite database now owns an immutable
  `shopper_profiles` registry bootstrapped from five reviewed rows whose
  `shopper_type` and `behavior` values map 1:1 to the committed live-evaluation
  profiles. Memory and chain-server read endpoints expose only ID, display
  name, type, behavior, and five-digit ZIP. A new UI session gates chat on an
  explicit dropdown choice of Guest mode or one of those five shoppers.
  A named selection displays its type, behavior, and saved ZIP in a compact
  navigation strip; Guest mode shows no profile strip.
  Switching choices clears visible chat/product/media/metric state and rotates
  the tab-scoped session, conversation, and cart identities; Reset retains the
  selected shopper mode. The UI sends only a named shopper's selected ID with
  each turn, while explicit Guest omits it. Memory atomically resolves and
  binds a named selection to the conversation, stores the nullable foreign key,
  and returns exactly type, behavior, and ZIP. The runtime injects that snapshot
  once as bounded soft guidance, absent for Guest and never persisted inside
  transcript text;
- startup migration 7 removes three obsolete derived conversation-response
  cache columns from earlier local SQLite schemas. It preserves authoritative
  turns, events, projections, carts, and profiles while allowing current
  durable-turn inserts to proceed and restores the unique one-started-turn
  index;
- a single memory-service SQLite replica now starts each turn durably before
  guardrail/model/tool work, returns bounded model-context-eligible raw turns
  plus the authoritative cart, and finalizes every completed, blocked, or
  failed outcome. Blocked turns remain durable and exactly replayable but are
  excluded from both the service's recent-turn projection and the chain prompt
  formatter. An exact retry of a finalized request replays its stored response
  without another model turn. Finalized ordered product cards create durable
  `candidate_set_presented` events and a compact reference index. One exact
  typed batch resolver returns 0/1/many same-conversation matches; only a unique
  match becomes request-local evidence, while zero or many require
  clarification. MemorySaver is request-scoped with a collision-safe pair of
  conversation ID and request ID, deleted after successful durable
  finalization, and preserved on finalize failure. The compact reference index
  is capped at 16,384 characters and resolution is enforced at most once per
  turn;
- six shopper skills are registered for product discovery, outfit styling,
  event context, cart management, budget shopping, and controlled store-policy
  answers. A
  required first model step selects the smallest applicable set each turn; the
  runtime then injects the complete selected files. Their frontmatter declares
  `role`, optional `exclusive_group`, and `tools_granted`; only the selected
  grant union becomes model-visible, and dispatch rechecks it against an
  independent immutable policy before invoking a shopping handler. Registry
  tools, frontmatter grant pairs, and policy pairs must match exactly at startup.
  The selected names are persisted with each durable terminal output. The
  immediately preceding turn supplies them as a read-only continuity signal
  only when that turn is eligible for model context; otherwise there is no
  hint. They neither force routing nor unlock tools.
  Terse item-only follow-ups inside an active outfit-building or style-led
  single-piece thread still select `outfit-styling` from conversation context.
  Event context is a zero-tool modifier available only beside outfit styling.
  It is selected whenever an event destination or venue is stated, or when the
  response would otherwise ask about or branch on missing destination or venue
  context.
  Explicit current or recent event destination and venue setting take
  precedence; saved ZIP is only a tentative event-location candidate to
  confirm as the shopper's usual area or elsewhere, never shopping, shipping,
  availability, or current location. Plan-first with missing context receives
  exactly two short sentences: one conditional direction, then one
  destination-or-venue question. With context complete, it uses one short
  paragraph and asks no further event-context question. Shop-now begins with
  one grounded requested or core role and, if location is still missing and
  materially changes the next recommendation, asks only event location
  alongside results; venue is deferred. The helper never assumes that a
  destination establishes a venue and performs no weather lookup or inference.
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
  fail-closed fallback when synthesis or editing cannot produce an answer.
  No-tool event-context turns also use a compact final editor under the shared
  deadline. That editor receives only saved-ZIP-candidate presence, never ZIP
  digits. On search-bearing event-context turns, the final text must retain at
  least one exact returned candidate; deterministic candidate rendering
  restores any missing candidates.
  Grounding now requires an explicit gap when the requested outcome depends on
  a functional product property absent from evidence, and deterministic
  fallback carries the same generic unverified-property disclosure. Selection and
  response metadata are regenerated from current files rather than retained in
  the request checkpoint;
- the runtime has an eleven-tool shopper registry plus one internal skill
  activation control tool. A turn receives only the tools granted by its
  selected skills. Invalid skill composition receives one reasoned correction;
  a repeated invalid selection returns a fixed clarification without running a
  shopping tool or exhausting the graph recursion budget. Multiple activation
  calls in one response execute none and clarify immediately. The shopping tools
  cover cart quantity update, controlled
  policy lookup, category-aware no-I/O availability, a no-I/O active-promotions
  signal, and deterministic durable same-conversation product resolution;
- memory-service schema migrations, turn start/finalize/replay, bounded
  recent-turn reads, and cart snapshots use transactional SQLite operations;
  stale active turns are recovered during startup and atomically at the next
  start. Only the latest abandoned sequence can reopen: it retains its request
  ID for cart idempotency but rotates a service-issued attempt token. A stale
  finalize is rejected and converted to a safe response without stale products;
  other finalize outages preserve the grounded response and add
  `memory_finalize_error` while retaining the request checkpoint. Successful
  finalization atomically records presented products and then deletes the
  checkpoint. Cancelled runtime turns are finalized before cancellation
  propagates. Deep Agents graph execution and the grounding editor now share a
  configurable 45-second default deadline. A graph timeout captures bounded
  partial state and finalizes as failed with `agent_timeout`; a grounding timeout
  finalizes as failed with `grounding_timeout`, uses deterministic catalog
  rendering for search-only evidence, and otherwise returns a fixed
  retry/cart-check response instead of the unverified draft. The checkpoint is
  released only after durable finalization. Database sessions
  remain request-scoped and are always returned to the SQLAlchemy pool after
  successful and failed API requests;
- dependency resolution retains `deepagents==0.6.12`, `langchain==1.3.11`,
  `langgraph==1.2.7`, and `langgraph-sdk==0.4.2`. The services that resolve
  `orjson` pin `3.11.5`, the last release limited to the Apache-2.0/MIT policy;
  Redis checkpoint packages remain absent;
- representative-shopper context cannot create a budget, product requirement,
  size, color, material, cart intent, product reference, skill selection, tool
  authorization, product fact, weather fact, or current-location claim.
  Explicit current-turn instructions take precedence, followed by explicit
  recent-conversation preferences; unknown caller fields remain ignored for
  backward compatibility, and caller-supplied persona objects are never
  injected; and
- both response paths retain additive agent diagnostics for activated skill
  files, ordered tool calls and arguments, rejected/duplicate calls, bounded
  current-turn product evidence from successful catalog search/detail results,
  bounded zero-result catalog scope outcomes, and final termination. Public
  query responses return an empty diagnostics object by default;
  `EXPOSE_AGENT_DIAGNOSTICS=true` is reserved for trusted operator/evaluation
  deployments. Standard Compose binds the unauthenticated memory-service host
  port to loopback while containers continue to use the private Compose
  network.
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
- every agent catalog call uses one flat executable search schema carrying
  `semantic_query`, product-agnostic `shopper_guidance`, nullable
  `requested_product_type`, capability-derived `taxonomy` and
  `required_constraints`, `scope_complete`, and optional `search_mode`. It has
  no model-authored taxonomy relationship or catalog-absence field. A
  shopper-named type that is not separately advertised may use one
  model-selected faithful advertised parent category; the type remains semantic
  direction and structured response evidence preserves the broader-scope
  caveat. If neither a direct type nor one faithful parent can be selected, the
  assistant asks one concise clarification directly without a tool call or absence claim. The
  search schema is generated from cached capabilities with exact taxonomy
  values, hard-filter properties and enum values, typed numeric ranges, and
  search-mode values. It omits cross-field validators, while the handler applies
  the existing strict semantic search model. Invalid individual values fail at
  the typed tool boundary and cross-field failures reach capability-aware
  validation.
  For text searches,
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
  call accepts at most one category. A genuinely open role selects exactly one
  advertised subcategory;
  alternatives, confirmations, comparisons, and follow-ups count as named
  types. For an open role, the model names the selected subcategory in
  `requested_product_type`. Invalid open-role provenance is rejected rather
  than silently reinterpreted. Capability-owned exact category/subcategory
  relationships validate the selection. For a malformed open-role call,
  deterministic validation stops
  before retrieval and reports the exact eligible subcategories from the
  current capability contract. The model operating under the active skill still
  chooses the semantic role; validation does not choose it for the model.
  Executable text search requires at least one taxonomy value; image-only
  requests use empty arrays;
- the optional search-mode enum is generated from the catalog's advertised
  retrieval modes, and an explicit unknown or unsupported mode stops before
  retrieval instead of becoming an automatic default;
- the chain maps those generic taxonomy roles to the catalog-advertised field
  names, infers owning categories for a valid subcategory-only selection,
  rejects incompatible category/subcategory selections, and applies the result
  as deterministic hard filters;
- a shopper-named type that is not separately advertised may use one
  model-selected faithful advertised parent category. The category is the only
  taxonomy filter, the original type remains semantic direction, and returned
  products retain their actual categories as closest alternatives. If neither a
  direct type nor one faithful parent can be selected, the assistant asks one
  concise clarification without retrieval or an absence claim. An unsupported
  modifier does not erase an advertised type, while subjective style remains
  semantic direction. A product type never belongs in
  `unadvertised_requirements`;
- duplicate search identity is normalized taxonomy plus hard constraints, not
  semantic wording. Repeating that identity is stopped even when the query is
  paraphrased. A shopper-named product scope also executes at most once per
  turn, so an adjacent taxonomy cannot replace a successful first search;
- repair accounting uses the full normalized `requested_product_type` phrase.
  Alternative, comparison, ordering, and negation semantics remain model-owned;
  deterministic repair does not reconstruct them from shopper prose or equate
  connector and ordering changes. Each scope receives one total repair. A
  schema correction or a fresh
  constraint-provenance review can consume that shared budget; constraint
  feedback returned by an in-flight schema repair closes the loop for synthesis
  rather than opening another repair. The repair is an isolated model phase:
  it receives the capability-derived typed `search_catalog_tool`, compact
  server-generated Catalog capabilities, the current shopper message, bounded
  sanitized validator feedback, and the complete active shopper-skill
  instructions. Only `search_catalog_tool` is available and parallel calls are
  disabled. The repair may submit one corrected search or return no tool call to
  signal that clarification is needed. A no-tool response is only
  branch/control state: the server marks it, discards the model prose, and emits
  the fixed clarification
  `Could you clarify the product type or requirement you want me to use?`.
  Active-skill responses containing more than one shopping tool call are
  rejected before execution.
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
  arguments. The repaired call must preserve it exactly; the strict handler
  rejects drift rather than overwriting the model's call. Repair middleware
  never restores or rewrites taxonomy, constraints, requested type, or search
  mode. It may restore only the independently valid structural
  `scope_complete` flag, reported by name in bounded `restored_fields`
  diagnostics.
  Open-role validation failures include the matching provenance rule in that
  same feedback, so one repair can correct both transport and scope errors. A
  shopper-named role retains the shopper's noun or umbrella; a genuinely open
  role selects and names one advertised subtype. The same isolated repair may
  return no tool call to signal that clarification is needed; its prose is
  replaced by the fixed server-authored clarification above.
  Native failures confined to `required_constraints` receive sanitized field
  feedback plus the typed tool and compact Catalog capabilities; free-form
  scope, query, and guidance remain excluded while shopper-grounded scope is
  compared privately. A changed grounded scope closes before execution as
  `repair_scope_changed`.
  A native schema-invalid call with a nonempty `unadvertised_requirements` lane
  closes without repair. Schema-valid,
  genuinely open requests retain the bounded review for
  proposed inferred requirements.
  Every unadvertised requirement on a shopper-stated product scope fails closed
  before retrieval, including a synonym rather than the shopper's exact
  wording. The bounded constraint review is reserved for a proposed inferred
  requirement on a genuinely open role when its shared
  repair budget remains. That review freezes requested type, taxonomy,
  completion state, `search_mode`, and all advertised hard
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
  when another scope has an unsupported requirement. A fixed unsupported-
  requirement response is used only when that rejection is the sole current-
  turn business-tool outcome.
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

The newest focused gate is recorded first. Older implementation gates remain
below as comparison points; generated quality and timing
artifacts stay in the required local archive rather than versioned source.

- Styling-weather guidance Slice 2 event-context gate (2026-07-28): the final
  offline gate passed 58 tests with 1 expected xfail; changed Python passed
  Ruff, all three Golden YAML files parsed, and whitespace checks passed. The
  focused GPT-5.2 app/Judge gate completed 4/4 target turns and 4/4 judgments at
  5.0000/5, with every turn selecting `outfit-styling` plus the no-tool
  `event-context` helper. Mean / median / p95 / maximum target time was 10.215s
  / 9.012s / 14.755s / 15.698s. Guest made no saved-location assumption, the
  selected-profile opener asked whether the event was in the shopper's usual
  area or elsewhere without exposing ZIP digits, explicit Cancun-on-sand
  context overrode that candidate, and the shop-now turn returned grounded
  catalog dresses before its one location question. The gate used 13 app
  completions, 4 Judge calls, one catalog search, and one text-embedding call;
  no Challenger, full cohort, weather, cart, or policy call ran. Versus the
  immediately prior cleanly paired four-turn WIP, quality held at 5.00 while
  mean latency rose 0.192s and p95/max improved 0.493s/0.556s. The immutable
  report is
  `~/exec-briefs/retail-shopping-assistant/quality/baselines/2026-07-28__styling-weather-guidance__event-context-slice2-commit-ready/quality-report.md`;
  the Staging comparison remains explicitly incomparable because scenario scope
  and Judge treatment differ.
- Styling-weather guidance Slice 1 profile-selection gate (2026-07-27): all 10
  focused UI tests passed, UI lint reported 0 errors and 3 pre-existing
  warnings, the production build compiled, and whitespace checks passed. This
  slice changes only the browser UI, so no hosted app-model, Judge, weather
  provider, or full live evaluation was run. The prior completed integration
  run was preserved as a qualified/inherited regression reference, not as
  feature-attributable evidence. Its immutable archive is
  `~/exec-briefs/retail-shopping-assistant/quality/baselines/2026-07-27__styling-weather-guidance__profile-dropdown-slice1/`;
  the staging comparison is
  `~/exec-briefs/retail-shopping-assistant/quality/baselines/comparisons/staging__to__2026-07-27__styling-weather-guidance__profile-dropdown-slice1.md`.
- Dormant weather-tool Slice 3 gate (2026-07-27): the full offline backend
  suite passed 1,136 tests with 1 expected xfail, and the focused
  weather/configuration/Compose/local-runner suite passed 193 tests. Changed
  Python passed Ruff; shell syntax, Docker Compose configuration,
  retail-local-runner skill validation, whitespace checks, and the
  chain-server image build passed. The decisive GPT-5.2 app/Judge regression
  guard completed 48/48 shopper turns and judgments without collector,
  response, timeout, or Judge errors at 3.8958/5, with 40/48 turns scoring at
  least 4. Mean / median / p95 / maximum latency was 20.525s / 18.187s /
  36.372s / 38.179s. Against the immediately preceding Slice 2 WIP, average
  quality moved -0.1250, score-4-or-better coverage moved -2 turns, 3/36/9
  paired scores improved/tied/regressed, and mean / median / p95 / maximum
  latency changed by +0.934s / -0.234s / +0.531s / -1.331s. This is a
  qualified dormant-path guard rather than direct feature evidence: weather
  was explicitly disabled, `WEATHER_API_KEY` was absent, the tool was
  unregistered and absent from model context, and zero provider calls were
  made. No live provider smoke was run. The immutable report is
  `~/exec-briefs/retail-shopping-assistant/quality/baselines/2026-07-27__architecture_updates__slice3-dormant-weather-tool/quality-report.md`;
  canonical current/Golden, previous-committed/current, and
  previous-WIP/current comparisons are under
  `~/exec-briefs/retail-shopping-assistant/quality/shopping/comparisons/`.
- Representative-shopper Slice 2 context gate (2026-07-27): the full offline
  backend suite passed 1,016 tests with 1 expected xfail; all 9 UI tests passed,
  UI lint reported 0 errors and 3 pre-existing warnings, and the production
  build compiled. The existing SQLite lineage upgraded through migration 6;
  local service smokes confirmed exact selected/Guest context, restrictive
  profile binding, unknown-ID failure before agent work, and the public request
  schema. Four direct selected-profile GPT-5.2 turns confirmed Casey supplied
  no budget skill or price constraint, Jordan supplied no cart intent, Morgan
  supplied no ungrounded product facts, and Alex's saved ZIP supplied no
  weather or location claim. The final Guest-only GPT-5.2 app/Judge regression
  guard completed 48/48 shopper turns and judgments with no collector errors at
  4.0208/5 and 42/48 turns scoring at least 4. Mean / median / p95 / maximum
  latency was 19.591s / 18.421s / 35.841s / 39.509s. Against the immediately
  preceding Slice 1 WIP, average quality improved by 0.1250, score-4-or-better
  coverage increased by 4 turns, 9/33/6 individual scores
  improved/tied/regressed, and mean / median / p95 / maximum latency changed by
  -0.443s / +0.044s / -1.538s / -5.520s. Profile-specific instructions were
  deliberately removed from Guest prompts before this decisive run. The broad
  run is a Guest regression/timing guard; direct feature evidence is archived
  at
  `~/exec-briefs/retail-shopping-assistant/quality/shopping/selected-profile-smoke/slice2_selected_profiles_20260727T185800Z.md`,
  with canonical comparisons under
  `~/exec-briefs/retail-shopping-assistant/quality/shopping/comparisons/`.
- Representative-shopper Slice 1 gate (2026-07-27): the full offline backend
  suite passed 975 tests with 1 expected xfail; all 7 UI tests passed, UI lint
  reported 0 errors and 3 pre-existing warnings, and the production build
  compiled. Direct memory/chain health and profile-list smokes returned the
  exact five reviewed profiles. The full GPT-5.2 app/Judge regression guard
  completed 48/48 shopper turns and judgments with no collector errors at
  3.8958/5 and 38/48 turns scoring at least 4. Mean / median / p95 / maximum
  latency was 20.034s / 18.377s / 37.379s / 45.029s. Against the immediately
  preceding WIP, average quality improved by 0.1042, score-4-or-better coverage
  stayed 38/48, and mean / median / p95 latency improved by 2.807s / 2.286s /
  6.738s. This is a qualified complete chat-regression guard, not direct Slice
  1 feature attribution: the fixed collector neither calls `/shopper-profiles`
  nor sends a selected profile with a query. Canonical comparisons are stored
  under
  `~/exec-briefs/retail-shopping-assistant/quality/shopping/comparisons/`.
- Budget activation-correction gate (2026-07-24): 25 focused tests passed with
  1 existing expected xfail. The budget opener recovered from 1/5 and graph
  recursion failure to 4/5 and normal completion, while latency improved from
  21.99s to 9.40s. The full GPT-5.2 app/Judge run completed all 48 shopper turns
  and 48 judgments. Against the preceding WIP, Judge average moved
  3.8333→3.7917, score-4-or-better coverage moved 40/48→38/48, mean latency
  moved 21.36s→22.84s, and p95 moved 41.24s→44.12s. The target fix passed;
  aggregate quality and latency slightly regressed, with one unrelated bounded
  `grounding_timeout` fallback.
- Parent-category alternative gate (2026-07-24): focused catalog-scope, skill,
  request-validation, evidence, and deterministic-fallback tests pass for a
  shopper-named subtype such as sneakers searched once under a model-selected
  advertised parent such as footwear. The semantic query remains unchanged, no
  subtype hard filter is invented, and returned flats or sandals retain their
  actual categories with an explicit closest-alternative caveat. No catalog
  service or catalog data changed.
- Evidence-honesty gate (2026-07-24): focused search-only grounding and
  deterministic-fallback coverage requires unconfirmed functional properties
  to remain explicit and prevents candidates from being presented as proven
  suitable for the requested outcome. No catalog request or ranking behavior
  changed.
- Three-review-slice gate (2026-07-24): the durable prior-skill hint,
  empty-editor fail-closed response, and diagnostics/network-default hardening
  passed an 18-test focused pre-commit gate. Changed-file Ruff,
  `git diff --check`, and Docker Compose configuration validation passed.
  The matching 48-turn GPT-5.2 app/Judge run completed all shopper and Judge
  calls with no collector or Judge errors. Judge average was 3.8958/5, with
  37/48 turns scoring at least 4. Mean / median / p95 / maximum latency was
  19.069s / 17.239s / 36.140s / 45.025s. Forty-six turns terminated
  `completed`; two used the bounded `grounding_timeout` path. Against the
  immediately preceding WIP, average quality improved by 0.1042, score-4-or-
  better coverage increased by one turn, and mean / median / p95 latency
  improved by 0.500s / 1.160s / 1.390s. The comparison is qualified because
  both runs were dirty working trees and the client timeout changed from 120
  to 300 seconds.
- Grounding deadline gate (2026-07-24): 7 focused offline tests passed. The
  grounding editor uses async invocation under the remaining model-stage
  deadline. Timeout cancels the editor, finalizes once as failed with
  `grounding_timeout`, and never returns an unverified mutation draft; ordinary
  editor failure or empty output also fails closed, while search-only
  deterministic fallback, explicit editor disablement, and graph
  `agent_timeout` behavior remain intact.
- Blocked-context isolation gate (2026-07-23): 4 focused offline tests passed.
  A blocked turn remains durably stored and exactly replayable, while both the
  memory-service recent-turn projection and the chain prompt formatter exclude
  its shopper and assistant text from the next request.
- Model-owned catalog semantics gate (2026-07-23): 4 focused offline tests
  passed. Coverage confirms that negated shopper language does not override the
  model-owned typed scope, a typed multi-subcategory selection receives
  candidate coverage without prose parsing, and generic repair retains its
  fixed clarification boundary.
- Superseded advertised-alternative parser gate (2026-07-23): the earlier
  deterministic tests asserted raw shopper-text pair detection and
  connector/order equivalence. Slice 1 removed that behavior because semantic
  alternatives and negation belong to the model-owned typed request. Its
  five-turn `conv_3.yaml` replay scored 4.6/5, but the “Heels or flats for this
  look?” turn answered directly without catalog search and therefore did not
  exercise the removed parser.
- Targeted search-boundary gate (2026-07-23): the focused offline suite passed
  210 tests with 1 intentional xfail. Coverage includes the fixed server-owned
  clarification boundary and preservation of grounded products when another
  requested search scope needs clarification. Successful cart and other
  shopping-tool evidence also remains in the existing grounded response path.
  Five context-complete live turns scored 3.6/5. The two previously failing
  targets scored 4/5 for “Do you have
  formal tops?” and 3/5 for “What bottoms go well with that?” Both activated
  `outfit-styling`, completed one valid catalog search, and avoided rejected
  calls, timeouts, and generic fallback. No paired turn score regressed against
  the matching Staging subset. Mean / p95 latency was 18.743s / 24.204s versus
  Staging's 7.520s / 12.898s. This is a qualified targeted gate, not a
  replacement for the canonical 48-turn evaluation.
- Historical superseded search-or-clarify gate: 185 focused tests passed with the one
  existing intentional xfail; targeted Ruff and `git diff --check` passed. One
  live replay of the first two `conv_2` turns used the GPT app and Judge with
  guardrails disabled. The previously poor “Do you have formal tops?” turn
  improved from the archived generic fallback at 1/5 to a grounded 4/5 response
  in 20.074 seconds. `outfit-styling` activated, the first invalid relation was
  rejected, the one bounded repair selected
  `apparel → blouses, camisoles, sweaters` with
  `member_of_requested_umbrella`, and one catalog search returned four grounded
  products. No retry or broader live cohort was run. The targeted artifact is
  stored outside the repository under the required local quality archive.
- Previous broad GPT baseline: the focused affected suite completed with 183
  passed and 1 xfailed, followed by the full offline suite with 929 passed and 1
  xfailed. Ruff, `git diff --check`, and Docker
  Compose configuration validation are green. Eight targeted live GPT-backed
  flows completed 8/8. The full GPT-qualified shopping cohort completed all 48
  shopper turns, Judge decisions, and timing records with no request errors or
  timeouts and two generic fallbacks. Judge average was 3.5208/5, with 32/48
  turns scoring at least 4; mean / median / p95 / maximum latency was 19.457s /
  19.185s / 35.665s / 47.774s. Comparison with earlier Ultra-backed app/Judge
  runs is qualified because both the app model and Judge model changed to GPT.
  Its two genuine generic fallbacks were formal tops and the beige-top bottoms
  follow-up; the current targeted gate above verifies both after the
  model-visible search-boundary simplification. Catalog-sensitive Golden drift
  still applies: older expected absence or category breadth can disagree with
  the active published catalog.
- Historical no-direct request-lane gate (superseded): 4 passed with one pre-existing
  `StarletteDeprecationWarning`. The live “What casual sneakers do you have?”
  smoke activated `product-discovery`, completed after one bounded no-direct
  guidance correction, recorded `no_direct_catalog_match`, returned no product
  evidence, and made no catalog retrieval. Total time was 13.982 seconds. No
  Judge call was run; the unjudged smoke and comparison to the two archived 2/5
  turns are stored under the required local quality archive.
- Focused promotions gate: 11 passed with one pre-existing
  `StarletteDeprecationWarning`. A fresh-identity live smoke for “Any sales on
  shoes?” activated only `product-discovery`, called
  `check_active_promotions_tool`, made no catalog search, and completed in
  13.054 seconds with the no-active-promotion response. No Judge call or broad
  evaluation was run. The smoke and its qualified comparison to the prior
  45.098-second sale timeout are stored under the required local quality
  archive.
- Focused execution-deadline gate: 78 passed with one pre-existing
  `StarletteDeprecationWarning`. Coverage includes positive/default/environment
  configuration, rejection of non-positive values, graph cancellation before
  partial-state capture, exactly-once failed durable finalization with the
  current attempt token, empty unsent product replay, checkpoint deletion only
  after finalization, unchanged external cancellation, unchanged finalize-
  failure preservation, and the neighboring graph-failure path. Ruff and
  Docker Compose configuration validation pass. A targeted five-turn
  `conv_4.yaml` rerun scored 2.8/5 with mean / median / maximum latency of
  16.179s / 10.220s / 45.098s. Its one hosted post-search stall terminated as
  `agent_timeout`; the following turn completed normally, with no
  `conversation_turn_in_progress` cascade. This targeted result validates
  containment only and does not replace the canonical 48-turn WIP baseline.
- Focused Slice 5 durable-product gate: 77 passed and 1 existing expected
  failure. Coverage includes server-derived `candidate_set_presented` events,
  compact projection rebuilds, exact 0/1/many typed resolution, strict
  clarification, skill grants and immutable policy, request-local evidence, and
  restart-safe memory-service lookup. The 14 focused runtime lifecycle tests
  also pass for collision-safe conversation/request checkpoint isolation,
  deletion only after successful durable finalization, and preservation on
  finalize failure.
  The committed Slice 5 state at `5d60623` subsequently completed the fixed
  48-turn Judge cohort at 3.0625/5, with 27/48 turns scoring at least 4. One
  hosted graph execution continued for 213 seconds after the evaluator's
  60-second client timeout and caused four immediate follow-ups to return
  `conversation_turn_in_progress`. That single cascade accounts for 14 of the
  17 Judge points lost against the prior WIP; the targeted execution-deadline
  rerun above confirms that this failure mode is bounded to one turn.
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
- Focused shopper-skill registry coverage validates all six `role` and
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
product-type provenance, one-repair search boundary, review-blocker fixes, and turn-scoped Judge
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
Selected skill names cross that boundary explicitly in typed durable turn
output and are returned only as a non-authorizing hint for the next turn's
fresh activation decision. Active anchors and effective preferences remain
unimplemented.
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
Promotion status is also a deterministic application stub. It currently reports
that no active sale or promotion is configured through the assistant; catalog
results and prices are not treated as markdown evidence. A live promotions
service remains future work.

The promotions smoke used an isolated fresh SQLite database because a local
pre-Slice database lacked the required `conversation_turns.attempt_id` column
and failed during startup recovery. That local database was not deleted or
modified. Upgrade compatibility for that legacy artifact requires a separate
migration-order audit.

`PRODUCT_REF` authorization exists only in request-local evidence. Current-turn
search adds it directly; one unique durable same-conversation resolution can
restore an earlier presented product after restart or on another worker.
Missing or ambiguous references never authorize a downstream tool. Matching is
exact, catalog revision is not yet enforced, and catalog replacement can still
require a fresh search.
The fixed representative-shopper registry is a trusted, typed source and its
selected row is now bound at durable turn start. It remains static soft
guidance, not learned preference state. Caller-supplied or mutable customer
personas remain unavailable until their ownership and input-safety contracts
are defined.

The weather tool is not yet a shopper capability. The no-tool event-context
helper now defines saved-ZIP versus explicit event-location and venue
precedence, but it performs no forecast lookup or inference. A later leveraging
slice must resolve relative dates, register and grant the weather tool only to
that helper, preserve its output as grounded current-turn evidence, display
provider attribution and forecast uncertainty, and prevent weather from
silently establishing waterproofing, warmth, safety, or any other catalog
constraint. Visual Crossing plan terms must be reviewed before storing or
displaying forecast data; the dormant boundary persists and displays none.

Judge product evidence is capped at 24 records and 32,000 serialized characters
per turn. `product_evidence_truncated` makes omissions explicit; a truncated
turn may still require a narrower rerun when the omitted fact is material to the
score.

Mandatory activation normally adds one bounded app-model step to every Deep
Agents turn; an invalid composition may add one corrective step. It guarantees
that shopping cannot run before a complete skill file is loaded, then limits
model visibility and dispatch to the selected `tools_granted` union. The
semantic choice among registered skills remains model-selected and is explicit
in `agent_diagnostics.skill_files_read`. Slice 0 does not yet prove explicit
current-turn mutation intent: selecting `cart-management` grants its mutators,
while server-owned intent authorization remains a later slice.

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
clarification. The full Slice 5 run confirmed the durable lookup path but also
showed that shopper-visible presentation order and candidate-set styling
references need their own later slice; they are not part of the execution-
deadline change.

The raw Judge score should not be described as an unqualified catalog-quality
gain until catalog-dependent Goldens are reconciled with the active inventory.
