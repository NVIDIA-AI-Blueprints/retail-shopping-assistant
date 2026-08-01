# Deep Agents SDK Migration Plan

This plan captures the intended migration from the current bespoke LangGraph
shopping-agent orchestration to a simpler Deep Agents SDK harness.

Deep Agents is still built on LangGraph internally. The goal is not to remove
LangGraph as a transitive runtime. The goal is to stop maintaining a custom
planner/cart/retriever/chatter graph in application code and let the Deep
Agents SDK own the agent harness, tool selection, skills, context management,
and streaming surface.

Reference docs:
- Deep Agents overview: https://docs.langchain.com/oss/python/deepagents/overview
- Deep Agents customization: https://docs.langchain.com/oss/python/deepagents/customization

## Current Implementation Notes

The first implementation slice uses `chain_server/src/deepagents_runtime.py`
as the Deep Agents SDK adapter and routes `/query/stream` and `/query/timing`
through that adapter. The older bespoke LangGraph graph files still exist in
the repository for reference and tests, but they are no longer the chain-server
entrypoint.

Current constraints:

- The current model-facing catalog boundary is one flat executable search
  schema constructed by the pure
  `chain_server/src/catalog_tool_contract.py` module and composed into registered
  wrappers by `chain_server/src/deepagents_runtime.py`, with reusable model-
  visible rules in `chain_server/src/catalog_scope.py`. Its fields are
  `semantic_query`, `shopper_guidance`, `requested_product_type`, `taxonomy`,
  `required_constraints`, `scope_complete`, and optional `search_mode`. It has
  no model-authored taxonomy relationship, clarification branch, or
  catalog-absence result. A shopper-named type not separately advertised may
  use one model-selected faithful advertised parent category while preserving
  the type as semantic direction. If neither a direct type nor one faithful
  parent can be selected, the assistant asks one concise clarification directly
  without calling the tool.
- The Deep Agents adapter exposes eleven thin request-scoped shopping tools over
  deterministic catalog, conversation-product, cart, policy, availability, and
  promotions functions, plus one internal skill-activation control tool.
- Five shopper-facing skills are discovered under
  `chain_server/skills/shopper/`: product discovery, outfit styling, cart
  management, budget shopping, and store-policy answers.
- Activation instructions designate product discovery and outfit styling as
  alternative primary procedures and prohibit selecting both. Product work
  uses one of them; standalone cart and policy turns do not need a primary.
  Budget shopping may modify either procedure only when the shopper states a
  budget.
- Registration is not activation. Every turn begins with a forced structured
  selection from those registered names. The runtime deterministically loads
  and injects each selected `SKILL.md` in full before it exposes any shopping
  tool. Each terminal output durably stores the selected names; the immediately
  preceding turn supplies them as a read-only continuity signal only when that
  turn is eligible for model context. Otherwise there is no hint. The names do
  not force routing or satisfy the activation gate. Shopping calls issued before
  activation or in the same model batch are rejected at execution time.
- Caller-supplied persona data is not injected into model context. Persona
  support remains deferred until its trust and validation contract is defined.
- Optional `session_id`, `conversation_id`, and `cart_id` fields are accepted,
  and the bundled UI now sends browser-session identifiers on every turn.
  Legacy callers that send only `user_id` are still mapped to deterministic
  compatibility identifiers.
- A single memory-service SQLite replica starts an ordered durable turn before
  guardrail, model, or tool work. Its negotiated v2 response returns three
  independent continuity lanes plus the authoritative cart: a non-authoritative
  rolling semantic summary, exact bounded raw turns strictly newer than the
  summary watermark, and a compact index of products actually presented.
  Blocked turns remain durable and exactly replayable but are excluded from the
  summary source, raw-tail projection, and chain prompt formatter. Every new
  turn is finalized as completed, blocked, or failed. An exact finalized
  request replay skips model/tool execution and returns stored response output.
  Each start returns a service-issued attempt token that finalization must echo.
  A start failure prevents agent execution; a generic finalize failure
  preserves the grounded response and request checkpoint and adds an operator
  diagnostic.
- The summary is semantic guidance only and cannot prove exact wording,
  product identity or facts, cart state, policy, availability, tool evidence,
  or permission. Memory separately offers an exact bounded oldest raw prefix
  outside normal model context. A tools-disabled compactor may combine only
  that prefix with the prior summary; memory then validates its projection
  version and boundary and commits the advance atomically with turn
  finalization. Finalized ordered product cards create durable
  `candidate_set_presented` events and the compact per-conversation index. One
  typed batch resolver returns 0/1/many exact matches; only one match becomes
  request-local product evidence. The index keeps the newest complete candidate
  sets within 16,384 serialized characters, and runtime permits at most one
  batched resolver call per turn. The same validated projection supplies a
  stricter capability lane: an exact unconflicted `PRODUCT_REF` may authorize
  only its scalar detail read without calling the resolver. Natural, ordinal,
  shortened, and ambiguous references still use the resolver.
- MemorySaver is request-scoped under a collision-safe pair of conversation ID
  and request ID, deleted only after successful durable finalization, and preserved on finalize failure. It
  remains process-local but no longer supplies cross-turn shopper memory or
  product-ref authorization.
- Active anchors and effective preferences remain reserved projection lanes.
  This slice does not add preference/sentiment extraction, fuzzy or embedding
  matching, cross-conversation lookup, or stale-catalog-revision handling.
- The compatibility mapping is not the final production identity design. A
  high-scale website should move to server-created session, conversation, and
  cart identifiers before broad rollout.
- The runtime caps Deep Agents recursion and distinct catalog scopes per turn.
  Every text search carries one taxonomy-independent semantic query, one
  required pre-retrieval product-agnostic `shopper_guidance` sentence, and the
  shortest product noun or true umbrella in `requested_product_type`. Guidance
  is empty and requested type is `null` only for image-only search. Duplicate
  identity is normalized taxonomy plus hard constraints, so paraphrasing cannot
  repeat the same retrieval.
- Cached Catalog capabilities generate the search schema's exact taxonomy and
  hard-filter values. The model owns semantic selection; deterministic code
  validates and maps the submitted structure but does not interpret shopper
  language or compare `requested_product_type` with taxonomy. Each search covers
  at most one category. Category/subcategory ownership, exact capability values,
  text-versus-image shape, hard constraints, retrieval mode, duplicate hard
  scopes, and turn limits remain deterministic. A category-only text search
  emits neutral evidence containing the model-authored requested role and
  advertised category searched; it does not assert unavailability or prove a
  parent relationship. Returned products keep their actual catalog categories.
  If the model cannot select a faithful scope, it clarifies without retrieval,
  substitution, or an absence claim. Any nonempty
  `unadvertised_requirements` lane fails closed without deterministic
  explicit-versus-inferred classification.
- The turn has one structural catalog-repair opportunity total. The isolated
  repair receives only the
  capability-derived search tool, compact server-generated Catalog
  capabilities, the current shopper message, bounded sanitized validator
  feedback, and the active skill instructions. Tool choice remains automatic:
  the model may submit one corrected search or return no tool call to signal
  that clarification is needed. That no-tool response is only branch/control
  state: the server marks it, discards the model prose, and emits the fixed
  clarification
  `Could you clarify the product type or requirement you want me to use?`.
  Parallel calls remain disabled.
  Runtime does not derive a repair key from shopper wording, lock a semantic
  scope, or reject a corrected role because its noun changed. The model may
  correct `requested_product_type` and taxonomy; both are validated afresh and
  never rewritten by runtime. Independently valid finite structural
  fields—advertised `required_constraints`, `scope_complete`, and
  `search_mode`—may be preserved across repair. A successful partial repaired
  search may continue to later valid work, but no second repair is available in
  that turn. The configured successful-search cap remains three.
- Dependency resolution retains `deepagents==0.6.12`, `langchain==1.3.11`,
  `langgraph==1.2.7`, and `langgraph-sdk==0.4.2`. Services that resolve `orjson`
  pin `3.11.5`, the last upstream release limited to the project's
  Apache-2.0/MIT policy. Redis checkpoint packages remain absent.
- Every turn retains additive agent diagnostics with activated skill-file
  paths, model-issued tool calls and arguments in order, deterministic
  rejection/duplicate markers, bounded structured product evidence from
  successful current-turn catalog search/detail results, a truncation flag,
  bounded `catalog_scope_outcomes` for `zero_results`, and a final termination
  reason. Per-search scopes remain
  attached to their own products. When a graph fails, its current-turn
  assistant/tool messages are read from the latest checkpoint before that
  checkpoint is deleted. Public query responses return an empty diagnostics
  object by default; only trusted operator/evaluation deployments explicitly
  enable exposure with `EXPOSE_AGENT_DIAGNOSTICS=true`.
- Catalog, cart, policy, availability, and promotions tools return to the Deep Agents loop
  so a single shopper turn can complete a compound request before the final
  shopper-facing response.
- The main Deep Agent reasons over the rolling semantic summary and full bounded
  shopper/assistant tail. Final evidence composition uses a separate narrow
  lane: current query, bounded shopper-authored continuity, active-skill
  guidance, historical product identity, current cart/images, and actual
  current-turn tool-role messages isolated by the server-owned request marker.
  It does not receive the rolling summary, prior assistant prose, prior tool
  output, or the main agent's draft.
  Every successful search records the model-authored semantic query as
  independent internal `SEARCH_DIRECTION_EVIDENCE` and the required
  pre-retrieval, product-agnostic `shopper_guidance` authored under the active
  skill. Completed successful search-only responses may receive one
  tools-disabled synthesis under the active skill, but the final composer starts
  from its allowed evidence lanes rather than that draft. Activated no-tool
  turns use the same boundary with an empty typed-evidence lane, preventing
  prior assistant facts from bypassing current authority. Static skill
  `response_guidance` and pre-retrieval guidance support deterministic fallback.
  Before fallback guidance becomes shopper-facing text, a
  narrow scrub replaces documented unsupported outdoor/weather guarantee terms
  with neutral selected-role guidance without changing the semantic query,
  taxonomy, hard constraints, or retrieval. Covered forms include
  outdoor-surface or outdoor-walking claims and constructions such as "handle rain," "work well
  for outdoor surfaces," or "stay secure for outdoor walking," plus `wet
  conditions` and "works well in wet weather/conditions." Product results,
  filters, and drafts are not converted into guidance after retrieval.
  Deterministic code separately
  renders every name, price, category, and search-scoped filter group, and
  deduplicates grouped candidates by `product_ref`, not display name.
  Mixed-outcome turns preserve successful groups when another scope has an
  unsupported requirement. A fixed unsupported-requirement response applies
  only when that rejection is the sole current-turn business-tool outcome.
  Scoped zero-result evidence retains the exact
  advertised taxonomy and filters and cannot establish broader absence.
- Final-response extraction ignores tool messages, assistant messages that
  still contain tool calls, and internal activation markers. If no
  shopper-facing answer remains, the runtime returns a safe retry response and
  records `incomplete_agent_response`.
- Optional VLM media perception runs before the Deep Agents turn when the
  `vlm` model role is enabled. It converts attached image/video media into a
  concise `MEDIA ANALYSIS` text block. Raw media is not persisted in
  conversation memory. Descriptive look-analysis requests are answered from
  `MEDIA ANALYSIS` without catalog retrieval; catalog tools remain authoritative
  for product names and prices when the shopper explicitly asks to find,
  compare, price-check, or add products. The no-I/O availability stub resolves
  only known conversation product refs and applies a fixed category rule; it
  does not query live inventory.

Filesystem and built-in Deep Agents tools:

- Deep Agents includes filesystem, todo, shell, and subagent tools by default.
- This shopping runtime registers a harness profile that excludes built-in
  filesystem write/edit/list/search tools, todo tools, shell tools, and the
  default general-purpose subagent.
- Core skill activation does not rely on model-issued `read_file` calls or
  pagination: the runtime injects complete selected files. Built-in `read_file`
  remains available only for static skill reference documents such as the
  shared trend snapshot, through a virtual-mode filesystem backend rooted at
  `chain_server/skills`.
- Customer profile, cart, price, inventory, order, and payment truth must not
  live in local files or the Deep Agents virtual filesystem.
- Skill instructions and reference files are static application assets. Any
  future store-backed per-conversation files must never use a shared writable
  customer namespace.

## Goals

- Use the Deep Agents SDK as the shopping assistant harness.
- Keep application code small, readable, and conventional.
- Remove bespoke planner and cart-agent orchestration from our code path.
- Prepare for adding more shopping tools over time.
- Prepare for Deep Agents skills without moving business rules into prompts.
- Prevent customer, session, conversation, cart, or persona context bleeding.
- Keep the public website API stable during the first migration.
- Support future high-scale retail traffic where many customers are active at
  the same time.

## Non-Goals

- Do not implement a full production cart lifecycle in the first migration.
- Do not rewrite catalog, memory, guardrails, or UI services as part of the
  first slice.
- Do not use the agent runtime as the source of truth for carts, customer
  profiles, prices, inventory, or orders.
- Do not add broad keyword routing or category-specific hacks to replace
  planner behavior.

## Design Rules

1. No two customers may share agent context.
2. No two active conversations may share a Deep Agents thread unless explicitly
   chosen by the backend.
3. Never use `customer_id` alone as the Deep Agents `thread_id`.
4. The server creates and owns session identity first; public API changes can
   come later when the website needs explicit thread management.
5. The Deep Agent semantically selects a skill set in a required first phase.
   The runtime owns complete skill loading and the pre-tool execution gate;
   deterministic tools own validation, state mutation, idempotency, and
   authorization.
6. Caller-supplied persona data remains unavailable. The UI may send one ID
   from the trusted, typed, read-only registry of five fixed representative
   shoppers. Durable turn start resolves and binds the server-owned snapshot;
   only its type, behavior, and ZIP enter current-turn context as soft guidance.
7. Carts are scoped by `cart_id`, not by conversation memory.
8. Conversation memory is scoped by `conversation_id`, not by customer alone.
9. Skills describe shopping behavior and domain knowledge. Tools perform
   reads, writes, and external service calls.
10. The model maps shopper intent to exact advertised taxonomy values.
    The same catalog capabilities define the allowed non-taxonomy constraint
    properties. Deterministic code validates those structured values and
    enforces tool-loop bounds; it does not replace semantic interpretation with
    keyword rules.

## Target Request Flow

```text
POST /query/stream
  -> resolve or create server-owned identity
  -> start durable conversation turn; load semantic summary, exact post-watermark raw tail, product index, prior selected skills, cart, and attempt token
  -> keep the memory-owned compaction source outside normal model context
  -> replay stored finalized output and stop, when request identity matches
  -> invoke Deep Agents SDK with thread_id = [conversation_id, request_id]
  -> give the main agent semantic summary + full bounded recent discussion; derive a shopper-only continuity lane for final composition
  -> force structured per-turn skill selection
  -> load and inject complete selected SKILL.md files
  -> expose deterministic shopping tools
  -> generate taxonomy and required-constraint schemas from catalog capabilities
  -> select and validate exact advertised values or stop on a no-retrieval path
  -> tools call catalog and cart services or controlled policy/availability/promotion boundaries
  -> stop the graph at the configured execution deadline and finalize agent_timeout
  -> compose from shopper-authored continuity + current typed evidence, excluding prior assistant prose and the agent draft
  -> finalize durable turn as completed, blocked, or failed with the current attempt token
  -> stream assistant response back to the UI
```

The first migration preserves the existing endpoint shape. The bundled UI sends
explicit browser-session identifiers, and the server can still derive internal
`session_id`, `conversation_id`, `cart_id`, and `request_id` for legacy callers
that do not send them.

## Identity Model

| Identifier | Meaning | Persistence | May Be Shared? |
| --- | --- | --- | --- |
| `customer_id` | Logged-in shopper identity. Optional for anonymous users. | Durable | Across that customer's sessions only |
| `session_id` | Website/browser session. | TTL-bound | Never across customers |
| `conversation_id` | One durable chat thread and one component of the request-scoped graph `thread_id`. | TTL-bound or user-visible thread | Never across unrelated sessions |
| `cart_id` | Active cart being mutated. | Cart lifecycle policy | May follow a customer across sessions |
| `persona_id` | Optional persona/profile source. | Durable or scenario-bound | Never across unrelated customers |
| `request_id` | One submitted turn and exact replay identity. | Durable with the turn-retention policy | Unique per turn |
| `attempt_id` | Service-issued execution fence for one start attempt; rotated on an allowed abandoned retry. | Until terminal finalize or a newer attempt | Returned by start; echoed only to finalize that attempt |

Server-side generation is the first step. Later, explicit session and
conversation APIs can expose these identifiers for multi-thread website
features.

The current bridge uses the legacy numeric `user_id` when explicit IDs are
absent, derives a stable internal key from `conversation_id` for conversation
memory when present, and derives a separate stable internal key from `cart_id`
for cart reads/writes when present. Callers may provide one bounded `request_id`
per turn; otherwise the server generates it. The memory service stores a digest
of shopper text and ordered media hashes under that ID. Matching finalized
retries replay exactly; changed input conflicts. Cart adds persist catalog
`product_id`, while reads expose the opaque, non-reusable
`CartItem.cart_line_id` as `CART_LINE_ID` for remove and quantity update. All
three mutation paths commit with one owner-scoped idempotency record. A retry may
reopen an abandoned turn only when it remains the conversation's latest
sequence. Reopening preserves `request_id` but rotates `attempt_id`, so a late
finalize from the older attempt is rejected rather than overwriting the retry.

Product evidence identity is a separate, narrower bridge. The runtime keeps at
most one request's returned or uniquely resolved `PRODUCT_REF` values in
request-local evidence. Ordered product cards from a finalized turn are stored
as durable same-conversation events. The validated compact projection also
provides an exact strict capability: one unconflicted opaque `PRODUCT_REF` can
authorize only its scalar detail read without a resolver call. Natural,
ordinal, shortened, and ambiguous references still use the typed resolver;
only a unique result enters general request-local evidence for details,
availability, or cart adds. This evidence is not part of the LangGraph
checkpoint. Product IDs remain valid only within the catalog's identity
guarantees, and stale-revision invalidation is not yet implemented.

## Session Isolation Requirements

- A request without a valid server session gets a new `session_id`.
- A request without a valid conversation gets a new `conversation_id`.
- Deep Agents working checkpoint state uses a collision-safe pair of conversation
  ID and request ID as the thread key. Durable conversation memory uses
  `conversation_id`.
- Durable raw turn, replay-output, and event rows use `conversation_id` plus
  ordered sequence and unique `request_id` identities. Each active execution is
  fenced by a memory-service-issued `attempt_id`.
- General product-ref authorization is request-local and is established by
  current-turn search or one unique durable same-conversation resolution. An
  exact unconflicted ref from the validated historical projection has the
  narrower authority to initiate only its scalar detail read. Neither path is
  recovered from the checkpoint.
- Customer profile lookup uses `customer_id`, but profile lookup must not merge
  conversation state across sessions.
- Anonymous sessions must not be addressable by guessable numeric IDs.
- Resetting the chat creates a new conversation. It should not automatically
  clear the customer's durable cart unless the user explicitly clears cart.
- Deleting a durable conversation removes its turn, event, and reserved
  projection rows. It does not delete cart, cart-mutation replay, or legacy user
  context rows.
- Same customer in two active browser sessions gets separate conversation
  memory unless the product explicitly opens the same conversation thread.

## Persona Handling

Personas are useful, but they are not chat memory and they are not cart state.

Persona precedence:

1. Current user request.
2. Store-policy and catalog tool results.
3. Explicit session preferences.
4. Loaded customer persona/profile snapshot.
5. Model assumptions are not allowed as facts.

The current runtime ignores unknown caller fields for backward compatibility
and never injects caller-supplied persona data; it also does not expose a
persona-loading tool. The bundled UI sends only a selected ID from the typed,
bounded, server-owned registry of five immutable representative shoppers.
Durable turn start resolves and binds the row atomically, and the runtime
renders one compact soft-guidance block. User-owned or mutable profiles
additionally require authenticated ownership and input-safety validation.
Individual skills should not independently fetch mutable persona state, because
that creates inconsistent context and hard-to-debug race conditions.

## Tools

The tool layer is small, typed, and deterministic:

- `search_catalog`: read-only, stateless product discovery.
- `get_product_details`: read-only product facts for one product authorized by
  current evidence, unique semantic resolution, or an exact strict historical
  `PRODUCT_REF` capability.
- `resolve_conversation_products`: read-only deterministic resolution against
  products presented in the same durable conversation; one batched call returns
  0/1/many matches and only a unique match becomes request-local evidence.
- `get_cart`: read-only cart state for a `cart_id`.
- `add_cart_item`: mutating cart write using a `PRODUCT_REF` previously returned
  by product search; identical idempotency-key retries replay the stored result.
- `remove_cart_item`: mutating cart write using a
  `CART_LINE_ID` returned by cart reads; identical retries replay once.
- `update_cart_item`: mutating cart quantity update using a current
  `CART_LINE_ID`; one absolute-value `PUT` sets the quantity and `0` removes the
  line. Identical idempotency-key retries replay the stored result; conflicting
  reuse is rejected without mutation.
- `view_cart_total`: deterministic arithmetic over cart line prices.
- `get_store_policy`: read-only controlled policy content.
- `check_product_availability`: read-only deliberate stub that reports general,
  sized apparel/footwear, or one-size availability for a known conversation
  product ref without calling live inventory.
- `check_active_promotions`: read-only no-I/O stub that reports no active sale or
  promotion configured through the assistant. Catalog retrieval and price do
  not establish markdown status.
- `get_weather_forecast`: implemented provider-neutral read-only contract for
  an exact US ZIP and optional ISO date/range, but dormant and not registered,
  granted, prompted, or connected to shopper context.

`load_customer_persona` remains unregistered. Representative-shopper context is
resolved as part of the existing durable turn-start call, not through an
agent-callable tool. It does not grant skills or tools and is not independently
refetched by shopper skills.

`get_weather_forecast` is absent from the serving registry. Direct construction
validates a closed ZIP/date schema, returns bounded normalized daily evidence
through the Visual Crossing adapter, rejects non-live forecast sources, and
emits sanitized typed failures. `WEATHER_ENABLED=false` is the default and
needs no credential; an enabled direct client reads the key only from the
environment variable named by config. A later leveraging slice owns any agent
registration, shopper-context connection, grounded weather evidence,
attribution display, or uncertainty language.

The active shopper-serving Deep Agents tool registry is documented in
[Shopper Agent Tool Registry](SHOPPER_AGENT_TOOL_REGISTRY.md). The active skill
registry is documented in
[Shopper Agent Skill Registry](SHOPPER_AGENT_SKILL_REGISTRY.md). Those
registries are the source for which tools and skills are actually registered
today, their boundaries, and current limitations. Historical migration slices
below record the state at that time; they do not override the active registries.

The activation control tool is forced at turn start. After activation, the Deep
Agent decides when to call a shopping tool; that tool decides whether the call
is valid and what state changes are allowed.

Tool-loop control is a separate deterministic boundary. The turn receives one
structural catalog-repair opportunity total. Runtime does not normalize
`requested_product_type` into a repair key, classify shopper wording, lock a
semantic scope, or reject a corrected role because its noun changed. The
isolated call receives the capability-derived typed search tool, compact
server-generated Catalog capabilities, the current shopper message, bounded
sanitized validator feedback, and the complete active shopper-skill
instructions. It exposes only the search tool, keeps tool choice automatic so
the model may signal clarification by returning no tool call, and disables
parallel calls. A no-tool repair is branch/control state: the server marks it,
discards the model prose, and emits the fixed clarification `Could you clarify
the product type or requirement you want me to use?`. Its message list contains
only the current shopper message and validator feedback in a separate Human
data message. Echoed rejected arguments are stripped, quoted text is explicitly
data, and the invalid AI/tool exchange and all other history are absent.

The model may correct semantic fields, including `requested_product_type` and
taxonomy. Runtime validates the repaired call afresh against structural and
capability-owned invariants and never rewrites those fields. Independently valid
finite structural fields—advertised `required_constraints`, `scope_complete`,
and `search_mode`—may be preserved, with any restoration named in bounded
`restored_fields` diagnostics. Malformed or nonempty free-form
`unadvertised_requirements` closes without repair or semantic provenance review.
After the repair is used, a later invalid catalog call closes to synthesis. A
successful partial repaired search may continue with later valid work, but no
second repair is available in that turn.
After a successful search is marked `scope_complete`, when a requirement or
taxonomy is unenforceable, or whenever a tool returns `STOP_TOOL_USE`, further
tool use closes. A successful or zero-result search that consumes the final
configured search slot instead records `SEARCH_BUDGET_EXHAUSTED`; the next
model step removes only the search tool, preventing a futile additional search
without blocking product details, availability, cart work, or honest partial
synthesis. Completed turns receive one tools-disabled synthesis from collected
evidence. The final composer then starts from the narrower allowed evidence
lanes, with deterministic rendering as fail-closed fallback.

The search contract has two explicit no-search boundaries. When the model
cannot select a faithful advertised scope, the assistant asks one concise
clarification directly and performs no retrieval. There is no tool-authored
`no_direct_catalog_match` outcome. When the model does select a category-only
text scope, tool evidence neutrally records the requested role and advertised
category searched. It does not assert that the requested role is unavailable,
prove that the category is its parent, or relabel results. Any nonempty
`unadvertised_requirements` lane is returned as unenforceable before retrieval.
Runtime does not inspect shopper prose to decide whether the model copied an
explicit requirement or inferred one, and it performs no
constraint-provenance review.

Every non-image search carries required `requested_product_type` product
noun/umbrella provenance and pre-retrieval `shopper_guidance` in addition to the
taxonomy-independent semantic query and structured catalog scope. The provenance
is the shortest product noun or true umbrella from the
shopper's current turn or direct antecedent, excluding color, material, fit,
occasion, weather, and style modifiers; it is not a catalog enum or ranking
query. Image-only search sets it to `null`. The model owns advertised taxonomy
selection and every exact, umbrella, open-role, alternative, comparison,
ordering, negation, and parent-category interpretation. Capability validation
enforces exact values and category/subcategory ownership without comparing
those semantics with shopper prose. When the model submits multiple advertised
subcategories from one category through the typed taxonomy field, the selection
produces one catalog execution, with a widened candidate window and
rank-preserving coverage when each selected subcategory returns a candidate.

Final composition accepts current-request facts only from tool-role messages
isolated by the request marker. The model's reasoning loop still sees the
rolling summary and bounded full recent discussion, but the composer receives
only the current query, shopper-authored recent continuity, active-skill
guidance, the historical product identity index, current cart/images, and
current typed tool evidence. Prior assistant prose, prior tool output, the
rolling summary, and the main agent draft are absent. Completed successful
search-only turns may receive one tools-disabled synthesis under the active
skill, but final composition starts again from the permitted evidence lanes.
Pre-retrieval `shopper_guidance` and static `response_guidance` support
deterministic fallback. Candidate facts and confirmed filters are rendered
deterministically, with each search retaining its own guidance, product, and
filter-evidence group. Grouped candidates deduplicate by `product_ref`, not
display name. Mixed-outcome turns retain successful product groups when another
scope has an unsupported requirement. Fixed unsupported-requirement responses
apply only when the rejection is the sole
current-turn business-tool outcome. A partial successful result set gets a neutral offer to continue with the
next requested piece or search scope. A zero-result group retains its exact
scope and supports no claim about other product types or catalog-wide absence.
## Skills

Skills are domain instructions and examples, not hidden business logic. The
registered set is:

- `product-discovery`: the primary procedure for general search, browse, and
  filtering without styling intent.
- `outfit-styling`: fashion judgment across anchors, complete looks,
  conversational follow-ups, and visual requests; it is the primary procedure
  instead of product discovery when styling is requested and remains primary
  for terse item-only follow-ups within that active thread. Cart-aware styling
  uses cart evidence supplied by a co-active `cart-management` skill.
- `cart-management`: explicit cart reads and mutations.
- `budget-shopping`: a modifier for a stated price ceiling or budget bundle.
- `store-policy-answers`: controlled policy answers through the policy tool.

Skills should contain domain procedure and judgment. Tool schemas own transport,
and runtime policy owns enforcement. Skills should not own cart mutation,
pricing, inventory, profile persistence, or order creation. Slice 2 applies
this boundary to `outfit-styling`; cleanup of `product-discovery` remains
deferred.

The runtime separates three phases:

1. **Registration** discovers metadata and makes a skill eligible for a turn.
2. **Activation** is a forced, structured model decision over the registered
   name enum, using the current request and conversation context.
3. **Loading** is deterministic application behavior: the runtime injects the
   complete selected files into the next model request.

The activation prompt and enum come directly from the current validated files
on every turn. The runtime does not enable the Deep Agents `SkillsMiddleware`
metadata channel in parallel, because that metadata is checkpointed and can be
stale for an existing checkpoint thread if skill files change while the process
remains alive. The filesystem middleware remains enabled for read-only
reference files.

During activation, only `activate_shopper_skills_tool` is visible and tool
choice is forced to it with parallel calls disabled. After a successful
activation, that control tool is removed and only the selected skills' grant
union becomes visible. The execution middleware independently rechecks the same
grant and requires a successful activation message for the current
`request_id`; therefore a model cannot bypass the boundary by emitting an
unadvertised shopping call or by placing activation and shopping calls in one
batch. A load failure exposes no shopping tools and fails closed. The
middleware also rejects a direct activation-phase answer with final termination
reason `skill_activation_failed`, so named tool choice is not the only
enforcement layer.

When a discovery or styling procedure applies, activation selects exactly one
of `product-discovery` and `outfit-styling`. `budget-shopping` is added only for
an explicit shopper budget. This composition rule lives in the activation
contract and the skill files; it is not a keyword router.

This design normally adds one bounded app-model step per shopper turn. An
invalid composition may add one corrective app-model step; a repeated invalid
composition returns a deterministic clarification without another model call.
It avoids broad keyword routing and keeps intent selection semantic, while
making the selected skill paths and any gate rejection observable. Correct
selection among the registered skills remains a model-quality concern;
silently running a shopping turn without complete skill instructions is no
longer possible.

Potential future roles include dedicated visual shopping and authenticated
persona-aware recommendation. Product comparison remains a procedure inside
the existing product or styling skill; do not promote it to another skill or
subagent unless evaluation demonstrates a separate tool budget or private
planning boundary is necessary.

## Filesystem And Context

Deep Agents supports virtual filesystem backends for skills, memory, and
context management. That is different from a regular single-loop agent, where
all relevant context is often stuffed into the prompt.

For this retail assistant:

- Use filesystem-backed or store-backed skills for static domain behavior,
  instructions, and examples.
- Use Deep Agents conversation state for per-thread working context.
- Use the memory-service SQLite turn store for ordered raw shopper/assistant
  history and exact finalized replay in a single-replica deployment. Serving
  continuity has three independent lanes: a non-authoritative rolling semantic
  summary, an exact bounded post-watermark raw tail, and the compact historical-
  product index. The memory-owned oldest compaction prefix is a separate source
  lane and is never copied into normal shopper-model context.
- Give the main Deep Agent the summary and exact bounded recent tail for
  reasoning. Derive a separate bounded shopper-authored continuity projection
  for final evidence composition; do not copy the summary, prior assistant
  prose, prior tool output, or the agent draft into that final lane.
- Do not use local files as the production source of truth for customer
  profiles, carts, prices, inventory, orders, or payment state.
- Do not share a writable filesystem namespace across customers.
- If virtual files are used for conversation context, namespace them by
  `conversation_id` and ensure lifecycle cleanup.
- At high scale, prefer store-backed or database-backed context over local disk.
  Local disk state does not scale cleanly across replicas.

The rule is: Deep Agents context is useful for reasoning and skill discovery;
commerce truth belongs in application services and databases.

Durable raw transcript is not the same as structured memory. The current Slice 5
boundary interprets only products actually sent as ordered cards: it stores one
candidate-set event, maintains a compact reference index, and resolves exact
typed descriptors to 0/1/many matches. Active anchors, preferences, sentiment,
fuzzy lookup, embeddings, and cross-conversation memory remain outside this
minimal design.

## Scaling Assumptions

This system should be able to grow toward a high-traffic retail site, including
peak events such as Black Friday.

Implications:

- App servers should remain horizontally scalable and mostly stateless.
- Session, cart, persona, and conversation state must live in shared durable
  stores, not process memory or local files.
- Slice 4 is a deliberate single-replica stepping stone: Compose persists the
  memory-service SQLite database on `memory-data`, but SQLite remains one local
  writer and is not the shared multi-replica production store described above.
- Deep Agents uses process-local MemorySaver keyed by a collision-safe pair of
  conversation ID and request ID. It holds one request's working graph state and
  is deleted after successful durable finalization. It is retained only when
  finalization fails and still disappears on process restart; it is not shopper
  memory and need not be shared for cross-turn continuity.
- Same-conversation product follow-ups use durable presented-product events and
  deterministic resolution. A unique match enters the current request's
  evidence; zero or many matches require clarification.
- Catalog revision is recorded when supplied but not yet enforced. A stable
  cross-catalog identity or revision invalidation is required before old refs
  can be guaranteed across catalog replacement.
- Mutating tools use owner-scoped idempotency keys so retries do not apply a
  cart change twice.
- Tool calls need timeouts, retry policy, and clear structured failures.
- Agent turns have both a strict graph-step limit and a configurable 45-second
  execution deadline so one broad or stalled request cannot monopolize model
  capacity. Successful failed-turn finalization releases the durable
  conversation turn; pre-graph work, bounded state capture, and finalization are
  outside the graph deadline, so the client must retain its own request timeout.
  A finalization outage preserves the checkpoint and remains an explicit retry
  condition.
- Catalog search should remain stateless and cacheable where possible.
- Cart writes have owner checks, one SQLite transaction for mutation plus replay
  record, and request idempotency. Multi-writer revision control remains future
  production work.
- Returned tool payloads should be compact. Large result sets should be
  summarized or offloaded instead of injected wholesale into context.
- Observability must include `request_id`, `session_id`, `conversation_id`,
  `cart_id`, ordered tool names and arguments, activated skill-file paths,
  rejected or duplicate calls (including `skill_activation_required`), final
  termination reason, latency, and error class. Failed-turn graph messages must
  be captured before checkpoint cleanup.
- Backpressure is required for model calls and slow downstream services.

## Migration Slices

The slices below are the historical implementation sequence. Their scope
statements describe the boundary at the time each slice landed; the Current
Implementation Notes and active registries above describe today's serving
runtime.

### Slice 1: Deep Agents Adapter

- Add a small adapter that creates and invokes the Deep Agents SDK harness.
- Keep `/query/stream` as the public entrypoint.
- Current PR decision: cut over the chain server to the Deep Agents harness
  directly and do not keep the bespoke LangGraph runtime as a selectable
  fallback. This keeps the harness path simple and readable while quality is
  improved in follow-up PRs.
- Known limitation for this slice: `/query/stream` remains SSE-framed but emits
  completed turn events after the Deep Agents turn finishes instead of
  token-level model chunks. Token-level Deep Agents streaming is a follow-up
  after the harness migration is stable.

### Slice 2: Stable Conversation Identity

- Accept explicit `session_id`, `conversation_id`, and `cart_id` from clients
  that have a stable browser/session identity.
- Keep deriving `request_id` server-side per turn.
- Combine `conversation_id` and `request_id` for the request-scoped Deep Agents
  `thread_id`.
- Keep the current request body compatible for legacy callers.
- The bundled UI now creates session-scoped IDs and sends them on every turn.
- Add isolation tests before relying on this in production.

### Slice 3: Shopping Tools

- Expose current catalog, cart, policy, availability, and promotions capabilities as typed
  tools.
- Keep product search stateless.
- Keep cart operations deterministic and idempotent.
- Make cart mutations ref-based: add by `PRODUCT_REF`, remove by
  `CART_LINE_ID`; do not hide product lookup or fuzzy cart-line matching inside
  mutation tools.
- Keep discovery/read tools chainable inside the agent loop so same-turn
  requests like "find a black bag and add it" can search first, then mutate by
  ref.
- Do not create standalone planner or cart-agent classes.

### Slice 4: Skills

- Add and iterate the registered product-discovery, outfit-styling,
  cart-management, budget-shopping, and store-policy skills.
- Require semantic per-turn activation and deterministic complete-file loading
  before exposing shopping tools.
- Keep product discovery and outfit styling mutually exclusive as primary
  procedures; add budget shopping only as an explicit-budget modifier.
- Keep skill instructions as guidance. Keep state changes in tools.

### Slice 5: Deep Agents Runtime Parity

- Persist one `candidate_set_presented` event only from finalized ordered
  product-card output; never from hidden search candidates or assistant prose.
- Maintain a compact per-conversation product-reference index while retaining
  the full authoritative product payload in durable events. Keep the newest
  complete candidate sets within a 16,384-character serialized cap.
- Add one exact typed batch resolver available only to product discovery,
  outfit styling, and cart management. Return 0/1/many deterministically; add
  only a unique result to request-local evidence and require clarification for
  zero or many. Enforce at most one batched resolver call per turn.
- Scope MemorySaver to a collision-safe pair of conversation ID and request ID;
  delete it only after successful durable finalization and preserve it on
  finalize failure.
- Do not add model calls, catalog changes, preferences, sentiment, active-anchor
  inference, fuzzy/embedding lookup, cross-conversation memory, or stale-
  revision enforcement in this slice.
- Run existing unit tests.
- Run targeted Challenger scenarios.
- Verify product search, image search, cart mutation claims, and grounded
  responses.
- Give the Judge the actual ordered prior shopper and generated assistant turns
  plus bounded per-turn structured evidence from successful catalog search and
  detail messages, its truncation flag, and bounded `catalog_scope_outcomes` for
  `zero_results`. Exclude semantic queries, raw
  tool messages, model reasoning, and every other diagnostic field. Treat
  generated history as authoritative over counterfactual reference-answer
  assumptions.
- Compare Deep Agents output against the committed golden integration
  conversations and prior WIP/committed baselines before subsequent quality
  changes.

### Slice 6: Cutover And Cleanup

- Continue improving the Deep Agents runtime now that the harness is the chain
  server path.
- Add token-level streaming for `/query/stream`.
- Remove bespoke planner/cart/retriever/chatter graph code only after the new
  path is stable.
- Keep deterministic tools and service contracts.

## Required Tests

- Two customers ask similar questions; context does not bleed.
- Same customer has two sessions; conversation context does not bleed.
- Same session has multiple turns; context is preserved.
- Only finalized ordered product-card output creates durable reference evidence.
- Exact typed historical resolution returns 0/1/many; only one match enters
  request-local evidence, while zero or many require clarification.
- An exact unconflicted historical `PRODUCT_REF` can initiate its scalar detail
  read without a resolver call. A conflicting, malformed, natural, ordinal, or
  shortened reference cannot use that strict path and performs no unauthorized
  catalog read.
- A chain-server restart does not remove durable same-conversation presented-
  product evidence.
- Graph thread IDs combine conversation and request identity; successful durable
  finalize deletes the thread, while finalize failure preserves it.
- Persona A and Persona B do not cross-contaminate.
- Cart writes affect only the intended `cart_id`.
- Retried cart mutation with the same idempotency key applies once.
- Search remains stateless and does not receive customer/session/cart context.
- Chatter does not claim cart changes unless a tool reports success.
- The first model step exposes only the forced skill-activation control tool.
- A successful activation injects every complete selected file before exposing
  only those skills' combined tool grants.
- A prior-turn activation, failed activation, or same-batch activation does not
  authorize a current shopping tool call.
- Prior-turn selected skill names cross the durable turn boundary explicitly and
  appear only as a read-only continuity signal in the next activation prompt;
  the model still makes a fresh semantic choice.
- Activation instructions and skill files require product discovery and outfit
  styling to remain alternative primary procedures and budget shopping to be
  selected only for a stated budget. Terse item-only follow-ups inside an active
  outfit or style-led single-piece thread remain styling intent.
- The turn receives one structural catalog-repair opportunity total. The
  isolated step receives the capability-derived typed `search_catalog_tool`,
  compact server-generated Catalog capabilities, the current shopper turn,
  bounded sanitized validator feedback, and the complete active shopper-skill
  instructions. It exposes only `search_catalog_tool`, keeps tool choice
  automatic so the model may signal clarification by returning no tool call,
  and disables parallel calls. Only that no-tool repair response receives the
  server clarification marker; the marker selects the branch, while model prose
  is discarded and replaced with `Could you clarify the product type or
  requirement you want me to use?`. Invalid AI/tool history and the base
  runtime prompt are absent.
- Repair tests prove that the model may correct semantic fields, including
  `requested_product_type` and taxonomy, while runtime validates the corrected
  call independently and never derives a semantic repair key, compares the
  corrected noun with shopper text, or rewrites those fields. Independently
  valid finite structural fields—advertised `required_constraints`,
  `scope_complete`, and `search_mode`—may be preserved and are named in bounded
  `restored_fields` diagnostics. A successful partial repaired search may
  continue with later valid work, but a later invalid catalog call closes to
  synthesis because no second repair is available in that turn. Malformed or
  nonempty free-form `unadvertised_requirements` closes without repair or
  semantic provenance review. Successful completed turns receive one
  tools-disabled synthesis from collected evidence; final composition starts
  from the narrower allowed evidence lanes, with deterministic fallback.
- Taxonomy and required-constraint schemas contain only capability-derived
  advertised fields and values, plus the explicit
  `unadvertised_requirements` lane.
- Every search schema includes product-agnostic `shopper_guidance` authored
  before retrieval under the active skill. It must be nonempty except for
  image-only search and cannot contain candidate facts or internal search
  mechanics.
- Every text search requires `requested_product_type` product noun/umbrella
  provenance: the shortest product noun or true umbrella from the current turn
  or direct antecedent, excluding color, material, fit, occasion, weather, and
  style modifiers; image-only search requires it to be `null`. The model owns
  exact, umbrella, open-role, alternative, comparison, ordering, negation, and
  parent-category interpretation. Runtime does not parse shopper prose or test
  semantic equivalence between this field and taxonomy. When the model submits
  multiple exact advertised subcategories from one category through the typed
  taxonomy field, runtime executes the selection once. It widens the candidate
  window and preserves ranking while keeping one returned candidate per
  selected subcategory when available.
- Each search has at most one category. Runtime validates exact advertised
  values and category/subcategory ownership only. A category-only text search
  records the model-authored requested role and advertised category separately;
  that evidence is neutral and cannot prove a parent relationship, assert
  unavailability, or relabel returned products. If the model cannot select a
  faithful advertised scope, the assistant asks one concise clarification
  without a tool call or catalog-absence claim.
- Duplicate search identity is normalized taxonomy plus hard constraints;
  changing only semantic wording cannot repeat a retrieval.
- Any nonempty `unadvertised_requirements` lane fails closed before retrieval.
  Runtime does not classify the requirement as explicit or inferred and opens
  no semantic provenance review.
- `zero_results` is the only no-product catalog scope outcome. It applies only
  to the exact submitted advertised taxonomy and filters and cannot establish
  absence for another product type or the catalog as a whole. There is no
  direct-match or taxonomy-relationship outcome.
- A successful or zero-result search that consumes the final configured search
  slot carries `SEARCH_BUDGET_EXHAUSTED`. The next model step exposes no search
  tool but retains applicable product-detail, availability, and cart tools plus
  honest partial synthesis.
- Final composition accepts facts only from current-turn tool-role evidence
  isolated by request ID. Its prompt contains shopper-authored continuity but
  excludes the rolling summary, prior assistant prose, prior tool output, and
  the main agent's draft; prior assistant claims therefore cannot survive as
  current evidence on a later product-only turn.
- Completed successful search-only output receives one tools-disabled synthesis
  under the active skill and then grounding against tool-role evidence.
  Pre-retrieval `shopper_guidance` and static `response_guidance` support
  deterministic fallback. Before fallback guidance becomes shopper-facing text, the
  runtime replaces documented prohibited outdoor/weather guarantee terms with
  neutral selected-role guidance without changing search semantics. Covered
  forms include outdoor-surface or outdoor-walking claims and constructions such
  as "handle rain," "work well for outdoor surfaces," or "stay secure for
  outdoor walking," plus `wet conditions` and "works well in wet
  weather/conditions."
  Deterministic code renders every candidate,
  adds a neutral continuation for partial successful evidence, and groups each
  search's guidance and filters with its originating products. It deduplicates
  grouped candidates by `product_ref`, not display name, and preserves
  successful groups across unsupported-requirement mixed outcomes. Fixed canned
  responses apply only when that rejection is the sole current-turn
  business-tool outcome. Scoped zero-result
  evidence cannot support a different-type or catalog-wide absence claim.
- Final-response extraction cannot return tool, tool-calling, or internal
  activation text. An otherwise empty answer returns a safe fallback and records
  `incomplete_agent_response`.
- Diagnostics preserve selected skill paths, ordered arguments, rejected and
  duplicate calls, termination reason, bounded product evidence and truncation,
  bounded no-product catalog scope outcomes, and bounded partial graph messages
  on failure. Public responses suppress that trace by default. Evaluation
  deployments explicitly enable it so the Judge receives only generated
  conversation history, product evidence/truncation, and catalog scope outcomes
  from those diagnostics.

## Resolved Decisions

- `CHECKPOINT_STORE=memory` is the only currently supported checkpoint
  configuration. Non-memory values fail during startup rather than silently
  falling back to process-local state.
- MemorySaver is request-scoped with a collision-safe pair of conversation ID
  and request ID. The runtime deletes it after successful durable finalization and preserves it after a
  finalize failure. Durable turns and presented-product events, not the graph
  checkpoint, provide cross-turn continuity.
- Deep Agents graph execution and the final evidence composer share one 45-second
  model-stage deadline through `DEEPAGENTS_EXECUTION_TIMEOUT_SECONDS`. The composer
  receives only the remaining time. A graph timeout finalizes as failed with
  `agent_timeout`, preserves bounded partial diagnostics, and clears product
  cards and images. It can deterministically render valid current-request typed
  search or detail evidence only when all observed and pending business
  calls are classified read-only by the immutable tool policy; mutating or
  unknown calls force the fixed retry/cart-check response. Salvaged text crosses
  output guardrails. A grounding timeout finalizes as failed with
  `grounding_timeout`; typed search and detail evidence use their
  deterministic renderers, while other turns return the fixed retry/cart-check
  response instead of the unverified draft. Other composer
  errors and empty or whitespace-only output use the same response rule with
  `grounding_error`. Checkpoint release
  occurs only after durable finalization.
- Durable turns use one memory-service SQLite replica with transactional start,
  terminal finalize, exact finalized replay, a rolling semantic summary, an
  exact bounded post-watermark raw tail, a memory-owned compaction-source lane,
  latest-sequence-only abandoned reopen, rotating attempt tokens, and
  operator-owned retention. Summary advances use projection-version and
  boundary compare-and-swap validation and commit atomically with finalization;
  invalid output, timeout, conflict, or failed turns leave the source available
  for retry. A stale finalize is rejected; the runtime emits a safe superseded-
  attempt response, while a generic finalize outage preserves the grounded
  response. Finalized ordered product cards produce durable
  `candidate_set_presented` events and a compact reference index; the typed
  same-conversation resolver adds only unique matches to request-local evidence.
- Registered skills use `/shopper/<skill>/SKILL.md` under the virtual backend;
  shared read-only references may live directly under `/shopper`.
## Open Decisions

- Which shared multi-writer store should replace single-replica SQLite for
  durable turns in a horizontally scaled deployment?
- What retention and deletion policy should apply to durable raw turns, replay
  output, and event envelopes?
- Should anonymous sessions use cookies, headers, or both?
- What is the cart TTL for anonymous and logged-in users?
- What retention and cleanup policy should apply to cart mutation idempotency
  records?
- Which typed persona fields and trusted profile source should a future
  authenticated integration support?
- Which tools require human approval or confirmation before mutation?
