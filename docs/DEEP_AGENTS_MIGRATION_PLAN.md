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
  schema constructed in `chain_server/src/deepagents_runtime.py`, with reusable
  model-visible rules in `chain_server/src/catalog_scope.py`. Its fields are
  `semantic_query`, `shopper_guidance`, `requested_product_type`, `taxonomy`,
  `required_constraints`, `scope_complete`, and optional `search_mode`. It has
  no model-authored taxonomy relationship, clarification branch, or
  catalog-absence result. A shopper-named type not separately advertised may
  use one model-selected faithful advertised parent category while preserving
  the type as semantic direction. If neither a direct type nor one faithful
  parent can be selected, the assistant asks one concise clarification directly
  without calling the tool.
- The Deep Agents adapter exposes eleven thin request-scoped shopping tools
  over deterministic catalog, conversation-product, cart, policy,
  availability, and promotions functions, plus one internal skill-activation
  control tool.
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
  guardrail, model, or tool work and returns bounded model-context-eligible raw
  turns plus the authoritative cart. Blocked turns remain durable and exactly
  replayable but are excluded from both the service projection and chain prompt
  formatter. Every new turn is finalized as completed, blocked, or failed. An
  exact finalized request replay skips model/tool execution and returns stored
  response output. Each start returns a service-issued attempt
  token that finalization must echo. A start failure prevents agent execution;
  a generic finalize failure preserves the grounded response and request
  checkpoint and adds an operator diagnostic.
- The legacy rolling context blob is no longer the serving continuity source;
  bounded durable shopper/assistant turns replace it. Finalized ordered product
  cards also create durable `candidate_set_presented` events and a compact
  per-conversation index. One typed batch resolver returns 0/1/many exact
  matches; only one match becomes request-local product evidence. The compact
  index keeps the newest complete candidate sets within 16,384 serialized
  characters, and runtime permits at most one batched resolver call per turn.
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
  and requested type are empty only for image-only search. Duplicate identity
  is normalized taxonomy plus hard constraints, so paraphrasing cannot repeat
  the same retrieval.
- Cached Catalog capabilities generate the search schema's exact taxonomy and
  hard-filter values. The model owns semantic selection; deterministic code
  validates and maps the selection but does not interpret shopper language.
  Each search covers at most one category. A genuinely open role selects and
  names exactly one advertised subcategory. A shopper-named role retains the
  shopper's noun or umbrella. A type not separately advertised may use one
  model-selected faithful advertised parent category and remains semantic
  direction rather than a claimed catalog type. If neither a direct type nor
  one faithful parent can be selected, the assistant clarifies without
  retrieval, substitution, or an absence claim.
  Unsupported modifiers do not erase an advertised type. Directly stated
  must-haves absent from the generated schema remain in
  `unadvertised_requirements` and fail closed instead of becoming semantic
  preferences.
- The server keys one total repair to the full normalized
  `requested_product_type` phrase. The isolated repair receives only the
  capability-derived search tool, compact server-generated Catalog
  capabilities, the current shopper message, bounded sanitized validator
  feedback, and the active skill instructions. Tool choice remains automatic:
  the model may submit one corrected search or return no tool call to signal
  that clarification is needed. That no-tool response is only branch/control
  state: the server marks it, discards the model prose, and emits the fixed
  clarification
  `Could you clarify the product type or requirement you want me to use?`.
  Parallel calls remain disabled.
  The repair cannot replace a shopper-grounded product noun, taxonomy,
  validated constraints, or search mode. Middleware may restore only structural
  `scope_complete`. A successful partial search may continue to another role
  with its own repair opportunity, but no scope receives two repairs and the
  configured turn cap remains three searches.
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
- Grounding uses actual tool-role messages only. Current-turn evidence is
  isolated by the server-owned request marker; prior-turn tool evidence may
  resolve direct references but cannot prove that a new search or mutation ran.
  Every successful search records the model-authored semantic query as
  independent internal `SEARCH_DIRECTION_EVIDENCE` and the required
  pre-retrieval, product-agnostic `shopper_guidance` authored under the active
  skill. Completed successful search-only responses receive one tools-disabled
  synthesis under the active skill and then the grounding editor. Static skill
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
6. Persona data remains unavailable until a trusted source and typed,
   input-safe untrusted-data boundary are implemented. A future snapshot is
   read-only for the turn unless a later feature supports profile updates.
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
  -> start durable conversation turn; load bounded model-context turns, prior selected skills, cart, and attempt token
  -> replay stored finalized output and stop, when request identity matches
  -> invoke Deep Agents SDK with thread_id = [conversation_id, request_id]
  -> force structured per-turn skill selection
  -> load and inject complete selected SKILL.md files
  -> expose deterministic shopping tools
  -> generate taxonomy and required-constraint schemas from catalog capabilities
  -> select and validate exact advertised values or stop on a no-retrieval path
  -> tools call catalog and cart services or controlled policy/availability/promotion boundaries
  -> stop the graph at the configured execution deadline and finalize agent_timeout
  -> ground current-turn results separately from prior-turn tool evidence
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
as durable same-conversation events. The typed resolver can restore a unique
earlier product after restart or on another worker; ambiguous and missing
references do not authorize details, availability, or cart adds. This evidence
is not part of the LangGraph checkpoint. Product IDs remain valid only within
the catalog's identity guarantees, and stale-revision invalidation is not yet
implemented.

## Session Isolation Requirements

- A request without a valid server session gets a new `session_id`.
- A request without a valid conversation gets a new `conversation_id`.
- Deep Agents working checkpoint state uses a collision-safe pair of conversation
  ID and request ID as the thread key. Durable conversation memory uses
  `conversation_id`.
- Durable raw turn, replay-output, and event rows use `conversation_id` plus
  ordered sequence and unique `request_id` identities. Each active execution is
  fenced by a memory-service-issued `attempt_id`.
- Product-ref authorization is request-local and is established by current-turn
  search or one unique durable same-conversation resolution. It is not recovered
  from the checkpoint.
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

The current runtime does not accept or inject caller-supplied persona data and
does not expose a persona-loading tool. A future implementation must use a
typed, bounded schema, authenticate profile ownership, validate values through
the input-safety boundary, and present the snapshot explicitly as untrusted
data rather than system instructions. Individual skills should not
independently fetch mutable persona state, because that creates inconsistent
context and hard-to-debug race conditions.

## Tools

The tool layer is small, typed, and deterministic:

- `search_catalog`: read-only, stateless product discovery.
- `get_product_details`: read-only product facts for one product.
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

`load_customer_persona` and typed turn-start persona snapshots remain planned;
neither is available in the current runtime.

The active shopper-serving Deep Agents tool registry is documented in
[Shopper Agent Tool Registry](SHOPPER_AGENT_TOOL_REGISTRY.md). The active skill
registry is documented in
[Shopper Agent Skill Registry](SHOPPER_AGENT_SKILL_REGISTRY.md). Those
registries are the source for which tools and skills are actually registered
today, their boundaries, and current limitations. Planned contracts in this
section should not be treated as agent-callable until they appear in the
registries as registered runtime capabilities.

The activation control tool is forced at turn start. After activation, the Deep
Agent decides when to call a shopping tool; that tool decides whether the call
is valid and what state changes are allowed.

Tool-loop control is a separate deterministic boundary. It keys one total model
repair to the full normalized, model-authored `requested_product_type` phrase.
It does not reconstruct alternatives, negation, ordering, or comparisons from
shopper prose. A schema correction or fresh
constraint-provenance review can consume that budget. Constraint feedback from
an in-flight schema repair closes the loop for synthesis. The isolated call
receives the capability-derived typed search tool, compact server-generated
Catalog capabilities, the current shopper message, bounded sanitized validator
feedback, and the complete active shopper-skill instructions. It exposes only
the search tool, keeps tool choice automatic so the model may signal
clarification by returning no tool call, and disables parallel calls. A no-tool
repair is only branch/control state: the server marks it, discards the model
prose, and emits the fixed clarification
`Could you clarify the product type or requirement you want me to use?`. Its
message list contains only the current shopper message and validator feedback
in a separate Human data message. Echoed rejected arguments are
stripped, quoted text is explicitly data, and the invalid AI/tool exchange and
all other history are absent. A successful partial search
may continue when another valid role remains; the next role receives its own
repair opportunity, and no scope receives a second repair. For native
tool-transport failures, the current scope is locked only when current or recent
shopper text grounds it; an ungrounded model-generated scope may be corrected.
A changed shopper-grounded scope is rejected and records
`repair_scope_changed`. Middleware never restores or rewrites taxonomy,
constraints, requested type, or search mode; it may restore only structural
`scope_complete`, recorded by name in bounded `restored_fields` diagnostics.
Runtime validation separately protects capability-owned advertised sibling
relationships. On a native schema-invalid call, malformed or nonempty free-form
`unadvertised_requirements` arguments close without repair. A schema-valid,
genuinely open request retains the bounded review for a proposed inferred
requirement.
After a successful search is marked `scope_complete`, when a requirement or
taxonomy is unenforceable, or whenever a tool returns `STOP_TOOL_USE`, further
tool use closes. A successful or zero-result search that consumes the final
configured search slot instead records `SEARCH_BUDGET_EXHAUSTED`; the next
model step removes only the search tool, preventing a futile additional search
without blocking product details, availability, cart work, or honest partial
synthesis. Completed turns receive one tools-disabled synthesis from collected
evidence. Search-only drafts then pass through grounding, with deterministic
rendering as fail-closed fallback.

The search contract has two explicit no-search boundaries. A concrete
shopper-named type not separately advertised may first use one model-selected
faithful advertised parent category, with results framed as closest alternatives
under their actual catalog types. When neither a direct type nor one faithful
parent can be selected, the assistant asks one concise clarification directly
and performs no retrieval. An unsupported
modifier does not erase an otherwise advertised type. Every
unadvertised requirement on a shopper-stated product scope is returned as
unenforceable before retrieval, including when the model uses a synonym rather
than the shopper's exact wording. The bounded constraint review is reserved for
a proposed inferred requirement on a genuinely open role when its shared repair
remains. That review preserves requested type, taxonomy, completion state,
`search_mode`, and all advertised hard
constraints. Within that preserved hard scope, it may correct only the soft
`semantic_query`, the reviewed unadvertised-requirement lane, and its associated
guidance; the requirement is either replaced with the shopper's shortest exact
wording or removed. Exact wording and unresolved provenance fail closed.
Removal scrubs product-attribute guidance. When a runtime semantic
open-role schema repair removes a proposed inferred requirement, runtime
replaces the submitted pre-search guidance with neutral generic guidance for
the selected role.

Every non-image search carries required `requested_product_type` product
noun/umbrella provenance and pre-retrieval `shopper_guidance` in addition to the
taxonomy-independent semantic query and structured catalog scope. The provenance
is the shortest product noun or true umbrella from the
shopper's current turn or direct antecedent, excluding color, material, fit,
occasion, weather, and style modifiers; it is not a catalog enum or ranking
query. Image-only search sets it to `null`. The model owns advertised taxonomy
selection. Capability-owned exact category/subcategory relationships validate
the selection. A genuinely open role selects one advertised subcategory and
names it in `requested_product_type`; that path is rejected for a shopper-named
scope rather than silently reinterpreted.
The model owns alternative, comparison, ordering, and negation semantics. When
it submits multiple advertised subcategories from one category through the
typed taxonomy field, the selection produces one catalog execution, with a
widened candidate window and rank-preserving coverage when each selected
subcategory returns a candidate. The runtime does not derive that selection
from the shopper's raw text.

Final grounding accepts evidence only from tool-role messages. The current
request is isolated by its request marker and prior-turn evidence is supplied
separately for references to products already shown. Completed successful
search-only turns receive one tools-disabled synthesis under the active skill,
then grounding against tool-role evidence. Pre-retrieval `shopper_guidance` and
static `response_guidance` support deterministic fallback. Candidate facts and confirmed filters are rendered
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

Planned future roles include dedicated visual shopping, product comparison, and
persona-aware recommendation. Keep them as markdown-guided behavior until
evaluation shows a need for a dedicated subagent or separate tool budget.

## Filesystem And Context

Deep Agents supports virtual filesystem backends for skills, memory, and
context management. That is different from a regular single-loop agent, where
all relevant context is often stuffed into the prompt.

For this retail assistant:

- Use filesystem-backed or store-backed skills for static domain behavior,
  instructions, and examples.
- Use Deep Agents conversation state for per-thread working context.
- Use the memory-service SQLite turn store for ordered raw shopper/assistant
  history and exact finalized replay in a single-replica deployment. The
  bounded recent-turn snapshot replaces the legacy rolling context blob.
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
- Each full normalized `requested_product_type` scope receives one total repair.
  Alternative, comparison, ordering, and negation semantics remain model-owned;
  deterministic repair does not reconstruct them from shopper prose. A schema
  correction or a fresh constraint-provenance review may consume the repair;
  constraint
  feedback from an in-flight schema repair closes for synthesis. The isolated
  step receives the capability-derived typed `search_catalog_tool`, compact
  server-generated Catalog capabilities, the current shopper turn, bounded
  sanitized validator feedback, and the complete active shopper-skill
  instructions. It exposes only `search_catalog_tool`, keeps tool choice
  automatic so the model may signal clarification by returning no tool call,
  and disables parallel calls. Only that no-tool repair response receives the
  server clarification marker;
  the marker selects the branch, while model prose is discarded and replaced
  with `Could you clarify the product type or requirement you want me to use?`.
  Invalid AI/tool history and the base runtime prompt are absent. A successful
  partial search may continue to another valid role and its own one-repair
  opportunity, and a second invalid call in the same scope and every
  `STOP_TOOL_USE` result close the loop. For native transport failures, a repair may change an
  ungrounded model-generated scope, but not one grounded in current or recent
  shopper text. Middleware may restore only structural `scope_complete`,
  recorded by name in bounded `restored_fields` diagnostics; it never rewrites
  taxonomy, constraints, requested type, or search mode. A changed grounded
  scope is rejected under `repair_scope_changed`. On a native schema-invalid
  call, malformed or nonempty free-form `unadvertised_requirements` arguments
  close without repair. Successful
  completed turns receive one tools-disabled synthesis from collected evidence;
  search-only drafts then pass through grounding with deterministic fallback.
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
  style modifiers. For a genuinely open role, it is the one advertised
  subcategory selected by the model; image-only search requires it to
  be `null`. Literal suffix binding can recover an advertised type from a
  modifier phrase but is disabled for explicit alternatives containing `and`,
  `or`, `/`, or `&`, leaving their umbrella/alternative interpretation
  model-owned. When the model submits multiple exact advertised subcategories
  from one category through the typed taxonomy field, runtime executes the
  selection once. It widens the candidate window and preserves ranking while
  keeping one returned candidate per selected subcategory when available. The
  runtime does not extract alternative members from shopper prose.
- Each search has at most one category. A genuinely open role selects exactly
  one advertised subcategory as a focused starting role only when the shopper
  named no type for that role, and names it in `requested_product_type`.
  Alternatives, confirmations, comparisons, and follow-ups count as named
  types. Invalid open-role provenance is rejected rather than silently
  reinterpreted.
- A product type not separately advertised may use one model-selected faithful
  advertised parent category, with its original noun retained as semantic
  direction and every result kept under its actual catalog type. If neither a
  direct type nor one faithful parent can be selected, the assistant asks one
  concise clarification without a tool call. An unenforceable direct must-have
  also performs no catalog retrieval. Unsupported modifiers do not erase an
  advertised type, and subjective style stays semantic.
- Duplicate search identity is normalized taxonomy plus hard constraints;
  changing only semantic wording cannot repeat a retrieval.
- A nonempty unadvertised-requirement lane on a shopper-stated product scope
  fails closed even when the model uses a synonym rather than the shopper's
  exact wording. The bounded constraint review is reserved for a proposed
  inferred requirement on a genuinely open role when the scope's shared repair
  remains; it copies the shopper's shortest
  exact wording or removes the inferred value while freezing requested type,
  taxonomy, completion state, `search_mode`, and all advertised
  hard constraints. Within that preserved hard scope, only the soft semantic
  query, reviewed unadvertised-requirement lane, and associated guidance may be
  corrected. Exact wording and unresolved provenance fail closed. Removal
  scrubs product-attribute guidance. A runtime semantic open-role schema
  repair that removes its proposed inferred requirement substitutes neutral
  generic pre-search guidance for the selected role.
- A successful or zero-result search that consumes the final configured search
  slot carries `SEARCH_BUDGET_EXHAUSTED`. The next model step exposes no search
  tool but retains applicable product-detail, availability, and cart tools plus
  honest partial synthesis.
- Grounding accepts only tool-role evidence, isolates current-turn evidence by
  request ID, and cannot treat a prior assistant draft as tool evidence.
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
- Deep Agents graph execution and the grounding editor share one 45-second
  model-stage deadline through `DEEPAGENTS_EXECUTION_TIMEOUT_SECONDS`. The editor
  receives only the remaining time. A graph timeout finalizes as failed with
  `agent_timeout` and preserves bounded partial diagnostics. A grounding timeout
  finalizes as failed with `grounding_timeout`; search-only evidence uses
  deterministic catalog rendering, while other turns return a fixed
  retry/cart-check response instead of the unverified draft. Other editor
  errors and empty or whitespace-only output use the same response rule with
  `grounding_error`. Checkpoint release
  occurs only after durable finalization.
- Durable turns use one memory-service SQLite replica with transactional start,
  terminal finalize, exact finalized replay, a bounded recent-turn snapshot,
  latest-sequence-only abandoned reopen, rotating attempt tokens, and
  operator-owned retention. A stale finalize is rejected; the runtime emits a
  safe superseded-attempt response, while a generic finalize outage preserves
  the grounded response. Finalized ordered product cards produce durable
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
