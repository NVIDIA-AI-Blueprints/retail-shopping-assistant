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

- The Deep Agents adapter exposes nine thin request-scoped shopping tools over
  deterministic catalog, cart, policy, and availability functions, plus one
  internal skill-activation control tool.
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
  tool. The prior turn's selected names are a read-only continuity signal for
  the next fresh semantic selection; they do not force routing or satisfy the
  activation gate. Shopping calls issued before activation or in the same model
  batch are rejected at execution time.
- Caller-supplied persona data is not injected into model context. Persona
  support remains deferred until its trust and validation contract is defined.
- Optional `session_id`, `conversation_id`, and `cart_id` fields are accepted,
  and the bundled UI now sends browser-session identifiers on every turn.
  Legacy callers that send only `user_id` are still mapped to deterministic
  compatibility identifiers.
- The compatibility mapping is not the final production identity design. A
  high-scale website should move to server-created session, conversation, and
  cart identifiers before broad rollout.
- The runtime caps Deep Agents recursion and distinct catalog taxonomy scopes
  per turn. The agent tool accepts one taxonomy-independent semantic query and
  one required pre-retrieval, product-agnostic `shopper_guidance` sentence. It
  must be nonempty except for `image_only` and `no_direct_catalog_match`, which
  require empty guidance.
  Duplicate identity is
  normalized taxonomy plus hard constraints, so paraphrasing cannot repeat the
  same retrieval while a genuinely different hard-filter scope may execute
  within the cap.
- Cached catalog capabilities generate both the exact taxonomy enum and the
  non-taxonomy required-constraint properties exposed to the model. The agent
  owns semantic selection through a structural transport schema; runtime then
  applies a separate strict semantic search model, so cross-field failures reach
  capability-aware validation and receive exact corrections. The model owns
  `taxonomy_status`; runtime never semantically rewrites it. Exact advertised
  category/subcategory coherence comes from the capability contract. Every
  text search also carries required `requested_product_type` provenance: the
  shortest product noun or true umbrella from the shopper's current turn or
  direct antecedent, excluding color, material, fit, occasion, weather, and
  style modifiers. It is `null` only for image-only search. Literal validation
  may bind the longest exact advertised suffix in a modifier-bearing phrase,
  such as `waterproof boots` to `boots`, but disables that shortcut for explicit
  alternatives containing `and`, `or`, `/`, or `&`; `closed shoes or boots`
  remains model-owned alternative or umbrella reasoning. One unambiguous literal
  pair of exact advertised subcategories in the current shopper turn is the
  bounded exception: runtime requires both model-authored branches under
  `member_of_requested_umbrella`. The pair uses one catalog execution with a
  pair-wide candidate window and rank-preserving coverage of each branch when a
  candidate is returned. Modified, synonymous, ambiguous, and cross-category
  alternatives remain model-owned. One call covers at most one category.
  `agent_selected_type` selects exactly one advertised
  subcategory as the focused starting role when a broad request names no
  concrete type. It is
  forbidden for a role whose type the
  shopper named, including an alternative, confirmation, comparison, or
  follow-up. Invalid open-role provenance is rejected rather than silently
  reinterpreted. Runtime retains `agent_selected_type` and derives the duplicate
  requested-type provenance from the selected advertised subcategory.
- `no_direct_catalog_match` stops before retrieval when an explicitly requested
  concrete type has no faithful advertised value; it uses empty taxonomy and no
  hard constraints. An unsupported modifier does not erase an advertised type,
  and subjective style remains semantic direction. A directly stated product
  must-have absent from the generated schema is retained in
  `unadvertised_requirements`, so it is reported as unenforceable rather than
  weakened into semantic relevance.
- The server keys one total repair to the full normalized
  `requested_product_type` phrase and protects distinct advertised siblings from
  being treated as the same scope. A schema correction or a fresh
  constraint-provenance review can consume that budget; constraint feedback from
  an in-flight schema repair closes the loop for synthesis rather than opening a
  second repair. The isolated request uses a concise, schema-generic system
  prompt instead of the base runtime prompt; the skill gate appends the complete
  active shopper-skill instructions. It exposes and forces only
  `search_catalog_tool` and disables parallel calls. Active responses containing
  more than one shopping tool call are rejected before execution. It contains
  only the current shopper message plus bounded, sanitized validator feedback in
  a separate Human data message. Echoed rejected arguments are stripped; native
  Pydantic feedback contains only rejected top-level field names, and free-form
  requested-scope text is not replayed. Invalid AI/tool history and earlier
  conversation history are absent. For native tool-transport failures, the scope is locked
  only when current or recent shopper text grounds it; an ungrounded
  model-generated scope may be corrected. A change to a grounded free-form
  scope that cannot be reconstructed safely is removed before execution and
  recorded in `agent_diagnostics` as `repair_scope_changed`. A strict request
  failure with independently valid constraints snapshots its
  capability-validated advertised
  `required_constraints` privately and places that exact finite object,
  including an explicit empty object, in the isolated feedback. This bounded
  capability-derived object is the exception to excluding free-form rejected
  arguments. Before execution, runtime restores every independently valid
  finite lock: the taxonomy relation, canonical advertised constraints
  (including an explicit empty object), explicit valid `scope_complete` and
  `search_mode`, and `requested_product_type` when a singleton exact or
  agent-selected taxonomy determines it. The model owns only invalid fields.
  Drift in a restorable lock is corrected in place; bounded tool-call
  diagnostics expose only affected field names in `restored_fields`. The lock
  follows accepted modifier removal, list-valued constraints compare
  canonically, and omitted optional defaults
  equal explicit empty values. A no-direct repair may clear constraints only
  while remaining no-direct; changing to retrieval must preserve the original
  advertised constraints. Native enum failures on
  `agent_selected_type` include the shopper-named/open-role provenance rule in
  the same bounded feedback. A shopper-named advertised subtype repairs to
  `exact_requested_type`; named umbrellas and alternatives repair to
  `member_of_requested_umbrella`. A valid no-direct outcome after repair keeps
  the specific not-advertised response. A native failure confined to
  `required_constraints` adds only its finite, validated taxonomy status and
  selection to repair feedback; free-form scope, query, and guidance remain
  excluded while scope is compared privately. Relation drift is restored before
  the repaired constraint call executes. A native taxonomy failure likewise
  restores independently valid constraints before execution. A locked boundary
  that cannot be restored safely remains comparison-protected and closes under
  the matching `repair_*_changed` reason. Malformed or nonempty free-form
  `unadvertised_requirements` arguments are never restored; a native
  schema-invalid call containing one closes without repair. A schema-valid,
  genuinely open `agent_selected_type` request retains the bounded review for a
  proposed inferred requirement. Every unadvertised requirement on a
  shopper-stated product scope fails closed, including when the model uses a
  synonym rather
  than the shopper's exact wording. The bounded constraint review is reserved
  for a proposed inferred requirement on a genuinely open
  `agent_selected_type` role when the shared repair remains; it freezes
  requested type, taxonomy status, taxonomy, completion state, `search_mode`,
  and every advertised hard constraint. Within that preserved hard scope, it
  may correct only the soft `semantic_query`, the reviewed
  unadvertised-requirement lane, and its associated guidance; the requirement
  is either replaced with the shopper's shortest exact wording or removed.
  Exact wording
  and unresolved provenance fail closed. Removal scrubs product-attribute
  guidance while `semantic_query` remains available for ranking. When a runtime
  semantic open-role schema repair removes its proposed inferred requirement,
  runtime replaces the submitted pre-search guidance with neutral generic
  guidance for the selected role. A successful partial search may continue to
  another valid role with its own one-repair
  opportunity. When a successful or zero-result search consumes the final
  configured slot, the result records `SEARCH_BUDGET_EXHAUSTED`; the next model
  step removes only `search_catalog_tool`. Product-detail, availability, and
  cart tools plus honest partial synthesis remain available.
- Dependency resolution retains `deepagents==0.6.12`, `langchain==1.3.11`,
  `langgraph==1.2.7`, and `langgraph-sdk==0.4.2`. Services that resolve `orjson`
  pin `3.11.5`, the last upstream release limited to the project's
  Apache-2.0/MIT policy. Redis checkpoint packages remain absent.
- Every turn exposes additive agent diagnostics with activated skill-file
  paths, model-issued tool calls and arguments in order, deterministic
  rejection/duplicate markers, bounded structured product evidence from
  successful current-turn catalog search/detail results, a truncation flag,
  bounded `catalog_scope_outcomes` for `no_direct_catalog_match` and
  `zero_results`, and a final termination reason. Per-search scopes remain
  attached to their own products. When a graph fails, its current-turn
  assistant/tool messages are read from the latest checkpoint before that
  checkpoint is deleted.
- Catalog, cart, policy, and availability tools return to the Deep Agents loop
  so a single shopper turn can complete a compound request before the final
  shopper-facing response.
- Grounding uses actual tool-role messages only. Current-turn evidence is
  isolated by the server-owned request marker; prior-turn tool evidence may
  resolve direct references but cannot prove that a new search or mutation ran.
  Every successful search records the model-authored semantic query as
  independent internal `SEARCH_DIRECTION_EVIDENCE` and the required
  pre-retrieval, product-agnostic `shopper_guidance` authored under the active
  skill. Completed successful search-only responses present that guidance
  without a final-synthesis model call; static skill `response_guidance` is the
  fallback. Before guidance becomes deterministic shopper-facing evidence, a
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
  Mixed-outcome turns preserve successful groups when another scope has no
  direct match or an unsupported requirement. A fixed no-direct or unsupported
  canned response applies only when that rejection is the sole current-turn
  business-tool outcome. Scoped zero-result evidence retains the exact
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
  compare, price-check, or add products. The availability tool reports
  `unknown` until a live inventory service exists.

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
  -> load session, conversation, and cart
  -> invoke Deep Agents SDK with thread_id = conversation_id
  -> force structured per-turn skill selection
  -> load and inject complete selected SKILL.md files
  -> expose deterministic shopping tools
  -> generate taxonomy and required-constraint schemas from catalog capabilities
  -> select and validate exact advertised values or stop on a no-retrieval path
  -> tools call catalog and cart services or controlled policy/availability boundaries
  -> ground current-turn results separately from prior-turn tool evidence
  -> stream assistant response back to the UI
  -> persist conversation state under the correct scope
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
| `conversation_id` | One chat thread and Deep Agents `thread_id`. | TTL-bound or user-visible thread | Never across unrelated sessions |
| `cart_id` | Active cart being mutated. | Cart lifecycle policy | May follow a customer across sessions |
| `persona_id` | Optional persona/profile source. | Durable or scenario-bound | Never across unrelated customers |
| `request_id` | One submitted turn. | Short-lived | Unique per turn |

Server-side generation is the first step. Later, explicit session and
conversation APIs can expose these identifiers for multi-thread website
features.

Current bridge implementation keeps the memory service schema unchanged. It
uses the legacy numeric `user_id` when explicit IDs are absent, derives a
stable internal key from `conversation_id` for conversation memory when present,
and derives a separate stable internal key from `cart_id` for cart reads/writes
when present. Cart reads expose the opaque, non-reusable
`CartItem.cart_line_id` as `CART_LINE_ID`; absolute quantity updates use one
service `PUT` scoped by that ID and cart owner, with the idempotency record and
mutation committed atomically.

Product evidence identity is a separate, narrower bridge. The runtime keeps at
most 50 returned `PRODUCT_REF` values per `conversation_id` in process memory,
and those refs are valid only for the active catalog snapshot. This cache is not
part of the LangGraph checkpoint. Both the graph checkpoint and product refs are
currently process-local, but they remain separate state. Another replica,
cache eviction, restart, or catalog replacement requires a fresh search before
product details or cart adds.

## Session Isolation Requirements

- A request without a valid server session gets a new `session_id`.
- A request without a valid conversation gets a new `conversation_id`.
- Deep Agents checkpointing, memory, and filesystem state use
  `conversation_id` as the thread key.
- Product-ref authorization is process-local and snapshot-bound; it is not
  recovered from the checkpoint.
- Customer profile lookup uses `customer_id`, but profile lookup must not merge
  conversation state across sessions.
- Anonymous sessions must not be addressable by guessable numeric IDs.
- Resetting the chat creates a new conversation. It should not automatically
  clear the customer's durable cart unless the user explicitly clears cart.
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
  by product search; its generated key is metadata until add deduplication is
  implemented in the memory service.
- `remove_cart_item`: mutating cart write using a
  `CART_LINE_ID` returned by cart reads.
- `update_cart_item`: mutating cart quantity update using a current
  `CART_LINE_ID`; one absolute-value `PUT` sets the quantity and `0` removes the
  line. Identical idempotency-key retries replay the stored result; conflicting
  reuse is rejected without mutation.
- `view_cart_total`: deterministic arithmetic over cart line prices.
- `get_store_policy`: read-only controlled policy content.
- `check_product_availability`: read-only deliberate stub that reports
  `unknown` until live inventory exists.

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
repair to the full normalized `requested_product_type` phrase and never
conflates distinct advertised sibling scopes. A schema correction or fresh
constraint-provenance review can consume that budget. Constraint feedback from
an in-flight schema repair closes the loop for synthesis. The concise,
schema-generic prompt replaces the base runtime prompt, and the skill gate
appends the complete active shopper-skill instructions. The repair exposes and
forces only the search tool and disables parallel calls. Its message list
contains only the current shopper message and bounded, sanitized validator
feedback in a separate Human data message. Echoed rejected arguments are
stripped, quoted text is explicitly data, and the invalid AI/tool exchange and
all other history are absent. A successful partial search
may continue when another valid role remains; the next role receives its own
repair opportunity, and no scope receives a second repair. For native
tool-transport failures, the current scope is locked only when current or recent
shopper text grounds it; an ungrounded model-generated scope may be corrected.
A grounded free-form scope that cannot be reconstructed safely remains
comparison-protected and records `repair_scope_changed`. Independently valid
finite locks are restored before execution, and bounded `restored_fields`
diagnostics list names only. Runtime validation separately protects
capability-owned advertised sibling relationships. On a native schema-invalid
call, malformed or nonempty free-form `unadvertised_requirements` arguments
remain outside restoration and close without repair; a schema-valid, genuinely
open `agent_selected_type` request retains the bounded review for a proposed
inferred requirement.
After a successful search is marked `scope_complete`, when a requirement or
taxonomy is unenforceable, or whenever a tool returns `STOP_TOOL_USE`, further
tool use closes. A successful or zero-result search that consumes the final
configured search slot instead records `SEARCH_BUDGET_EXHAUSTED`; the next
model step removes only the search tool, preventing a futile additional search
without blocking product details, availability, cart work, or honest partial
synthesis. Completed successful search-only evidence renders deterministically;
mixed-tool paths may receive synthesis from their collected evidence.

The search contract has two explicit no-search boundaries. A concrete requested
type with no faithful taxonomy value uses `no_direct_catalog_match` and performs
no retrieval; that status carries empty taxonomy arrays and no hard constraints.
An unsupported modifier does not erase an otherwise advertised type. Every
unadvertised requirement on a shopper-stated product scope is returned as
unenforceable before retrieval, including when the model uses a synonym rather
than the shopper's exact wording. The bounded constraint review is reserved for
a proposed inferred requirement on a genuinely open `agent_selected_type` role
when its shared repair remains. That review preserves requested type, taxonomy
status, taxonomy, completion state, `search_mode`, and all advertised hard
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
selection and `taxonomy_status`; runtime never semantically rewrites the status.
Capability-owned exact category/subcategory relationships validate the
selection. `agent_selected_type` is rejected for a shopper-named scope rather
than silently reinterpreted. For a genuinely open role, the model selects one
advertised subcategory and runtime derives the duplicate requested-type
provenance while retaining `agent_selected_type`.
One unambiguous literal pair of exact advertised subcategories in the current
shopper turn is validated as a whole: the model-authored request must retain
both branches under `member_of_requested_umbrella`. It still produces one
catalog execution, with a widened candidate window and rank-preserving branch
coverage when each branch returns a candidate. All nonliteral alternative
reasoning remains model-owned.

Final grounding accepts evidence only from tool-role messages. The current
request is isolated by its request marker and prior-turn evidence is supplied
separately for references to products already shown. Completed successful
search-only turns present pre-retrieval `shopper_guidance` authored under the
active skill; static `response_guidance` is the fallback. There is no
final-synthesis model call. Candidate facts and confirmed filters are rendered
deterministically, with each search retaining its own guidance, product, and
filter-evidence group. Grouped candidates deduplicate by `product_ref`, not
display name. Mixed-outcome turns retain successful product groups when another
scope has no direct match or an unsupported requirement. Fixed no-direct and
unsupported canned responses apply only when the rejection is the sole
current-turn business-tool outcome. A partial successful result set gets a neutral offer to continue with the
next requested piece or search scope. A zero-result group retains its exact
scope and supports no claim about other product types or catalog-wide absence.

## Skills

Skills are domain instructions and examples, not hidden business logic. The
registered set is:

- `product-discovery`: the primary procedure for general search, browse, and
  filtering without styling intent.
- `outfit-styling`: fashion styling across anchor product, no-anchor discovery,
  cart styling, conversational mid-browse, and visual requests; it is the
  primary procedure instead of product discovery when styling is requested and
  remains primary for terse item-only follow-ups within that active outfit or
  style-led single-piece thread.
- `cart-management`: explicit cart reads and mutations.
- `budget-shopping`: a modifier for a stated price ceiling or budget bundle.
- `store-policy-answers`: controlled policy answers through the policy tool.

Skills may guide tool use, explain constraints, and provide examples. They
should not own cart mutation, pricing, inventory, profile persistence, or order
creation.

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
activation, that control tool is removed and the nine shopping tools become
visible. The execution middleware independently requires a successful
activation message for the current `request_id`; therefore a model cannot
bypass the boundary by emitting an unadvertised shopping call or by placing
activation and shopping calls in one batch. A load failure exposes no shopping
tools and fails closed. The middleware also rejects a direct activation-phase
answer with final termination reason `skill_activation_failed`, so named tool
choice is not the only enforcement layer.

When a discovery or styling procedure applies, activation selects exactly one
of `product-discovery` and `outfit-styling`. `budget-shopping` is added only for
an explicit shopper budget. This composition rule lives in the activation
contract and the skill files; it is not a keyword router.

This design adds one bounded app-model step per shopper turn. It avoids broad
keyword routing and keeps intent selection semantic, while making the selected
skill paths and any gate rejection observable. Correct selection among the
registered skills remains a model-quality concern; silently running a shopping
turn without complete skill instructions is no longer possible.

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
- Do not use local files as the production source of truth for customer
  profiles, carts, prices, inventory, orders, or payment state.
- Do not share a writable filesystem namespace across customers.
- If virtual files are used for conversation context, namespace them by
  `conversation_id` and ensure lifecycle cleanup.
- At high scale, prefer store-backed or database-backed context over local disk.
  Local disk state does not scale cleanly across replicas.

The rule is: Deep Agents context is useful for reasoning and skill discovery;
commerce truth belongs in application services and databases.

## Scaling Assumptions

This system should be able to grow toward a high-traffic retail site, including
peak events such as Black Friday.

Implications:

- App servers should remain horizontally scalable and mostly stateless.
- Session, cart, persona, and conversation state must live in shared durable
  stores, not process memory or local files.
- Deep Agents currently uses process-local MemorySaver keyed by
  `conversation_id`. It loses graph state on restart and does not share threads
  across workers or replicas. Successful thread histories also remain in heap
  without a TTL or capacity bound, so it does not yet meet the production
  scaling assumption above.
- Graph checkpointing does not cover the separate process-local product-ref
  cache. Cross-replica product follow-ups require either a fresh search or a
  future shared evidence store with catalog-snapshot identity.
- Mutating tools must use idempotency keys so retries do not double-add items.
- Tool calls need timeouts, retry policy, and clear structured failures.
- Agent turns need strict step limits so one broad request cannot monopolize
  model capacity during traffic spikes.
- Catalog search should remain stateless and cacheable where possible.
- Cart writes need ownership checks, optimistic concurrency or transactions,
  and request idempotency.
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
- Map `conversation_id` to the Deep Agents `thread_id`.
- Keep the current request body compatible for legacy callers.
- The bundled UI now creates session-scoped IDs and sends them on every turn.
- Add isolation tests before relying on this in production.

### Slice 3: Shopping Tools

- Expose current catalog, cart, policy, and availability capabilities as typed
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

- Run existing unit tests.
- Run targeted Challenger scenarios.
- Verify product search, image search, cart mutation claims, and grounded
  responses.
- Give the Judge the actual ordered prior shopper and generated assistant turns
  plus bounded per-turn structured evidence from successful catalog search and
  detail messages, its truncation flag, and bounded `catalog_scope_outcomes` for
  `no_direct_catalog_match` and `zero_results`. Exclude semantic queries, raw
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
- Persona A and Persona B do not cross-contaminate.
- Cart writes affect only the intended `cart_id`.
- Retried cart mutation with the same idempotency key applies once.
- Search remains stateless and does not receive customer/session/cart context.
- Chatter does not claim cart changes unless a tool reports success.
- The first model step exposes only the forced skill-activation control tool.
- A successful activation injects the complete selected file before exposing
  all nine shopping tools.
- A prior-turn activation, failed activation, or same-batch activation does not
  authorize a current shopping tool call.
- Prior-turn selected skill names appear only as a read-only continuity signal
  in the next activation prompt; the model still makes a fresh semantic choice.
- Activation instructions and skill files require product discovery and outfit
  styling to remain alternative primary procedures and budget shopping to be
  selected only for a stated budget. Terse item-only follow-ups inside an active
  outfit or style-led single-piece thread remain styling intent.
- Each full normalized `requested_product_type` scope receives one total repair,
  keyed by the server with distinct advertised siblings kept separate. A schema
  correction or a fresh constraint-provenance review may consume it; constraint
  feedback from an in-flight schema repair closes for synthesis. The isolated
  step uses the concise, schema-generic repair-only system prompt, forces only
  `search_catalog_tool`, and disables parallel calls. Its only messages are the
  current shopper turn and bounded, sanitized validator feedback in a separate
  Human data message. Echoed rejected arguments are stripped; invalid AI/tool
  history and the base runtime prompt are absent; the complete active
  shopper-skill instructions remain. A successful partial
  search may continue to another valid role and its own one-repair opportunity,
  and a second invalid call in the same scope and every `STOP_TOOL_USE` result
  close the loop. For native transport failures, a repair may change an
  ungrounded model-generated scope, but not one grounded in current or recent
  shopper text. Independently valid finite locks are restored before execution
  and recorded by name in bounded `restored_fields` diagnostics; a grounded
  free-form scope that cannot be reconstructed safely remains protected by
  `repair_scope_changed`. On a native schema-invalid call, malformed or nonempty
  free-form `unadvertised_requirements` arguments close without repair. Successful
  search-only turns render deterministically; mixed-tool turns synthesize from
  collected evidence.
- Taxonomy and required-constraint schemas contain only capability-derived
  advertised fields and values, plus the explicit
  `unadvertised_requirements` lane.
- Every search schema includes product-agnostic `shopper_guidance` authored
  before retrieval under the active skill. It must be nonempty except for
  `image_only` and `no_direct_catalog_match`, which require empty guidance, and
  it cannot contain candidate facts or internal search mechanics.
- Every text search requires `requested_product_type` product noun/umbrella
  provenance: the shortest product noun or true umbrella from the current turn
  or direct antecedent, excluding color, material, fit, occasion, weather, and
  style modifiers. For `agent_selected_type`, runtime derives it from the one
  advertised subcategory selected by the model; image-only search requires it to
  be `null`. Literal suffix binding can recover an advertised type from a
  modifier phrase but is disabled for explicit alternatives containing `and`,
  `or`, `/`, or `&`, leaving their umbrella/alternative interpretation
  model-owned. For one unambiguous current-turn literal pair whose members are
  exact advertised subcategories of the same category, runtime requires both
  model-authored branches under `member_of_requested_umbrella` and executes the
  pair once. It widens the candidate window and preserves ranking while keeping
  one returned candidate per branch when available. Modified, synonymous,
  ambiguous, and cross-category alternatives remain model-owned.
- Each search has at most one category. `agent_selected_type` selects exactly
  one advertised subcategory as a focused starting role only when the
  shopper named no type for that role; alternatives, confirmations,
  comparisons, and follow-ups count as named types. Invalid open-role
  provenance is rejected rather than silently reinterpreted; runtime retains
  `agent_selected_type` and derives its requested type from that subcategory.
- `no_direct_catalog_match` uses empty taxonomy and no hard constraints; it and
  an unenforceable direct must-have perform no catalog retrieval. Unsupported
  modifiers do not erase an advertised type, and subjective style stays
  semantic.
- Duplicate search identity is normalized taxonomy plus hard constraints;
  changing only semantic wording cannot repeat a retrieval.
- A nonempty unadvertised-requirement lane on a shopper-stated product scope
  fails closed even when the model uses a synonym rather than the shopper's
  exact wording. The bounded constraint review is reserved for a proposed
  inferred requirement on a genuinely open `agent_selected_type` role when the
  scope's shared repair remains; it copies the shopper's shortest
  exact wording or removes the inferred value while freezing requested type,
  taxonomy status, taxonomy, completion state, `search_mode`, and all advertised
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
- Completed successful search-only output uses pre-retrieval `shopper_guidance`
  authored under the active skill, with static `response_guidance` as fallback
  and no final-synthesis model call. Before that guidance becomes evidence, the
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
  successful groups across no-direct or unsupported mixed outcomes. Fixed
  canned responses apply only when that rejection is the sole current-turn
  business-tool outcome. Scoped zero-result
  evidence cannot support a different-type or catalog-wide absence claim.
- Final-response extraction cannot return tool, tool-calling, or internal
  activation text. An otherwise empty answer returns a safe fallback and records
  `incomplete_agent_response`.
- Diagnostics preserve selected skill paths, ordered arguments, rejected and
  duplicate calls, termination reason, bounded product evidence and truncation,
  bounded no-product catalog scope outcomes, and bounded partial graph messages
  on failure. The Judge receives only generated conversation history, product
  evidence/truncation, and catalog scope outcomes from those diagnostics.

## Resolved Decisions

- `CHECKPOINT_STORE=memory` is the only currently supported checkpoint
  configuration. Non-memory values fail during startup rather than silently
  falling back to process-local state.
- Registered skills use `/shopper/<skill>/SKILL.md` under the virtual backend;
  shared read-only references may live directly under `/shopper`.

## Open Decisions

- Which Apache-2.0/MIT-compatible backend should provide shared, durable Deep
  Agents checkpoints for production replicas?
- Should anonymous sessions use cookies, headers, or both?
- What is the cart TTL for anonymous and logged-in users?
- What retention and cleanup policy should apply to cart quantity idempotency
  records?
- Which typed persona fields and trusted profile source should a future
  authenticated integration support?
- Which tools require human approval or confirmation before mutation?
