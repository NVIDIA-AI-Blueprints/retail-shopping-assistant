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
  tool. Shopping calls issued before activation or in the same model batch are
  rejected at execution time.
- The public request body remains backward compatible and accepts an optional
  caller-supplied read-only `persona` snapshot.
- Optional `session_id`, `conversation_id`, and `cart_id` fields are accepted,
  and the bundled UI now sends browser-session identifiers on every turn.
  Legacy callers that send only `user_id` are still mapped to deterministic
  compatibility identifiers.
- The compatibility mapping is not the final production identity design. A
  high-scale website should move to server-created session, conversation, and
  cart identifiers before broad rollout.
- The runtime caps Deep Agents recursion and distinct catalog taxonomy scopes
  per turn. The agent tool accepts one semantic query. Duplicate identity is
  normalized taxonomy plus hard constraints, so paraphrasing cannot repeat the
  same retrieval while a genuinely different hard-filter scope may execute
  within the cap.
- Cached catalog capabilities generate both the exact taxonomy enum and the
  non-taxonomy required-constraint properties exposed to the model. The agent
  owns semantic selection from that schema; deterministic code validates and
  maps the structured selection without interpreting shopper language. One call
  covers at most one category. `agent_selected_type` may include every
  advertised subcategory that serves one focused semantic role when a broad
  request names no concrete type.
- `no_direct_catalog_match` stops before retrieval when an explicitly requested
  concrete type has no faithful advertised value; it uses empty taxonomy and no
  hard constraints. An unsupported modifier does not erase an advertised type,
  and subjective style remains semantic direction. A directly stated product
  must-have absent from the generated schema is retained in
  `unadvertised_requirements`, so it is reported as unenforceable rather than
  weakened into semantic relevance.
- After the first search-schema validation failure, the runtime exposes one
  search-only repair step. A partial multi-role search with an unadvertised
  requirement uses that same step to review the current shopper context once:
  preserve a directly stated must-have, or remove only an inference from broad
  season, weather, occasion, or style context. A successful repaired partial
  search may continue to another valid role, but a second repair is never
  available. A completed current scope, unsupported requirement, or any
  `STOP_TOOL_USE` result forces answer synthesis.
- Every turn exposes additive agent diagnostics with activated skill-file
  paths, model-issued tool calls and arguments in order, deterministic
  rejection/duplicate markers, and a final termination reason. When a graph
  fails, its current-turn assistant/tool messages are read from the latest
  checkpoint before that checkpoint is deleted.
- Catalog, cart, policy, and availability tools return to the Deep Agents loop
  so a single shopper turn can complete a compound request before the final
  shopper-facing response.
- Grounding uses actual tool-role messages only. Current-turn evidence is
  isolated by the server-owned request marker; prior-turn tool evidence may
  resolve direct references but cannot prove that a new search or mutation ran.
  Every successful search records the model-authored semantic query as
  `SEARCH_DIRECTION_EVIDENCE`. Search-only styling responses expose that text as
  ranking preference, never product fact, and nominate the first ranked result
  or one first result per requested role. Names, prices, categories, and
  confirmed filters remain deterministic, with no separate rationale model
  call.
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
6. Persona data is loaded as a read-only snapshot for the turn unless a later
   feature explicitly supports profile updates.
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
  -> load session, conversation, cart, and optional persona snapshot
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
when present.

Product evidence identity is a separate, narrower bridge. The runtime keeps at
most 50 returned `PRODUCT_REF` values per `conversation_id` in process memory,
and those refs are valid only for the active catalog snapshot. This cache is not
part of the LangGraph checkpoint. Redis therefore preserves conversation state
but does not make product refs portable across replicas or durable across a
restart. Another replica, cache eviction, restart, or catalog replacement
requires a fresh search before product details or cart adds.

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

Callers may pass one optional `persona` object with `/query/stream` or
`/query/timing`. The runtime prepends non-empty values as a read-only turn-start
context block. It does not fetch or mutate the persona and does not expose a
persona-loading tool. Individual skills should not independently fetch mutable
persona state, because that creates inconsistent context and hard-to-debug race
conditions. The runtime treats the object as advisory, caller-provided context;
it does not authenticate ownership or allowlist fields. Production callers must
perform both checks upstream before forwarding persona data.

## Tools

The tool layer is small, typed, and deterministic:

- `search_catalog`: read-only, stateless product discovery.
- `get_product_details`: read-only product facts for one product.
- `get_cart`: read-only cart state for a `cart_id`.
- `add_cart_item`: mutating cart write with idempotency, using a `PRODUCT_REF`
  previously returned by product search.
- `remove_cart_item`: mutating cart write with idempotency, using a
  `CART_LINE_ID` returned by cart reads.
- `update_cart_item`: mutating cart quantity update using a current
  `CART_LINE_ID`; quantity `0` removes the line.
- `view_cart_total`: deterministic arithmetic over cart line prices.
- `get_store_policy`: read-only controlled policy content.
- `check_product_availability`: read-only deliberate stub that reports
  `unknown` until live inventory exists.

`load_customer_persona` remains planned. Persona snapshots are caller-supplied
turn context in the current runtime, not an agent-callable tool.

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

Tool-loop control is a separate deterministic boundary. A first invalid
`search_catalog_tool` schema yields one model repair step with only that tool
available. A successful repaired search may continue when it is explicitly
partial and another valid role remains, but no second repair is available.
After a successful search is marked `scope_complete`, when a requirement or
taxonomy is unenforceable, or whenever a tool returns `STOP_TOOL_USE`, the next
model step has no tools and must synthesize from the evidence already collected.

The search contract has two explicit no-search boundaries. A concrete requested
type with no faithful taxonomy value uses `no_direct_catalog_match` and performs
no retrieval; that status carries empty taxonomy arrays and no hard constraints.
An unsupported modifier does not erase an otherwise advertised type. A directly
stated must-have missing from the capability-derived constraint properties is
carried in `unadvertised_requirements` and returned as unenforceable, while
subjective style stays semantic. On a partial multi-role search, the runtime
permits one contextual review before closure so the model can remove only an
inferred broad-context requirement; a direct product requirement remains
blocked.

Final grounding accepts evidence only from tool-role messages. The current
request is isolated by its request marker and prior-turn evidence is supplied
separately for references to products already shown. For search-only turns, code
renders candidate facts and confirmed filters deterministically. A successful
search's `SEARCH_DIRECTION_EVIDENCE` is the model-owned semantic ranking
preference, not a product fact; styling output labels it accordingly and
nominates the first ranked result for each requested role. No separate rationale
model is called.

## Skills

Skills are domain instructions and examples, not hidden business logic. The
registered set is:

- `product-discovery`: the primary procedure for general search, browse, and
  filtering without styling intent.
- `outfit-styling`: fashion styling across anchor product, no-anchor discovery,
  cart styling, conversational mid-browse, and visual requests; it is the
  primary procedure instead of product discovery when styling is requested.
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
stale for an existing Redis thread after a skill deployment. The filesystem
middleware remains enabled for read-only reference files.

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
- Deep Agents checkpointing uses Redis in production and is keyed by
  `conversation_id`. The in-process store is for development and tests only.
- Redis checkpoint durability does not cover the current process-local product-
  ref cache. Cross-replica product follow-ups require either a fresh search or a
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
  and treat that history as authoritative over counterfactual reference-answer
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
- Activation instructions and skill files require product discovery and outfit
  styling to remain alternative primary procedures and budget shopping to be
  selected only for a stated budget.
- An invalid search schema receives at most one search-only repair step. A
  successful repaired partial search may continue to another valid role, but a
  second invalid call and every `STOP_TOOL_USE` result force answer synthesis.
- Taxonomy and required-constraint schemas contain only capability-derived
  advertised fields and values, plus the explicit
  `unadvertised_requirements` lane.
- Each search has at most one category. `agent_selected_type` may include the
  advertised subcategories that serve one focused semantic role.
- `no_direct_catalog_match` uses empty taxonomy and no hard constraints; it and
  an unenforceable direct must-have perform no catalog retrieval. Unsupported
  modifiers do not erase an advertised type, and subjective style stays
  semantic.
- Duplicate search identity is normalized taxonomy plus hard constraints;
  changing only semantic wording cannot repeat a retrieval.
- A partial multi-role search gets at most one contextual constraint review;
  direct must-haves remain blocked and broad-context inferences may be removed.
- Grounding accepts only tool-role evidence, isolates current-turn evidence by
  request ID, and cannot treat a prior assistant draft as tool evidence.
- Search-only output renders candidate facts and recorded ranking direction
  deterministically, with no separate rationale model step. It labels
  `SEARCH_DIRECTION_EVIDENCE` as model-owned ranking preference, never product
  fact, and nominates the first ranked result for each requested role.
- Final-response extraction cannot return tool, tool-calling, or internal
  activation text. An otherwise empty answer returns a safe fallback and records
  `incomplete_agent_response`.
- Diagnostics preserve selected skill paths, ordered arguments, rejected and
  duplicate calls, termination reason, and bounded partial graph messages on
  failure.

## Resolved Decisions

- Redis is the primary production store for Deep Agents checkpointing;
  in-process memory is the development/test default. Select it with
  `CHECKPOINT_STORE` and configure `CHECKPOINT_REDIS_URL` and
  `CHECKPOINT_TTL_SECONDS`. This does not persist the current process-local,
  catalog-snapshot-bound product-ref cache.
- Registered skills use `/shopper/<skill>/SKILL.md` under the virtual backend;
  shared read-only references may live directly under `/shopper`.

## Open Decisions

- Should anonymous sessions use cookies, headers, or both?
- What is the cart TTL for anonymous and logged-in users?
- Which persona fields should the upstream integration allowlist for each
  authenticated shopper context?
- Which tools require human approval or confirmation before mutation?
