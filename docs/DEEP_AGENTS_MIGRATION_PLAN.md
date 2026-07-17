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

- Existing commerce tool implementations are not modified in this slice.
- The Deep Agents adapter exposes thin request-scoped wrapper tools over the
  existing commerce functions.
- The first shopper-facing skill is loaded from
  `chain_server/skills/shopper/outfit-styling/SKILL.md` through the Deep
  Agents SDK skills interface.
- The public request body remains backward compatible.
- Optional `session_id`, `conversation_id`, and `cart_id` fields are accepted,
  and the bundled UI now sends browser-session identifiers on every turn.
  Legacy callers that send only `user_id` are still mapped to deterministic
  compatibility identifiers.
- The compatibility mapping is not the final production identity design. A
  high-scale website should move to server-created session, conversation, and
  cart identifiers before broad rollout.
- The runtime caps Deep Agents recursion and distinct catalog taxonomy scopes
  per turn. The agent tool accepts one semantic query, and a normalized scope
  can execute only once even if later wording paraphrases it. This prevents
  exploratory same-scope loops while allowing bounded searches for different
  outfit components.
- Catalog search and cart-read tools return to the Deep Agents loop so a single
  shopper turn can discover products or read the cart before mutating it.
  Mutating cart tools return directly with the authoritative cart result.
- Optional VLM media perception runs before the Deep Agents turn when the
  `vlm` model role is enabled. It converts attached image/video media into a
  concise `MEDIA ANALYSIS` text block. Raw media is not persisted in
  conversation memory. Descriptive look-analysis requests are answered from
  `MEDIA ANALYSIS` without catalog retrieval; catalog tools remain authoritative
  for product names, prices, and availability when the shopper explicitly asks
  to find, compare, price-check, or add products.

Filesystem and built-in Deep Agents tools:

- Deep Agents includes filesystem, todo, shell, and subagent tools by default.
- This shopping runtime registers a harness profile that excludes built-in
  filesystem write/edit/list/search tools, todo tools, shell tools, and the
  default general-purpose subagent.
- Built-in `read_file` remains available only so Deep Agents can read static
  skill files from a virtual-mode filesystem backend rooted at
  `chain_server/skills`.
- Customer profile, cart, price, inventory, order, and payment truth must not
  live in local files or the Deep Agents virtual filesystem.
- If future skills use filesystem-backed instructions, those files must be
  static application assets or store-backed per-conversation state, never a
  shared writable customer namespace.

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
5. The Deep Agent may choose tools and skills, but deterministic tools own
   validation, state mutation, idempotency, and authorization.
6. Persona data is loaded as a read-only snapshot for the turn unless a later
   feature explicitly supports profile updates.
7. Carts are scoped by `cart_id`, not by conversation memory.
8. Conversation memory is scoped by `conversation_id`, not by customer alone.
9. Skills describe shopping behavior and domain knowledge. Tools perform
   reads, writes, and external service calls.

## Target Request Flow

```text
POST /query/stream
  -> resolve or create server-owned identity
  -> load session, conversation, cart, and optional persona snapshot
  -> invoke Deep Agents SDK with thread_id = conversation_id
  -> Deep Agent selects skills and calls deterministic shopping tools
  -> tools call catalog, memory, cart, policy, or persona services
  -> stream assistant response back to the UI
  -> persist conversation summary and tool results under the correct scope
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

## Session Isolation Requirements

- A request without a valid server session gets a new `session_id`.
- A request without a valid conversation gets a new `conversation_id`.
- Deep Agents checkpointing, memory, and filesystem state use
  `conversation_id` as the thread key.
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
2. Explicit session preferences.
3. Loaded customer persona/profile snapshot.
4. Store policy and catalog facts.
5. Model assumptions are not allowed as facts.

The target app should load one persona snapshot near the start of a turn and
pass that snapshot to the Deep Agent. Individual skills should not independently
fetch mutable persona state, because that creates inconsistent context and
hard-to-debug race conditions.

## Tools

Initial tools should be small, typed, and deterministic:

- `search_catalog`: read-only, stateless product discovery.
- `get_product_details`: read-only product facts for one product.
- `get_cart`: read-only cart state for a `cart_id`.
- `add_cart_item`: mutating cart write with idempotency, using a `PRODUCT_REF`
  previously returned by product search.
- `remove_cart_item`: mutating cart write with idempotency, using a
  `CART_LINE_ID` returned by cart reads.
- `view_cart_total`: deterministic arithmetic over cart line prices.
- `get_store_policy`: read-only controlled policy content.
- `load_customer_persona`: read-only persona/profile snapshot.

The active shopper-serving Deep Agents tool registry is documented in
[Shopper Agent Tool Registry](SHOPPER_AGENT_TOOL_REGISTRY.md). The active skill
registry is documented in
[Shopper Agent Skill Registry](SHOPPER_AGENT_SKILL_REGISTRY.md). Those
registries are the source for which tools and skills are actually registered
today, their boundaries, and current limitations. Planned contracts in this
section should not be treated as agent-callable until they appear in the
registries as registered runtime capabilities.

The Deep Agent decides when to call a tool. The tool decides whether the call is
valid and what state changes are allowed.

## Skills

Initial skills should be domain instructions and examples, not hidden business
logic. The first registered skill is:

- `outfit-styling`: fashion styling across anchor product, no-anchor
  discovery, cart styling, conversational mid-browse, visual, and budget-aware
  styling requests.

Skills may guide tool use, explain constraints, and provide examples. They
should not own cart mutation, pricing, inventory, profile persistence, or order
creation.

Planned future skills remain product discovery, visual shopping,
cart-management, budget-sensitive recommendation, persona-aware recommendation,
and store-policy answers. Keep them as markdown-guided behavior until
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
- Deep Agents checkpointing must use a production store keyed by
  `conversation_id`.
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
  `cart_id`, tool name, latency, and error class.
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

- Expose current catalog and cart capabilities as typed tools.
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

- Add and iterate initial shopping skills. `outfit-styling` is the first
  registered skill and remains a markdown-guided behavior surface.
- Keep skills as guidance. Keep state changes in tools.

### Slice 5: Deep Agents Runtime Parity

- Run existing unit tests.
- Run targeted Challenger scenarios.
- Verify product search, image search, cart mutation claims, and grounded
  responses.
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

## Open Decisions

- Which production store should back Deep Agents checkpointing and memory?
- Should anonymous sessions use cookies, headers, or both?
- What is the cart TTL for anonymous and logged-in users?
- Which persona fields are allowed into the model context?
- Which tools require human approval or confirmation before mutation?
- What is the first Deep Agents skill directory structure?
