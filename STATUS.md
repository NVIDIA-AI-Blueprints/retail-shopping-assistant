# Project Status

Updated: 2026-07-19

## Current Milestone

The current working tree extends the shopper-serving Deep Agent architecture:

- conversation checkpointing is configurable through `CHECKPOINT_STORE`, with
  in-process memory as the development/test default and Redis as the production
  shared store. The conversation-scoped product-ref cache remains process-local
  and is not made durable by Redis;
- five shopper skills are registered for product discovery, outfit styling,
  cart management, budget shopping, and controlled store-policy answers. A
  required first model step selects the smallest applicable set each turn; the
  runtime then injects the complete selected files before exposing shopping
  tools. Selection metadata is regenerated from current files rather than
  retained in the conversation checkpoint;
- the runtime has nine shopper-facing tool boundaries plus one internal skill
  activation control tool. The new shopping tools cover cart quantity update,
  controlled policy lookup, and an honest availability stub;
- `/query/stream` and `/query/timing` accept optional caller-supplied read-only
  persona context without changing existing requests or adding another model
  call. This context is advisory and untrusted; production callers must
  authenticate its owner and allowlist fields upstream; and
- both response paths expose additive agent diagnostics for activated skill
  files, ordered tool calls and arguments, rejected/duplicate calls, and final
  termination. Failed graph messages are captured from the current checkpoint
  before cleanup deletes it. Pre-activation and same-batch shopping calls are
  execution-blocked and reported as `skill_activation_required`.

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
- every agent catalog call requires one `semantic_query`, a `taxonomy` envelope,
  and `required_constraints`. The allowed `taxonomy.category` and
  `taxonomy.subcategory` values are generated from the cached catalog
  capabilities rather than application taxonomy. Each call accepts at most one
  category. `agent_selected_type` may include the advertised subcategories that
  serve one focused semantic role. Executable text search requires at least one
  taxonomy value; image-only and no-direct no-retrieval requests use empty
  arrays;
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
  paraphrased; a genuinely different hard-constraint scope may run within
  `max_catalog_searches_per_turn`;
- search-schema recovery is limited to one model repair. A successful repaired
  partial search may continue with another valid role; a second invalid call,
  completed scope, unsupported requirement, or deterministic stop closes the
  tool loop. Successful search evidence preserves the model-authored semantic
  query as ranking direction. Deterministic styling responses label it as a
  preference, never a product fact, and nominate the first ranked result for
  each requested role without a rationale model call;
- final-response extraction excludes tool messages, assistant tool-call
  messages, and internal activation markers. A completed graph with no
  shopper-facing answer returns a safe retry response and records
  `incomplete_agent_response`.

The catalog service performs embeddings, filtering, and deterministic retrieval
only. It makes no chat/completion call or shopper-language interpretation.

The concise serving map is
[Shopper Agent Architecture](docs/SHOPPER_AGENT_ARCHITECTURE.md); catalog detail
lives in [Schema-Driven Catalog Architecture](docs/CATALOG_REFACTOR_PLAN.md).

## Verification

- Full offline unit suite: 744 passed with one pre-existing
  `StarletteDeprecationWarning` in 5.13 seconds.
- Focused skill-activation coverage includes a compiled Deep Agents loop and
  verifies forced activation-only binding, fresh registry descriptions,
  complete selected-file injection, all-nine-tool exposure only afterward,
  current-request isolation, same-batch execution rejection, and fail-closed
  behavior.
- Agent observability coverage verifies current-turn skill activation, model-issued
  tool-call order and arguments, rejection/duplicate classification, pending
  calls, additive API/SSE propagation, and snapshot-before-delete failure
  handling.
- Focused shopper-skill registry tests: 9 passed.
- Focused commerce contract and adapter tests: 30 passed.
- Current capability API response: 147,792 bytes; compact LLM projection:
  5,769 bytes, down from 12,962 bytes before this change while retaining
  taxonomy keys, every current enum value, and semantic/filter roles.
- UI production build: compiled successfully.
- UI lint: 0 errors and 3 pre-existing warnings.
- Docker Compose configuration: valid.
- Chain-server Docker image: built successfully with the Redis checkpointer
  dependency installed under Python 3.11.
- The latest live shopping evaluation and timing comparison are preserved in
  the canonical local quality archive rather than versioned as source.
- The exact current-tree run,
  `incoming_architecture_commit_ready_final_20260719`, completed all 48 shopper
  turns with 48 Judge results and 48 timing records. Judge average was
  3.4166667/5; distribution was 2: 8, 3: 13, 4: 26, and 5: 1. Twenty-seven of
  48 turns (56.25%) scored at least 4.
- Mean / median / p95 / maximum latency was 7.513s / 5.557s / 26.334s /
  30.014s. All 48 turns included agent diagnostics. No request or Judge failure
  was observed, and a result-content audit found no
  `SHOPPER_SKILL_ACTIVATION_*` marker in either validated result copy.
- Against the prior WIP, Judge average improved from 3.2917 to 3.4166667 and
  score-4-or-better coverage improved from 25/48 to 27/48. Mean and maximum
  latency improved from 9.128s and 108.693s to 7.513s and 30.014s; median and
  p95 moved from 5.258s and 25.195s to 5.557s and 26.334s.
- Against `previous_committed` / Staging, Judge average improved from 3.2083 to
  3.4166667 and score-4-or-better coverage from 23/48 to 27/48. Mean / median /
  p95 / maximum latency improved from 14.922s / 8.863s / 57.367s / 94.609s to
  7.513s / 5.557s / 26.334s / 30.014s.

The Staging comparison remains qualified because the catalog treatment changed;
the prior-WIP comparison is directly comparable. Catalog-sensitive Golden drift
still applies: existing beige skirts and pastel tops contradict older absence
answers, so raw scores are not a standalone catalog-quality verdict.

The judged run covers the final deterministic search-direction renderer and
final-response extraction safeguard.

## Remaining Quality Risk

The new store-policy YAML contains explicit operator placeholders and is not
production policy until an operator replaces them. Redis checkpointing also
depends on an external Redis 8+ or Redis Stack service and has not been treated
as part of the bundled Compose stack. The memory service still identifies cart
lines by display name, so `CART_LINE_ID` is an alias and a positive quantity
update is a non-atomic remove-then-add operation. Product availability remains
`unknown` until a live inventory/variant service is integrated.

Same-conversation `PRODUCT_REF` evidence is held in a bounded process-local
cache and is valid only for the active catalog snapshot. It does not move with a
Redis checkpoint. A restart, another replica, cache eviction, or catalog
replacement therefore requires a fresh search before details or cart adds.
Caller-supplied persona fields are also advisory rather than authenticated
profile truth; production integrations must authenticate ownership and enforce
an upstream field allowlist.

Mandatory activation adds one bounded app-model step to every Deep Agents turn.
It guarantees that shopping cannot run before a complete skill file is loaded;
the semantic choice among registered skills remains model-selected and is now
explicit in `agent_diagnostics.skill_files_read`.

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

Eight turns still scored 2. The remaining response-quality gaps are concentrated
in two honest categories: safe but terse no-match/unenforceable answers where
older Goldens prefer speculative adjacent products, and weak styling usefulness
for subjective requests such as formal, bold, sporty, or starting an outfit from
one anchor. Those cases need better evidence-aware conversation and styling
behavior, not weaker catalog or availability boundaries.

The raw Judge score should not be described as an unqualified catalog-quality
gain until catalog-dependent Goldens are reconciled with the active inventory.
