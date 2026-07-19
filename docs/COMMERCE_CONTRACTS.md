# Commerce Contracts

This document defines the first internal contract layer for commerce tools. It
is intentionally independent of ACP, UCP, and any future protocol plugin. Those
protocols should map into these app-owned contracts through thin adapters later.

## Goals

- Give agents, tools, tests, and future protocol adapters one consistent shape
  for products, carts, and tool results.
- Move commerce behavior toward deterministic tools instead of agent-specific
  prose parsing.
- Preserve the current runtime behavior while creating a safe path toward a
  Deep Agents runtime.

## Purpose Of This PR

This PR establishes the internal commerce language and starts routing current
runtime behavior through it. It gives product search and cart operations a small,
reviewed target for product identity, cart identity, tool metadata, and
structured errors while preserving the existing public API.

The main design decision is separation:

- Product/catalog search is read-only and stateless.
- Cart operations are stateful and mutating.
- Store-policy lookup is read-only and controlled.
- ACP/UCP mappings are future adapter concerns, not core model fields.

## Current Phase

The stacked commerce-tool work now has runtime wiring through the Deep Agents
SDK adapter:

- `DeepAgentsRuntime` registers nine request-scoped tools: catalog search,
  product details, cart read, cart total, cart add, cart remove, cart quantity
  update, store-policy lookup, and product availability. These use the internal
  commerce adapters plus deterministic cart-total calculation.
- Deep Agents cart mutation tools use explicit refs: `PRODUCT_REF` values
  returned by catalog search for add operations, and `CART_LINE_ID` values
  returned by cart reads for remove operations. They do not perform hidden
  product-name lookup or fuzzy cart-line matching.
- Deep Agents product details lookup uses explicit `PRODUCT_REF` values from a
  prior catalog search in the same conversation, then reads the active catalog
  through `GET /products/{product_id}`. It is not a second broad search path.
  Authorization of that ref is a bounded process-local cache, not durable
  checkpoint state.
- All wrapper tools return results to the agent loop so compound discovery,
  policy, availability, and cart requests can finish before the final
  shopper-facing response.
- The legacy `RetrieverAgent` and `CartAgent` files still exist in the repo for
  reference and tests, but they are not the chain-server entrypoint.
- Catalog search remains stateless: no user, cart, memory, session, or
  conversation-history fields are passed to `search_catalog`.
- Catalog request-building is now separate from catalog execution. The
  request-builder layer validates structured agent intent against catalog-owned
  capabilities and produces a `CatalogSearchPlan`; the catalog execution layer
  only maps that plan to catalog service requests.
- Structured agent intent has three required parts: one `semantic_query`, a
  capability-derived `taxonomy` envelope, and `required_constraints`. The chain
  maps generic taxonomy roles to advertised field names, validates scope
  consistency and other must-haves, and produces catalog hard filters. Each call
  accepts at most one category. For a broad request that names no type,
  `agent_selected_type` may include the advertised subcategories that serve one
  focused semantic role. Duplicate identity is normalized taxonomy plus hard
  constraints, so semantic paraphrases do not fan out while genuinely different
  hard-filter scopes can run within the per-turn cap.
- Deep Agents prompt context is also built from catalog-owned capabilities.
  Chain-server no longer ships a product category allowlist for the active
  runtime; changing catalog shape is handled by the JSONL role sidecar plus the
  ingested catalog data.
- Catalog retriever uses source product IDs and covers the complete current
  snapshot by default. Query responses include structured `products`,
  diagnostics, and an optional `no_result_reason` in addition to the legacy
  parallel arrays.
- Catalog search timeout is configurable through
  `catalog_search_timeout_seconds`. The default is `null`, preserving the
  previous no-timeout catalog POST behavior for slower remote embedding calls.
- Cart tools are stateful and adapt the current memory service API without
  changing the public service schema.
- Cart quantity update reads the current cart, removes the full matching line,
  and adds the requested positive quantity back because the memory service has
  no dedicated update endpoint. Quantity `0` stops after the full-line remove.
- Store policy is loaded from an operator-managed static YAML file and cached
  for the process lifetime. Product availability is a deliberate no-I/O stub
  that always reports `unknown` until a live inventory service exists.

The runtime Deep Agents tool names, risk classes, skill access boundaries, and
registered-vs-planned status are tracked separately in
[Shopper Agent Tool Registry](SHOPPER_AGENT_TOOL_REGISTRY.md). This document
defines the shared contract layer; the registry defines what the
shopper-serving Deep Agent can actually call today.

No ACP/UCP adapter layer has been added yet.

## Core Models

| Model | Purpose |
| --- | --- |
| `Availability` | Controlled stock signal with an explicit `unknown` state. |
| `Money` | Currency amount with a default `USD` currency. |
| `ProductSummary` | Search-result-safe product display fields and current product identifier. |
| `ProductDetail` | Full product shape with variants and optional source URI. |
| `ProductVariant` | Variant identity, options, price, and availability. |
| `StorePolicy` | Controlled store-policy content such as returns or shipping. |
| `CartLine` | Cart row keyed by `cart_line_id`, `product_id`, and optional `variant_id`. |
| `Cart` | User cart with structured lines and optional subtotal. |
| `CommerceError` | Structured tool error with code, message, retryability, and details. |
| `ToolMeta` | Trace and idempotency metadata returned by tools. |
| `CatalogCapabilities` | Catalog-owned field roles, observed values/ranges, nested taxonomy scopes, and retrieval modes. |

## Tool Contracts

The first tool contract set is:

| Contract | Type | Purpose |
| --- | --- | --- |
| `SearchCatalogInput` / `SearchCatalogResult` | Read-only | Find products by query, category, filters, and `top_k`. |
| `GetProductDetailsInput` / `GetProductDetailsResult` | Read-only | Fetch deterministic detail fields for one known product ref from the active catalog snapshot. |
| `CheckProductAvailabilityInput` / `CheckProductAvailabilityResult` | Read-only | Return the explicit `unknown` availability boundary for a known product ref and optional variant hint. |
| `GetCartInput` / `GetCartResult` | Read-only | Read the authoritative cart for a user. |
| `GetStorePolicyInput` / `GetStorePolicyResult` | Read-only | Fetch controlled store-policy text by topic. |
| `AddCartItemInput` / `CartMutationResult` | Mutating | Add a product or variant to the cart from an explicit product ref. |
| `UpdateCartItemInput` / `CartMutationResult` | Mutating | Change cart-line quantity. Quantity `0` means remove. |
| `RemoveCartItemInput` / `CartMutationResult` | Mutating | Remove an explicit cart line by `cart_line_id`. |

Mutating inputs require `idempotency_key` so future agent retries and protocol
adapters have a stable key to enforce safe retries. In the current memory
service adapter, the key is echoed in tool metadata but is not stored or used to
deduplicate mutations yet.

### Stateless Catalog Search

`SearchCatalogInput` intentionally has no `user_id`, cart, memory, session, or
conversation-history fields. The agent layer uses conversation context to
produce one `semantic_query`, a capability-derived `taxonomy` envelope, and
structured `required_constraints`. The chain maps taxonomy roles to the actual
advertised field names, checks every required field and value, refuses requests
that cannot be enforced, and sends `queries=[semantic_query]` plus the validated
hard filters. `search_catalog` itself remains a pure read.

An explicitly requested concrete type with no faithful advertised value uses
`no_direct_catalog_match`: both taxonomy arrays and all hard constraints are
empty, and no retrieval occurs. That decision is based on product type alone;
an unsupported modifier does not erase an advertised type. Unsupported direct
must-haves use `unadvertised_requirements`, while subjective style and other soft
preferences remain in `semantic_query`.

The catalog makes no chat/completion call and performs no shopper-language
interpretation or query expansion. It generates the configured text/image
embeddings, performs vector retrieval and candidate fusion, deduplicates by
source product ID, applies hard filters and thresholds, and sorts results
deterministically. The lower-level `queries` list remains for direct/internal
compatibility, but the
serving agent sends one entry and bounds distinct taxonomy scopes per turn.

Every successful search tool result carries `SEARCH_DIRECTION_EVIDENCE`, the
model-authored `semantic_query` used as a catalog ranking preference. It is not
a confirmed product attribute. Search-only styling responses are assembled
deterministically from that direction, returned candidate facts, and confirmed
filters. They explicitly label the direction as preference and nominate the
first ranked result, or one first result per requested role, without invoking a
separate rationale model. Tool-loop repair is also bounded: one invalid search
may receive one search-only repair; a successful repaired partial scope may
continue to another valid role, but no second repair is allowed.

The chain-server request-builder consumes `CatalogCapabilities` before it
creates a product search request. Authoritative field roles come from the
catalog sidecar. Enum/list values, numeric ranges, coverage, and nested
category/subcategory scopes come from the active JSONL rows. New values or
categories therefore need no chain-server change. The operational guide is
[Catalog Schema and Filters](CATALOG_FILTERS.md).

The active Deep Agents runtime caches the first successfully fetched
`CatalogCapabilities` for the chain-server process lifetime. Deterministic
request validation uses that full object; the agent system prompt receives
only a compact projection of filter names, types, values or ranges, scoped
field applicability, and semantic/filter/detail roles. This keeps the language
layer aware of available hard filters without sending counts, coverage, or
repeated scoped values on every turn and without maintaining a second category
list in `shared/configs/chain_server/config.yaml`. After a catalog replacement,
restart the chain server once the catalog is healthy so all layers consume the
new catalog-owned shape.

This keeps product search reusable by the Deep Agents adapter, future skills or
subagents, and later protocol adapters without coupling catalog results to
shopper session state.

Catalog search maps the source `record_id` into `ProductSummary.product_id`;
Milvus primary keys and product names are never commerce identities. The
current feed's generated IDs are only guaranteed within the active catalog
snapshot. Detail reads and cart adds verify refs against that snapshot and
require a fresh search when a ref is stale. The runtime remembers at most 50
refs per conversation in process memory. Redis checkpointing does not persist
this cache, so a restart, another replica, eviction, or catalog replacement also
requires a fresh search.

### Policy And Availability Boundaries

`get_store_policy` reads only controlled content for the six supported topics:
returns, shipping, sizing, payment, price matching, and gift cards. A missing
file or topic produces a structured error rather than a model-authored policy.
The bundled YAML contains operator placeholders that must be replaced before
production.

`check_product_availability` makes no catalog or inventory call. It always
returns `availability="unknown"` with a consistent shopper-safe message. This
contract prevents catalog presence from being mistaken for live stock while a
real inventory and variant service remains out of scope.

### Diagnostic Boundaries

`SearchCatalogResult.diagnostics` describes deterministic catalog retrieval
work. The chain server's separate `agent_diagnostics` field describes one Deep
Agents turn: activated/injected skill paths, ordered tool calls,
rejection/duplicate outcomes, termination, and bounded partial graph messages
after failure. Final shopper-text extraction excludes tool messages,
tool-calling assistant messages, and internal activation markers. If none
remains, the runtime returns a safe retry response and records
`incomplete_agent_response`. These runtime behaviors do not change any shared
commerce request or result model.

## Remaining Direction

The active Deep Agents runtime now uses these wrappers. Remaining commerce
identity work is to obtain an upstream ID guarantee and persist source product
IDs in the cart service instead of relying on display names for stored lines.
Until then, `CART_LINE_ID` is a display-name alias and positive quantity updates
are a non-atomic remove-then-add operation.
Legacy `RetrieverAgent` and `CartAgent` code remains for compatibility tests but
is not the serving entrypoint.

The important rule is that ACP/UCP compatibility should be added as adapters
around these contracts, not as fields or naming choices inside the core models.
