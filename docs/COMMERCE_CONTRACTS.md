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

- `DeepAgentsRuntime` registers six request-scoped tools: catalog search,
  product details, cart read, cart total, cart add, and cart remove. These use
  the internal `search_catalog`, `get_product_details`, `get_cart`,
  `add_cart_item`, and `remove_cart_item` adapters plus deterministic cart-total
  calculation.
- Deep Agents cart mutation tools use explicit refs: `PRODUCT_REF` values
  returned by catalog search for add operations, and `CART_LINE_ID` values
  returned by cart reads for remove operations. They do not perform hidden
  product-name lookup or fuzzy cart-line matching.
- Deep Agents product details lookup uses explicit `PRODUCT_REF` values from a
  prior catalog search in the same conversation, then reads the active catalog
  through `GET /products/{product_id}`. It is not a second broad search path.
- Catalog search and cart-read wrapper tools return results to the agent loop
  so explicit cart-mutation requests can search/read and then mutate in one
  turn. Mutation tools still return the authoritative cart result directly.
- The legacy `RetrieverAgent` and `CartAgent` files still exist in the repo for
  reference and tests, but they are not the chain-server entrypoint.
- Catalog search remains stateless: no user, cart, memory, session, or
  conversation-history fields are passed to `search_catalog`.
- Catalog request-building is now separate from catalog execution. The
  request-builder layer validates structured agent intent against catalog-owned
  capabilities and produces a `CatalogSearchPlan`; the catalog execution layer
  only maps that plan to catalog service requests.
- Structured agent intent separates soft/descriptive semantic text from
  `required_constraints`. The request builder validates every must-have against
  current capabilities, refuses unsupported requirements, and converts only
  supported entries into catalog hard filters.
- Deep Agents prompt context is also built from catalog-owned capabilities.
  Chain-server no longer ships a product category allowlist for the active
  runtime; changing catalog shape is handled by the JSONL role sidecar plus the
  ingested catalog data.
- Catalog retriever uses source product IDs, covers the complete current
  snapshot by default, applies hard filters against structured metadata, then
  trims to the requested `top_k`. Query
  responses include structured `products`, diagnostics, and an optional
  `no_result_reason` in addition to the legacy parallel arrays.
- Catalog search timeout is configurable through
  `catalog_search_timeout_seconds`. The default is `null`, preserving the
  previous no-timeout catalog POST behavior for slower remote embedding calls.
- Cart tools are stateful and adapt the current memory service API without
  changing the public service schema.

The runtime Deep Agents tool names, risk classes, skill access boundaries, and
registered-vs-planned status are tracked separately in
[Shopper Agent Tool Registry](SHOPPER_AGENT_TOOL_REGISTRY.md). This document
defines the shared contract layer; the registry defines what the
shopper-serving Deep Agent can actually call today.

No ACP/UCP adapter layer has been added yet.

## Core Models

| Model | Purpose |
| --- | --- |
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
conversation-history fields. The agent layer can use conversation context to
produce soft/descriptive semantic text plus structured `required_constraints`.
The chain-server request builder checks every required field and value against
the active capabilities, refuses the request if any must-have cannot be
enforced, and maps the validated subset to `SearchCatalogInput.filters`.
`search_catalog` itself remains a pure read against the catalog for that
validated request.

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
require a fresh search when a ref is stale.

## Remaining Direction

The active Deep Agents runtime now uses these wrappers. Remaining commerce
identity work is to obtain an upstream ID guarantee and persist source product
IDs in the cart service instead of relying on display names for stored lines.
Legacy `RetrieverAgent` and `CartAgent` code remains for compatibility tests but
is not the serving entrypoint.

The important rule is that ACP/UCP compatibility should be added as adapters
around these contracts, not as fields or naming choices inside the core models.
