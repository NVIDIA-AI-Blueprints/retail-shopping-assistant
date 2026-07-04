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

- `DeepAgentsRuntime` exposes request-scoped wrapper tools over the internal
  `search_catalog`, `get_cart`, `add_cart_item`, and `remove_cart_item`
  wrappers.
- Deep Agents cart mutation tools use explicit refs: `PRODUCT_REF` values
  returned by catalog search for add operations, and `CART_LINE_ID` values
  returned by cart reads for remove operations. They do not perform hidden
  product-name lookup or fuzzy cart-line matching.
- Catalog search and cart-read wrapper tools return results to the agent loop
  so explicit cart-mutation requests can search/read and then mutate in one
  turn. Mutation tools still return the authoritative cart result directly.
- The legacy `RetrieverAgent` and `CartAgent` files still exist in the repo for
  reference and tests, but they are not the chain-server entrypoint.
- Catalog search remains stateless: no user, cart, memory, session, or
  conversation-history fields are passed to `search_catalog`.
- Catalog search timeout is configurable through
  `catalog_search_timeout_seconds`. The default is `null`, preserving the
  previous no-timeout catalog POST behavior for slower remote embedding calls.
- Cart tools are stateful and adapt the current memory service API without
  changing the public service schema.

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

## Tool Contracts

The first tool contract set is:

| Contract | Type | Purpose |
| --- | --- | --- |
| `SearchCatalogInput` / `SearchCatalogResult` | Read-only | Find products by query, category, filters, and `top_k`. |
| `GetProductDetailsInput` / `GetProductDetailsResult` | Read-only | Reserved for fetching one product by durable `product_id` once the catalog exposes one. |
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
decide what query or query terms, optional image, categories, and filters to
send, but `search_catalog` itself should remain a pure read against the catalog
for the supplied request.

This keeps product search reusable by the Deep Agents adapter, future skills or
subagents, and later protocol adapters without coupling catalog results to
shopper session state.

Current catalog search results map the retriever's returned `ids` field into
`ProductSummary.product_id`. Those IDs are Milvus retriever primary keys and can
change after reindexing. Treat them as transient catalog result IDs until the
catalog import provides a durable product ID or SKU.

## Migration Direction

The migration wraps existing behavior behind these contracts without changing
the public API first:

1. `RetrieverAgent` calls an internal `search_catalog` tool wrapper.
2. `CartAgent` calls internal cart tool wrappers.
3. Treat current catalog result IDs as transient `ProductSummary` and mutation
   result fields only; durable ID persistence in chain-server state and cart
   memory rows is future work.
4. Deep Agents SDK wrapper tools now reuse the same internal commerce wrappers
   and require explicit product/cart-line refs for cart mutations.
5. Add Deep Agents skills and subagents on top of the same tool layer.

The important rule is that ACP/UCP compatibility should be added as adapters
around these contracts, not as fields or naming choices inside the core models.
