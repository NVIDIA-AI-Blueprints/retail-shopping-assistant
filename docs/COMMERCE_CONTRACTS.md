# Commerce Contracts

This document defines the first internal contract layer for commerce tools. It
is intentionally independent of ACP, UCP, and any future protocol plugin. Those
protocols should map into these app-owned contracts through thin adapters later.

## Goals

- Give agents, tools, tests, and future protocol adapters one stable shape for
  products, carts, and tool results.
- Move commerce behavior toward deterministic tools instead of agent-specific
  prose parsing.
- Preserve the current runtime behavior while creating a safe path toward a
  Deep Agents runtime.

## Purpose Of This PR

This PR establishes the internal commerce language that later tools will use. It
does not make the current assistant more capable by itself; it gives the next
PRs a small, reviewed target for product identity, cart identity, tool metadata,
and structured errors.

The main design decision is separation:

- Product/catalog search is read-only and stateless.
- Cart operations are stateful and mutating.
- Store-policy lookup is read-only and controlled.
- ACP/UCP mappings are future adapter concerns, not core model fields.

## Current Phase

Phase 1 is contract-only. The models live in `shared/commerce_contracts.py` and
are not wired into the chain server runtime yet.

This means:

- No user-facing behavior changes.
- No cart or catalog service schema changes yet.
- Existing LangGraph agents continue to run as they do today.

## Core Models

| Model | Purpose |
| --- | --- |
| `Money` | Currency amount with a default `USD` currency. |
| `ProductSummary` | Search-result-safe product identity and display fields. |
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
| `GetProductDetailsInput` / `GetProductDetailsResult` | Read-only | Fetch one product by stable `product_id`. |
| `GetCartInput` / `GetCartResult` | Read-only | Read the authoritative cart for a user. |
| `GetStorePolicyInput` / `GetStorePolicyResult` | Read-only | Fetch controlled store-policy text by topic. |
| `AddCartItemInput` / `CartMutationResult` | Mutating | Add a product or variant to the cart. |
| `UpdateCartItemInput` / `CartMutationResult` | Mutating | Change cart-line quantity. Quantity `0` means remove. |
| `RemoveCartItemInput` / `CartMutationResult` | Mutating | Remove a cart line. |

Mutating inputs require `idempotency_key` so future agent retries and protocol
adapters can avoid duplicate cart changes.

### Stateless Catalog Search

`SearchCatalogInput` intentionally has no `user_id`, cart, memory, session, or
conversation-history fields. The agent layer can use conversation context to
decide what query, categories, and filters to send, but `search_catalog` itself
should remain a pure read against the catalog for the supplied request.

This keeps product search reusable by the current LangGraph retriever, future
Deep Agents subagents, and later protocol adapters without coupling catalog
results to shopper session state.

## Migration Direction

The next phases should wrap existing behavior behind these contracts without
changing the public API first:

1. Make `RetrieverAgent` call an internal `search_catalog` tool wrapper.
2. Make `CartAgent` call internal cart tool wrappers.
3. Preserve catalog IDs in chain-server state and cart memory rows.
4. Add Deep Agents skills and subagents on top of the same tool layer.

The important rule is that ACP/UCP compatibility should be added as adapters
around these contracts, not as fields or naming choices inside the core models.
