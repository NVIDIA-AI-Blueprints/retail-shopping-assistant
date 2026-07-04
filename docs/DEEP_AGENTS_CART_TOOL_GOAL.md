# Deep Agents Cart Tool Goal

This branch first proves the minimal cart-tool loop before adding broader
features.

## Commit Gate

The ref-based cart tool changes should pass this live four-step shopper flow
before commit:

1. Find a black bag under $100 and add it to my cart.
2. What is in my cart?
3. Remove that bag from my cart.
4. What is my total?

## Intended Shape

- Use Deep Agents as the only shopper-language interpreter.
- Keep cart mutations as deterministic tools.
- Add by explicit `PRODUCT_REF` returned from catalog search.
- Remove by explicit `CART_LINE_ID` returned from cart reads.
- Keep discovery and cart-read tools chainable inside the agent loop.
- Keep mutation tools authoritative and idempotent.

## Constraints

- Do not reintroduce hidden product-name lookup inside add-cart mutation.
- Do not reintroduce fuzzy cart-line matching inside remove-cart mutation.
- Do not add a standalone cart agent or planner.
- Do not add broad keyword taxonomies for shopper intent.

## Next Work

1. Keep this smoke green while changing the cart harness.
2. If search-then-add fails, fix the Deep Agents tool contract or prompt first.
3. If remove-by-reference fails, improve cart-read output or tool naming first.
4. Only then consider richer features such as durable SKU lookup, product-detail
   tools, multi-item cart operations, or cart confirmation policy.
