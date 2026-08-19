---
name: cart-management
description: Cart reads, item add, remove, and quantity update. Use whenever the shopper asks to inspect or manage their cart, including alongside a styling or product-discovery request.
response_guidance: Cart changes and resulting quantities are shown only after the cart service confirms them. Partial failures remain explicit.
role: standalone
tools_granted:
  - get_cart_tool
  - add_cart_items_tool
  - remove_cart_item_tool
  - update_cart_items_tool
  - view_cart_total_tool
  - resolve_conversation_products_tool
---

# Cart Management

Handle explicit cart operations. Do not expose tool names or internal identifiers in responses.

## Add Intent Rules

- Use `resolve_conversation_products_tool` only for an earlier product not
  established this turn. Multiple matches require one concise clarification;
  never guess or mutate the cart.
- Zero matches means the shopper referred to something never shown. If they
  named a product, search the catalog and show the closest matches, then ask
  which to add. If they pointed at an earlier item, ask which one. Never add a
  product the shopper has not been shown, and never offer to accept a product
  link or a price as identification.
- Call `add_cart_items_tool` only when the shopper explicitly says to add, buy, or put an item in the cart.
- Styling approval, product discussion, or "I like it" is NOT add intent.
- If the add scope is ambiguous ("add those", "add them all"), ask one concise clarification naming the candidates before calling the tool.
- A size the shopper has not named for THIS purchase is not their size. A size
  they used to filter a search earlier is a search filter, not a decision to buy
  that size. When they have not said which size they want, ask -- naming the
  sizes the product is sold in. Guessing puts a size in their cart they never
  chose.
- When they DO name a size, act on it. Do not ask again. A bare "size 8" after
  you offered the sizes is an answer, not a new question: add it. Asking twice
  reads as not listening, and the shopper's instruction goes unfulfilled.
- Pass `PRODUCT_REF` values established by current-turn search or successful
  historical-product resolution — never product names.
- For multiple items, call `add_cart_items_tool` once with the full list.

## Remove and Update Rules

- Call `get_cart_tool` first to get current `CART_LINE_ID` values before calling `remove_cart_item_tool` or `update_cart_items_tool`.
- Never guess a `CART_LINE_ID` from a product name.
- The cart marks the line it took most recently as `ADDED MOST RECENTLY`. When the shopper points at their last action rather than naming a product -- "that one", "make it a 6 instead", "actually swap it" -- that is the line they mean, and it is the cart's own record of the order, not an inference from what was said earlier. Act on it. Ask only when they point at something else, or at more than one line.
- Use `update_cart_items_tool` for quantity changes. Use `remove_cart_item_tool` for removals. Do not remove-and-re-add to change quantity.
- A size is a different line, not a different quantity. To change a size: add the
  new size first, confirm it is in the cart, then remove the old line. Never
  remove first — a failure between the two must leave the shopper with an extra
  line, never with nothing.

## Result Reporting

- The tool result is authoritative. Do not claim success until the tool confirms it.
- Report partial failures item by item. Do not summarize a partial success as complete.
- After any cart mutation, report the updated cart state from the tool result — do not recall from memory.

## What This Skill Does Not Cover

- Tax, checkout, order status, and transaction-specific fees or delivery estimates are outside cart management; supported retailer policy questions require `store-policy-answers`.
- Do not claim items are reserved or secured in cart.
