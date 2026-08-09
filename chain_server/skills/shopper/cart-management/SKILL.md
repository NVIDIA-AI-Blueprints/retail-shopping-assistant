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
- Pass `PRODUCT_REF` values established by current-turn search or successful
  historical-product resolution — never product names.
- For multiple items, call `add_cart_items_tool` once with the full list.

## Remove and Update Rules

- Call `get_cart_tool` first to get current `CART_LINE_ID` values before calling `remove_cart_item_tool` or `update_cart_items_tool`.
- Never guess a `CART_LINE_ID` from a product name.
- Use `update_cart_items_tool` for quantity changes. Use `remove_cart_item_tool` for removals. Do not remove-and-re-add to change quantity.

## Result Reporting

- The tool result is authoritative. Do not claim success until the tool confirms it.
- Report partial failures item by item. Do not summarize a partial success as complete.
- After any cart mutation, report the updated cart state from the tool result — do not recall from memory.

## What This Skill Does Not Cover

- Tax, checkout, order status, and transaction-specific fees or delivery estimates are outside cart management; supported retailer policy questions require `store-policy-answers`.
- Do not claim items are reserved or secured in cart.
