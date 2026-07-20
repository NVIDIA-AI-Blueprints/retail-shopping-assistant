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
---

# Cart Management

Handle explicit cart operations. Do not expose tool names or internal identifiers in responses.

## Add Intent Rules

- Call `add_cart_items_tool` only when the shopper explicitly says to add, buy, or put an item in the cart.
- Styling approval, product discussion, or "I like it" is NOT add intent.
- If the add scope is ambiguous ("add those", "add them all"), ask one concise clarification naming the candidates before calling the tool.
- Pass `PRODUCT_REF` values from prior search in this conversation — never product names.
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

- Tax, checkout, order status, and transaction-specific shipping fees or delivery estimates are not available through the assistant. Use `get_store_policy_tool` for controlled shipping policy questions.
- Do not claim items are reserved or secured in cart.
