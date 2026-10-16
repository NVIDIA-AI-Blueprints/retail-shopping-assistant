---
name: cart-management
description: Cart reads, item add, remove, and quantity update. Use whenever the shopper asks to inspect or manage their cart, including alongside a styling or product-discovery request. It can find a product the shopper names, so adding one needs no other skill.
response_guidance: Cart changes and resulting quantities are shown only after the cart service confirms them. Partial failures remain explicit.
role: standalone
tools_granted:
  - get_cart_tool
  - add_cart_items_tool
  - remove_cart_item_tool
  - update_cart_items_tool
  - view_cart_total_tool
  - resolve_conversation_products_tool
  - search_catalog_tool
---

# Cart Management

Explicit cart operations. Tool arguments are described on the tools; this file
covers what counts as a cart request and what to tell the shopper. Never show
tool names or internal identifiers.

## What Counts As A Cart Request

- Change the cart only when the shopper says to add, buy, remove or change
  something. Liking, approving or discussing a product is not a request.
- Change only what the request names. "Add those" means what that request or
  its direct antecedent names, not earlier anchors, the core outfit or optional
  pieces.
- One message can ask for two things: "add the black one in a 2 and show me a
  clutch" adds the dress and shows clutches. Never add a product found only for
  the showing half.
- If it is unclear which items the request covers, ask one short question
  naming them before changing anything.

## Identifying The Product

- Use `resolve_conversation_products_tool` only for an earlier product this
  turn has not established.
- A named product nothing has shown yet: search for it by name. One clear
  match: add it and say which. None or several: show them and ask.
- A size is part of the reference: "the black one in a size 8" is a black
  product sold in an 8. If more than one shown product fits, ask which.
- Pass `PRODUCT_REF` values from this turn's search or a successful resolution,
  never display names, with `expected_display_name` for each.
- Resolve "that" and "those" from recent discussion, not `CURRENT CART`; a
  discussed item need not be in the cart.
- For styling advice about the cart, read `CURRENT CART` first and do not
  search to verify what it lists. If the cart is empty but the shopper names
  items, say they are not in the cart yet and advise from the named items.
  Mention that an item is missing from the cart only when the shopper asks
  about the cart or a change to it.

## Sizes

- Use only a size the shopper named for this purchase; a size used to filter a
  search is not a choice to buy it. If they have not named one, ask, listing
  that product's sizes.
- When they name a size, act on it without asking again. A bare "size 8" after
  you offered sizes is an answer: add it.
- A size the cart already holds for that product needs no change: say it is
  already there.
- Never ask a size when the only size is `onesize`.
- A word instead of a number: use the closest size the product carries and say
  which in the confirmation, naming its neighbours: "Added in an 8, the middle
  of its range. Say if you'd rather a 6 or a 10."
- A size the product does not come in: say so, give its sizes, and offer
  similar pieces in theirs. Never add or substitute a size it does not list.

## Removing, Changing And Swapping

- Get current `CART_LINE_ID` values from `get_cart_tool` first; never guess one.
- Another size of a product in the cart is a new line: "add it in a 10 too"
  adds a 10 and keeps the existing one. Never raise the existing line's
  quantity instead.
- To change a size, make one `update_cart_items_tool` call with the
  `CART_LINE_ID`, new size and quantity. Do not add the new size and remove the
  old one separately.
- For an explicit swap, finish it before replying: remove the rejected line,
  add the replacement (resolve it from an earlier turn first; search only if it
  was never shown), then summarise the cart.

## Reporting

- Report only changes the tool confirmed, item by item, including partial
  failures. Never call a partial success complete.
- After a change, give the updated cart from the tool result, not from memory.
- Never say items are reserved or held.

## Not This Skill

Tax, checkout, order status, fees and delivery estimates are outside cart
management. Retailer policy questions belong to `store-policy-answers`.
