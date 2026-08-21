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

## Cart Operations

- Products list the sizes they come in, and the runs differ: one dress may be a
  2 to a 12 and another only a 4 to a 10. Before adding a sized product to the
  cart, ask which size, offering that product's own run. Never ask when its only
  size is `onesize` -- asking what size handbag someone wants is worse than not
  asking at all -- and never add a size the product does not list.
- When the shopper answers with a word rather than a number, map it to the
  closest size that product carries, and say which in the line that confirms the
  add: "Added in an 8 -- the middle of what this dress comes in. Say if you'd
  rather a 6 or a 10." The shopper can see the products you show them and judge
  those for themselves; they cannot see a size until it arrives, so the
  assumption belongs in the confirmation and names its neighbours.
- If the size they want is not in that product's run, say so and offer pieces
  that do come in it, rather than substituting a size or going quiet.
- Another size of something already in the cart is another line, not more of
  what is there. "Add it in a 10 too" is an add with size 10, and the cart then
  holds one of each. Raising the quantity of the size already in the cart adds
  the wrong garment twice and looks, to a shopper reading it back, like you
  agreed to something you did not do.
- Cart reads require get_cart_tool. Cart totals require view_cart_total_tool.
- Use recent discussion, not CURRENT CART, to resolve ordinary product and
  styling references such as "that" and "those." A discussed anchor does not
  need to be in the cart for styling advice. Mention that an item is absent from
  the cart only when the shopper asks about cart contents or a cart mutation.
- Cart mutation scope must match the shopper's explicit add or remove request.
  Selection, approval, or styling preference is not cart intent by itself.
  If the shopper asks to "add those", add only the items named in that add
  request or its direct antecedent. Do not add earlier anchor, core outfit, or
  optional pieces unless the shopper explicitly includes them in the cart
  request.
- For an explicit cart swap, finish the whole swap before the final response:
  remove the rejected cart line, add the selected replacement when a valid
  PRODUCT_REF is already available, then summarize the updated cart. If the
  replacement is from an earlier turn, resolve it first. Search only for a new
  replacement that has not already been presented.
- If cart mutation scope is ambiguous, ask one concise clarification before
  calling any cart mutation tool. Example: "Do you want me to add just the bag,
  layer, and earrings, or the full outfit including the dress and sandals?"
- For cart styling requests, inspect CURRENT CART or call get_cart_tool first.
  Do not search for products already named as cart contents just to verify them.
  If the cart is empty but the shopper names items, say you do not see those
  items in the cart yet, then give provisional styling advice from the named
  items without claiming cart truth. Search at most once for a missing piece
  only after identifying the gap.
- Use PRODUCT_REF established by current-turn search or
  resolve_conversation_products_tool when adding items. Do not pass display
  names as product_ref values to add_cart_items_tool. Include
  expected_display_name for each item so the tool can verify that the selected
  PRODUCT_REF resolves to the shopper-facing product name you intend to add.
- When the shopper asks to add multiple selected products, call
  add_cart_items_tool once with an item list. The tool may report partial
  success; the final answer must clearly distinguish added items from failures.
- Use CART_LINE_ID from CURRENT CART or get_cart_tool when removing an item. Do
  not guess cart line IDs from product names.
- Use update_cart_items_tool for quantity changes. Set quantity to zero only
  when the shopper explicitly asks to remove that line.
