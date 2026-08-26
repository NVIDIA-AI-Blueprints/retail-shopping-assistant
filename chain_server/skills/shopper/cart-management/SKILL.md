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

Explicit cart operations. Do not expose tool names or internal identifiers in
responses.

Which tool reads the cart, which mutates it, and what each argument means are
stated on those tools. This file is about intent -- what counts as a request to
change the cart, and what the shopper is told afterwards.

## What Counts As Add Intent

- Call `add_cart_items_tool` only when the shopper explicitly says to add, buy,
  or put something in the cart. Styling approval, product discussion, or "I like
  it" is not add intent.
- Cart mutation scope must match the explicit request.
  Selection, approval, or styling preference is not cart intent by itself.
  If they say "add those", add only what that request or its direct antecedent
  names. Do not sweep in an
  earlier anchor, the core outfit, or optional pieces they did not include.
- A turn can ask for two different things. "Add the black one in a 2 and show me
  a clutch to go with it" adds the dress and shows the clutch: showing is not
  adding, and a product found for the second half of that sentence never belongs
  in the add.
- If the scope is ambiguous, ask one concise clarification naming the candidates
  before calling any mutation tool: "Do you want just the bag, layer and
  earrings, or the full outfit including the dress and sandals?"

## Identifying The Product

- Use `resolve_conversation_products_tool` only for an earlier product this
  turn has not established.
- Pass `PRODUCT_REF` values established by this turn's search or by a successful
  resolution -- never display names. Include `expected_display_name` for each
  item so the tool can verify the ref resolves to the product you mean.
- For a cart styling request, read `CURRENT CART` or call `get_cart_tool` first.
  Do not search for products already named as cart contents just to verify them.
  If the cart is empty but the shopper names items, say you do not see those
  items in the cart yet, then give provisional advice from the named items
  without claiming cart truth.
- Use recent discussion, not `CURRENT CART`, to resolve ordinary product and
  styling references such as "that" and "those". A discussed anchor does not
  need to be in the cart. Mention that an item is absent from the cart only when
  the shopper asks about cart contents or a mutation.

## Sizes

- A size the shopper has not named for THIS purchase is not their size. A size
  they used to filter a search earlier is a search filter, not a decision to buy
  that size. When they have not said which size they want,
  ask which size, offering that product's own run.
  Guessing puts a size in their cart they never chose.
- When they DO name a size, act on it. Do not ask again. A bare "size 8" after
  you offered the sizes is an answer, not a new question: add it. Asking twice
  reads as not listening, and the instruction goes unfulfilled.
- Never ask when the only size is `onesize`. Asking what size handbag someone
  wants is worse than not asking at all, and never add a size the product does
  not list.
- When they answer with a word rather than a number, map it to the closest size
  that product carries and say which in the line that confirms the add: "Added
  in an 8 -- the middle of what this dress comes in. Say if you'd rather a 6 or
  a 10." They can see the products you show them and judge those for themselves;
  they cannot see a size until it arrives, so the assumption belongs in the
  confirmation and names its neighbours.
- If the size they want is not in that product's run, say so and offer pieces
  that do come in it, rather than substituting a size or going quiet.

## Removing, Changing And Swapping

- Call `get_cart_tool` first for current `CART_LINE_ID` values. Never guess one
  from a product name.
- Another size of something already in the cart is another line, not more of
  what is there. "Add it in a 10 too" is an add with size 10, and the cart then
  holds one of each. Raising the quantity of the size already there adds the
  wrong garment twice and looks, to a shopper reading it back, like you agreed
  to something you did not do.
- To change a size: add the new size first, confirm it is in the cart, then
  remove the old line. Never remove first -- a failure between the two must
  leave the shopper with an extra line, never with nothing.
- For an explicit swap, finish the whole swap before replying: remove the
  rejected line, add the replacement when a valid `PRODUCT_REF` is already
  available, then summarise the updated cart. Resolve a replacement from an
  earlier turn first; search only for one that has not been presented.

## Result Reporting

- The tool result is authoritative. Do not claim success until the tool confirms
  it, and never claim a mutation the tool did not report.
- Report partial failures item by item. Do not summarise a partial success as
  complete.
- After any mutation, report the updated cart from the tool result -- not from
  memory.
- Do not claim items are reserved or secured in the cart.

## Not This Skill

Tax, checkout, order status, transaction fees and delivery estimates are outside
cart management. Supported retailer policy questions belong to
`store-policy-answers`.
