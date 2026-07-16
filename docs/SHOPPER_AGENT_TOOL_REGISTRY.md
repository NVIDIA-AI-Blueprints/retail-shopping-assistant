# Shopper Agent Tool Registry

This registry documents the internal tools available to the shopper-serving
Deep Agent. These names are for engineers, evaluators, and agent instructions.
They are not shopper-facing UI language and should not appear in assistant
responses.

The runtime source of truth is
`chain_server/src/deepagents_runtime.py::DeepAgentsRuntime._create_agent`. A
tool is available to the shopper-serving Deep Agent only when it is registered
in the `create_deep_agent(..., tools=[...])` call.

## Current Runtime Boundary

The active Deep Agents runtime registers six app-owned shopping tools and
excludes default Deep Agents filesystem write/edit/list/search tools, todo
tools, shell tools, and the general-purpose subagent from the shopper-facing
harness. Built-in `read_file` remains available so the model can read static
skill files from the virtual-mode skill backend rooted at `chain_server/skills`.
Customer data, catalog truth, cart state, and prices stay in application
services rather than local files or agent-owned memory.

| Tool | Risk class | Source of truth | Status |
| --- | --- | --- | --- |
| `search_catalog_tool` | `read_only_catalog` | Catalog retriever | Registered |
| `get_product_details_tool` | `read_only_catalog` | Active catalog snapshot; conversation cache authorizes the ref | Registered |
| `get_cart_tool` | `read_only_cart` | Memory cart service | Registered |
| `view_cart_total_tool` | `computed_read_cart` | Memory cart service plus cached line prices | Registered |
| `add_cart_items_tool` | `mutating_cart` | Memory cart service | Registered |
| `remove_cart_item_tool` | `mutating_cart` | Memory cart service | Registered |

## Risk Classes

| Class | Meaning | Rules |
| --- | --- | --- |
| `read_only_catalog` | Reads catalog data without shopper state mutation. | Broadly available to discovery, styling, visual, comparison, and budget skills. |
| `read_only_catalog_cache` | Legacy/cache-only catalog read classification. | No active registered tool uses the cache as product-detail truth. |
| `read_only_cart` | Reads the authoritative cart for the scoped `cart_id`. | Safe for cart summaries and budget checks. |
| `computed_read_cart` | Computes over authoritative cart reads. | Do arithmetic in code, not model prose. |
| `mutating_cart` | Changes cart contents. | Requires explicit shopper intent, valid refs, and service-side success before claiming the change. |
| `future_high_risk` | Checkout, payment, orders, account changes. | Not registered. Requires stronger auth, idempotency, ownership checks, and confirmation policy before use. |

## Registered Tools

### `search_catalog_tool`

Purpose: Product discovery and recommendation over the catalog.

Inputs:

- `semantic_query`: Product meaning plus soft or descriptive preferences, such
  as product type, style, occasion, material, silhouette, or visual descriptors.
  A preference such as "maybe cotton" belongs here.
- `required_constraints`: Every shopper must-have as a structured field and
  value, including requirements that capabilities mark semantic/detail-only or
  do not advertise. Advertised numeric constraints use objects such as
  `{"max": 100}`; advertised enum constraints use exact capability values.
- `search_mode`: Optional `text`, `image`, or `hybrid` when supported by
  catalog capabilities.

Preconditions:

- Requires either semantic product text or an attached image.
- Deterministic validation converts supported `required_constraints` into
  catalog hard filters. Any unsupported field, value, or operator stops the
  search instead of being dropped or treated as semantic relevance.
- The per-turn catalog search cap applies. When reached, the tool returns a
  `STOP_TOOL_USE` instruction so the agent should answer from evidence already
  collected instead of continuing the loop.

Outputs:

- Candidate text results with `PRODUCT_REF`, name, category when available,
  price and image URL when available, plus a reminder that search evidence only
  supports names, prices, categories, image availability, and modest styling
  role. Search results intentionally do not expose the full catalog description
  to the agent loop.
- Product payloads are also appended to the runtime product result stream.
- Product image URLs are appended to the runtime retrieved-image map.

Side effects:

- Caches returned `PRODUCT_REF` values inside the scoped conversation for later
  product details or cart add calls.
- Cached refs let later product-detail or cart calls resolve a previously found
  product without another catalog search. Each shopper turn still runs the
  assistant model.
- Records catalog timing and model-usage metadata.

Failure behavior:

- Returns a short tool-readable failure string such as unavailable catalog,
  invalid search request, catalog failure, or no matching products.
- Returns without calling catalog search when a required constraint cannot be
  enforced by the active capabilities.
- If the Deep Agents loop fails after catalog search has returned products, the
  runtime clears the failed thread checkpoint and returns a grounded partial
  product summary instead of a generic shopper-facing error.

Permitted skill roles (only registered skills are active):

- `product-discovery`
- `outfit-styling`
- `visual-shopping`
- `budget-shopping`
- `product-comparison`

Current limitations:

- Returned product IDs are source `record_id` values. The current feed does not
  guarantee those generated IDs across catalog replacements.
- Broad multi-item outfit turns must stay within the configured search cap.

### `get_product_details_tool`

Purpose: Read deeper facts for a known product returned by
`search_catalog_tool`.

Inputs:

- `product_ref`: A `PRODUCT_REF` from a prior catalog search in the same
  conversation.

Preconditions:

- The ref must exist in the current conversation's product-ref cache.
- The agent must not pass display names as refs.
- The per-turn product-detail read cap applies. When reached, the tool returns
  a `STOP_TOOL_USE` instruction so the agent answers from details already read.

Outputs:

- Product name, category, brand when available, price, and image URL when
  available.
- Generic structured attributes marked `detail` by the active catalog sidecar,
  including exact composition or care text when present. Missing attributes
  remain unavailable; the tool does not infer them or expose raw marketing copy
  as factual detail evidence.
- Product image URLs are appended to the runtime retrieved-image map so the UI
  can show images for detail follow-ups without requiring a repeat search.
- Cart items are rehydrated from the conversation's cached product refs when
  possible, so a later "show that image again" turn can return the image after
  the item has been added to cart.

Typical use:

- Product fact questions about a known result.
- Detail expansion after a product search.
- Detailed comparison tables or claims about materials, dimensions, pockets,
  closures, care, comfort, or outdoor practicality.
- Not required for the first broad no-anchor outfit recommendation; the agent
  should search by item role and keep that initial rationale modest.

Side effects:

- None beyond normal tool-call telemetry.

Failure behavior:

- Returns guidance to search the catalog first when the ref is unknown.
- Returns guidance to search again when the cached ref is absent from the
  active catalog snapshot.

Permitted skill roles (only registered skills are active):

- `product-discovery`
- `outfit-styling`
- `visual-shopping`
- `budget-shopping`
- `product-comparison`

Current limitations:

- Source IDs in the current feed are generated and are not guaranteed across
  catalog replacements. Lookup is deterministic within the active snapshot.

### `get_cart_tool`

Purpose: Read the current authoritative cart.

Inputs:

- None from the model. The runtime supplies the scoped cart identity.

Preconditions:

- Request identity must include or derive the active `cart_id`.

Outputs:

- Cart lines formatted with `CART_LINE_ID`, quantity, display name, and cached
  unit price when available.

Side effects:

- Updates `state.cart` with the latest cart service response.

Failure behavior:

- Returns an empty cart if the cart read fails through the wrapper.

Permitted skill roles (only registered skills are active):

- `budget-shopping`
- `cart-management`
- `outfit-styling`

Current limitations:

- The memory service currently stores a simple cart shape. Durable product IDs,
  variants, and inventory are future work.

### `view_cart_total_tool`

Purpose: Compute the current cart total deterministically.

Inputs:

- None from the model. The runtime reads the scoped cart.

Preconditions:

- Cart line prices must be present to include an item in the subtotal.

Outputs:

- Itemized line totals and subtotal.
- Notes items whose cached price is unavailable.

Side effects:

- Updates `state.cart` with the latest cart service response.

Failure behavior:

- Reports an empty cart total when no cart lines are available.

Permitted skill roles (only registered skills are active):

- `budget-shopping`
- `cart-management`
- `outfit-styling`

Current limitations:

- Computes from cached cart line prices only. It does not reprice against live
  catalog, discounts, tax, shipping, inventory, or promotions.

### `add_cart_items_tool`

Purpose: Add one or more searched catalog products to the cart.

Inputs:

- `items`: Non-empty list of products to add.
- `items[].product_ref`: A `PRODUCT_REF` from `search_catalog_tool`.
- `items[].quantity`: Positive integer quantity. Missing values default to `1`;
  invalid values are rejected by the tool schema.
- `items[].expected_display_name`: Shopper-facing product name the agent
  intends to add. When the shopper explicitly names the product, the agent
  should copy that exact name so the tool can verify the selected ref.

Preconditions:

- Shopper must explicitly ask to add, buy, or put the item in the cart.
- Each product must have been returned by catalog search in this conversation.
- The agent must pass `PRODUCT_REF` values, not display names.
- The selected `PRODUCT_REF` must resolve to the intended
  `expected_display_name` when provided.
- Cart mutation scope must match the explicit add request. Styling approval or
  product selection is not enough to add an item unless the shopper also asks
  to add, buy, or put it in the cart.
- When the current add request contains exact cached product names or
  sufficiently specific abbreviated cached names, the tool blocks requested
  refs outside that named set before mutating the cart. This prevents a wrong
  remembered ref from adding a different item than the shopper asked for.
- Pronouns such as "those" should resolve only to the products named in the add
  request or its direct antecedent, not every previously discussed outfit
  component.
- If the add scope is ambiguous, the agent should ask one concise clarification
  before calling this tool.
- For multi-item add requests, the agent should call this tool once with the
  selected item list instead of making separate add calls.

Outputs:

- Tool-readable add result with separate `Added` and `Failed` sections.
- Current cart contents and deterministic cart total after the mutation attempt.

Side effects:

- Mutates the scoped cart through the memory cart service.
- Reads the cart again after mutation and updates `state.cart`.
- Uses a runtime-generated idempotency key.
- Duplicate refs in one call are aggregated before mutation.

Failure behavior:

- Unknown refs and service failures are reported per item.
- Expected-name mismatches and explicit-scope mismatches are blocked before any
  cart mutation in that call; the model can retry with the correct cached ref
  or ask a clarification.
- Partial success is allowed and must be described accurately in the final
  assistant response.

Permitted skill roles (only registered skills are active):

- `cart-management`

Current limitations:

- The idempotency key is generated and echoed in metadata, but the current
  memory service adapter does not enforce deduplication yet.
- Variant, inventory, and checkout validation are not part of this tool.

### `remove_cart_item_tool`

Purpose: Remove quantity from an explicit cart line.

Inputs:

- `cart_line_id`: A `CART_LINE_ID` from current cart output or
  `get_cart_tool`.
- `quantity`: Positive integer quantity. Invalid or missing values are coerced
  to `1`.

Preconditions:

- Shopper must explicitly ask to remove an item.
- The tool must receive a valid `CART_LINE_ID`.
- The agent must not guess cart line IDs from product names.

Outputs:

- Authoritative success or failure message from the cart mutation wrapper.
- The result is returned to the model, not directly to the shopper, so compound
  requests such as "swap this for that and tell me the new total" can finish
  all required tool calls before the final response.

Side effects:

- Reads the current cart to validate the line.
- Mutates the scoped cart through the memory cart service.
- Reads the cart again after mutation and updates `state.cart`.
- Uses a runtime-generated idempotency key.

Failure behavior:

- Returns a clear missing-line or mutation failure message and must not be
  described as success.

Permitted skill roles (only registered skills are active):

- `cart-management`

Current limitations:

- The idempotency key is generated and echoed in metadata, but the current
  memory service adapter does not enforce deduplication yet.
- Quantity update and clear-cart operations are not registered as separate
  tools.

## Skill Access Matrix

This matrix documents which active tools each skill may guide the agent to use.
Only skills listed as registered in
[Shopper Agent Skill Registry](SHOPPER_AGENT_SKILL_REGISTRY.md) are loaded by
the runtime today.

| Skill | Allowed registered tools |
| --- | --- |
| `product-discovery` | `search_catalog_tool`, `get_product_details_tool` |
| `outfit-styling` | `search_catalog_tool`, `get_product_details_tool`, `get_cart_tool`, `view_cart_total_tool`; cart mutation tools only when explicit cart intent is present |
| `visual-shopping` | `search_catalog_tool`, `get_product_details_tool` |
| `budget-shopping` | `search_catalog_tool`, `get_product_details_tool`, `get_cart_tool`, `view_cart_total_tool` |
| `cart-management` | `get_cart_tool`, `view_cart_total_tool`, `add_cart_items_tool`, `remove_cart_item_tool` |
| `product-comparison` | `search_catalog_tool`, `get_product_details_tool` |
| `preference-aware-shopping` | `search_catalog_tool`, `get_product_details_tool` |
| `store-policy-answers` | No registered policy tool yet |

Multi-intent turns may use more than one skill in the same assistant pass. For
example, a request to "find shoes and a bag for this outfit under $120 and add
the shoes" can combine visual shopping, outfit styling, budget shopping, product
discovery, and cart management. The cart mutation still happens only after a
valid product ref is selected and explicit add intent is present.

## Not Registered Today

The following capabilities may exist as contracts, docs, or future direction,
but they are not registered tools in the active Deep Agents runtime:

| Capability | Current status |
| --- | --- |
| `get_store_policy_tool` | Planned. No registered runtime tool. |
| `update_cart_item_tool` | Planned contract. No registered runtime tool. |
| `load_customer_persona_tool` | Planned. No registered runtime tool. |
| Cross-catalog durable product identity | Planned; requires an upstream stable ID guarantee. |
| Inventory, variant, and size availability lookup | Not implemented as a tool. |
| Checkout, order, payment, address, or account mutation | Not implemented and should be treated as `future_high_risk`. |
| Outfit styling tool | Not a tool. Styling is model behavior guided by skills over catalog results. |
| Media perception tool | Not an agent-callable tool. Media analysis runs before the Deep Agents turn and is passed as context. |

## Registration Standards For New Tools

Every new shopper-agent tool should satisfy these requirements before it is
added to the Deep Agents runtime:

1. Stable internal tool name that matches this registry.
2. Typed input schema for nontrivial arguments.
3. Docstring that states the action in model-readable language.
4. Explicit risk class.
5. Clear source of truth and owning service.
6. Compact output that includes refs needed by follow-on tools.
7. Structured failure behavior that the final response can ground on.
8. No raw secrets, private hosts, or customer-sensitive data in tool output.
9. `return_direct=True` only for tool outputs that should bypass the final
   model response. Cart reads, totals, and mutations should normally return to
   the model so multi-step cart requests can finish before the shopper-facing
   answer.
10. Unit coverage proving the tool is registered and that required refs are
    produced or consumed correctly.

Mutating tools also need explicit user intent, idempotency design, ownership
checks, and retry behavior. Checkout, payment, order, account, and profile-write
tools require a separate confirmation and authorization design before they are
eligible for the shopper-facing agent.
