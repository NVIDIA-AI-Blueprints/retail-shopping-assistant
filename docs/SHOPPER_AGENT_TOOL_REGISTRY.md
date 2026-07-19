# Shopper Agent Tool Registry

This registry documents the internal tools available to the shopper-serving
Deep Agent. These names are for engineers, evaluators, and agent instructions.
They are not shopper-facing UI language and should not appear in assistant
responses.

The runtime sources of truth are
`chain_server/src/deepagents_runtime.py::DeepAgentsRuntime._create_agent` and
`chain_server/src/skill_activation.py::ShopperSkillActivationMiddleware`, with
termination policy in
`chain_server/src/tool_loop_control.py::ToolLoopControlMiddleware`. A tool is
registered with the shopper-serving Deep Agent only when it appears in the
`create_deep_agent(..., tools=[...])` call. Per-turn availability is also
controlled by the activation and loop-control phases described below.

## Current Runtime Boundary

The active Deep Agents runtime registers nine app-owned shopper commerce tools
plus one internal activation control tool. Every turn begins in an activation
phase where the model sees only `activate_shopper_skills_tool`, its use is
forced, and parallel tool calls are disabled. After the model semantically
selects the smallest skill set for the complete current intent, the runtime
validates those names and deterministically injects the full selected
`SKILL.md` contents. Only then does the next model step receive the nine shopper
commerce tools.

For primary shopper procedure selection, `product-discovery` and
`outfit-styling` are mutually exclusive. `budget-shopping` is a modifier and is
selected only when the shopper states a budget. Cart or policy skills may still
join the applicable procedure for a genuine multi-intent turn; standalone cart
and policy turns do not require a product primary.

The active catalog capabilities generate both the exact taxonomy values and
the non-taxonomy required-constraint properties in the search-tool schema.
The model owns semantic selection from that schema. Deterministic code validates
and maps the values but does not infer structured fields from shopper prose.
Each call covers at most one category. For a broad request that names no type,
`agent_selected_type` may include the advertised subcategories that serve one
focused semantic role.

The activation boundary fails closed. Missing or invalid skill content exposes
no commerce tools. A commerce call placed in the same model response as the
activation call is rejected, and activation from a prior turn cannot unlock the
current turn. An activation-phase model response without the required call is
rejected as `skill_activation_failed`, rather than becoming shopper-facing
prose. The gate guarantees activation before commerce; it does not
implement a per-skill tool access-control list. Skill instructions guide which
of the exposed tools the model should use, while tool schemas and wrappers
enforce deterministic request and state preconditions.

This design adds one bounded model step to every turn. Deterministic skill-file
injection itself adds no model call. The runtime also excludes default Deep
Agents filesystem write/edit/list/search tools, todo tools, shell tools, and the
general-purpose subagent from the shopper-facing harness. Built-in `read_file`
remains available after activation for static read-only references in the
virtual-mode skill backend rooted at `chain_server/skills`; activated
`SKILL.md` content is already injected. Customer data, catalog truth, cart
state, and prices stay in application services rather than local files or
agent-owned memory.

The activation registry is the only skill-selection metadata source. The
runtime does not also enable Deep Agents' checkpointed `SkillsMiddleware`
metadata, so existing Redis threads cannot retain stale skill descriptions or
removed names across a deployment.

After activation, parallel shopping calls remain disabled. A first
`search_catalog_tool` schema-validation failure opens one search-only repair
step. A successful repaired partial search may continue to another valid role,
but a second repair is never available. A completed scope, a failed repaired
call, or any tool result beginning with `STOP_TOOL_USE` removes tools from the
following model step and forces synthesis from evidence already collected.

A partial multi-role search that includes an unadvertised requirement uses the
same one repair step for contextual review. The model must preserve a product
must-have stated directly in the current request, or remove only a requirement
inferred from broad season, weather, occasion, or style context. A completed
scope, unsupported taxonomy or requirement, failed repaired result, or
deterministic stop closes the loop. Only a successful repaired result whose
scope remains explicitly partial may continue with another valid role.

Grounding reads only tool-role messages and partitions current-turn evidence by
the server-owned request marker. Prior-turn tool evidence may resolve a direct
reference but cannot prove that a new search or mutation ran. For search-only
turns, successful search results carry `SEARCH_DIRECTION_EVIDENCE`: the
model-authored semantic query used as ranking preference. Code labels it as
preference rather than product fact and nominates the first ranked result, or
one first result per requested role, alongside deterministic candidate facts and
confirmed filters. No separate rationale model is called.

The runtime records model-issued calls for the activation control, app-owned,
and built-in tools in `agent_diagnostics`. Calls retain their model-issued order
and structured arguments; deterministic stops identify rejected and duplicate
calls. Search-schema validation failures are recorded as rejected catalog
requests rather than executed searches. A completed activation records each injected
`/shopper/<name>/SKILL.md` path in `skill_files_read`; a later successful
`read_file` of a skill file is recorded there as well. Pre-activation commerce
rejections use `skill_activation_required`. On graph failure, bounded
current-turn assistant/tool messages are read from the checkpoint before
cleanup. These diagnostics are operator/evaluation metadata and are not
shopper-facing response text.

Final-response extraction ignores tool messages, assistant messages that still
contain tool calls, and internal activation markers. If a completed graph has no
shopper-facing text, the runtime returns a safe retry response and changes the
termination reason to `incomplete_agent_response`.

| Tool | Risk class | Source of truth | Status |
| --- | --- | --- | --- |
| `activate_shopper_skills_tool` | `internal_control` | Validated static shopper-skill registry | Registered; required first step |
| `search_catalog_tool` | `read_only_catalog` | Catalog retriever | Registered |
| `get_product_details_tool` | `read_only_catalog` | Active catalog snapshot; process-local conversation cache authorizes the ref | Registered |
| `get_cart_tool` | `read_only_cart` | Memory cart service | Registered |
| `view_cart_total_tool` | `computed_read_cart` | Memory cart service plus cached line prices | Registered |
| `add_cart_items_tool` | `mutating_cart` | Memory cart service | Registered |
| `remove_cart_item_tool` | `mutating_cart` | Memory cart service | Registered |
| `update_cart_items_tool` | `mutating_cart` | Memory cart service | Registered |
| `get_store_policy_tool` | `read_only_policy` | Operator-managed static policy file | Registered |
| `check_product_availability_tool` | `read_only_catalog` | Application availability contract; no live inventory source | Registered |

## Risk Classes

| Class | Meaning | Rules |
| --- | --- | --- |
| `internal_control` | Selects and activates static behavioral instructions; it does not read or mutate commerce state. | Forced once at turn start; cannot be batched with commerce execution. |
| `read_only_catalog` | Reads catalog-domain data or reports its availability boundary without shopper state mutation. | Broadly available to discovery, styling, visual, comparison, and budget skills. |
| `read_only_catalog_cache` | Legacy/cache-only catalog read classification. | No active registered tool uses the cache as product-detail truth. |
| `read_only_cart` | Reads the authoritative cart for the scoped `cart_id`. | Safe for cart summaries and budget checks. |
| `computed_read_cart` | Computes over authoritative cart reads. | Do arithmetic in code, not model prose. |
| `mutating_cart` | Changes cart contents. | Requires explicit shopper intent, valid refs, and service-side success before claiming the change. |
| `read_only_policy` | Reads controlled operator-managed policy content. | Never substitute model knowledge when a topic is absent. |
| `future_high_risk` | Checkout, payment, orders, account changes. | Not registered. Requires stronger auth, idempotency, ownership checks, and confirmation policy before use. |

## Registered Tools

### `activate_shopper_skills_tool`

Purpose: Select the registered shopper skills whose complete instructions must
govern the current turn.

Inputs:

- `skill_names`: A non-empty list drawn from the registered skill-name enum.
  The runtime removes duplicates, and the model should choose the smallest set
  whose descriptions cover the complete current intent.

Preconditions:

- This is the forced first model step on every turn.
- Selection uses the full conversation and skill descriptions. It is semantic
  model work, not deterministic keyword matching.
- `product-discovery` and `outfit-styling` are alternative primary procedures;
  `budget-shopping` accompanies one only for an explicit shopper budget.
- The activation call must be completed before a shopper commerce call is
  issued in a later model step.

Outputs:

- A completion marker listing the virtual paths of the selected skill files.
- On the next model step, the complete selected files are injected into system
  context and the nine shopper commerce tools become available.

Side effects:

- Activates static instructions for the current request only.
- Adds the injected paths to `agent_diagnostics.skill_files_read`.
- Makes no catalog, cart, policy, availability, or other external service call.

Failure behavior:

- Unknown, empty, or invalid skill content fails closed and exposes no shopper
  commerce tools.
- A model response that tries to finish without activation fails the turn with
  final termination reason `skill_activation_failed`.
- Same-batch commerce calls are rejected with
  `skill_activation_required`; a successful activation from an earlier request
  does not satisfy the current turn.

Current limitations:

- The activation gate enforces the two-phase ordering boundary, not the
  advisory per-skill/tool mapping later in this document.
- The required selection phase adds one bounded model step to each turn.

### `search_catalog_tool`

Purpose: Product discovery and recommendation over the catalog.

Inputs:

- `semantic_query` (required): One soft/descriptive product query. Product words
  may be repeated for relevance, but this field never enforces taxonomy or
  another must-have. Use an empty string only for image-only search.
- `taxonomy` (required): `category` and `subcategory` arrays whose allowed values
  are generated from cached catalog capabilities. Use exact advertised values;
  executable text search requires at least one value, while image-only and
  no-direct no-retrieval requests use two empty arrays. A subcategory-only value
  maps to its owning category; an ambiguous owner or incompatible pair is
  rejected. Each call accepts at most one category.
- `taxonomy_status` (required): `exact_requested_type`,
  `member_of_requested_umbrella`, `agent_selected_type`,
  `no_direct_catalog_match`, or `image_only`. `agent_selected_type` is for a
  broad request that names no type; it may include every advertised subcategory
  serving one focused semantic role in one category. The no-direct value is only
  for an explicitly requested concrete product type with no faithful advertised
  taxonomy value, uses empty taxonomy and no hard constraints, and performs no
  retrieval.
- `required_constraints` (required): Capability-derived non-taxonomy properties
  only. Advertised enum values are exact and numeric constraints use
  `min`/`max`. A directly stated product must-have absent from those properties
  goes in `unadvertised_requirements`; unknown ad hoc properties are rejected.
  A preference such as "maybe cotton" stays in `semantic_query`.
- `scope_complete` (required): True only when this search plus existing
  current-turn evidence can answer the complete current request. A multi-role
  request keeps it false while another explicitly requested role, detail read,
  availability check, or cart action remains.
- `search_mode`: Optional `text`, `image`, or `hybrid` when supported by
  catalog capabilities.

Preconditions:

- Requires either product text or an attached image.
- The agent semantically maps the shopper's request to exact advertised
  taxonomy values. The deterministic runtime validates and maps those values to
  catalog fields; it does not interpret shopper language or maintain taxonomy
  keyword aliases.
- If an explicitly requested product type has no advertised match, the agent
  uses `no_direct_catalog_match`, which reports the gap without retrieval,
  before asking permission to search an adjacent type. It does not omit the
  type or broaden to its parent category. An unsupported modifier does not erase
  an advertised type, and subjective style stays in `semantic_query`.
- A directly stated product must-have that is absent from the generated schema
  is retained in `unadvertised_requirements`; it is never silently weakened into
  semantic relevance.
- Capability-derived taxonomy mapping and all other hard-filter validation
  complete before retrieval. Unsupported fields, values, and operators stop the
  search instead of becoming semantic relevance.
- One normalized taxonomy-plus-hard-constraint scope may execute once per turn.
  Duplicate values are removed before the scope key is reserved, and partially
  incompatible category/subcategory sets are rejected before retrieval. A
  repeated scope returns `STOP_TOOL_USE` even when `semantic_query` is
  paraphrased; genuinely different hard-constraint scopes share the configured
  search cap.

Outputs:

- Candidate text results with `PRODUCT_REF`, name, category when available,
  price and image URL when available, plus a reminder that search evidence only
  supports names, prices, categories, image availability, and modest styling
  role. Successful results also carry machine-readable advertised-taxonomy and
  confirmed-filter evidence. Search results intentionally do not expose the
  full catalog description to the agent loop.
- `SEARCH_DIRECTION_EVIDENCE` containing the model-authored `semantic_query`
  used for ranking. It is preference evidence, not a catalog-confirmed product
  fact.
- Product payloads are also appended to the runtime product result stream.
- Product image URLs are appended to the runtime retrieved-image map.
- The serving agent sends one text query per catalog call. The catalog performs
  no shopper-language interpretation, query expansion, or learned reranking.
- For a search-only styling response, code renders product names, prices,
  categories, confirmed filters, and ranking direction deterministically. It
  labels the direction as preference and nominates the first ranked result, or
  one first result per requested role, without calling a separate rationale
  model.

Side effects:

- Caches returned `PRODUCT_REF` values inside the scoped conversation for later
  product details or cart add calls.
- Cached refs let later product-detail or cart calls resolve a previously found
  product without another catalog search. Each shopper turn still runs the
  assistant model.
- The cache holds at most 50 refs per conversation and is process-local. Redis
  checkpointing does not persist or replicate it.
- Records catalog timing and model-usage metadata.

Failure behavior:

- Returns a short tool-readable failure string such as unavailable catalog,
  invalid search request, catalog failure, or no matching products.
- `no_direct_catalog_match` returns before catalog retrieval and produces the
  fixed shopper-safe no-match response.
- Returns without calling catalog search when a required constraint cannot be
  enforced by the active capabilities. A direct unadvertised must-have produces
  a fixed shopper-safe response asking whether to continue as a preference.
- On a partial multi-role search, `unadvertised_requirements` triggers one
  contextual review step. A directly stated must-have remains blocked; only an
  inference from broad season, weather, occasion, or style context may be
  removed before the single repaired search.
- A first search-schema validation failure permits one search-only repair. A
  successful repaired partial search may continue to another valid role; a
  second repair is not permitted.
- A completed scope, unsupported taxonomy or requirement, repeated scope,
  exhausted search/detail budget, or other `STOP_TOOL_USE` result forces final
  synthesis without another tool call.
- If the Deep Agents loop fails after catalog search has returned products, the
  runtime clears the failed thread checkpoint and returns a grounded partial
  product summary instead of a generic shopper-facing error.

Relevant activated skill roles (behavioral guidance, not a runtime ACL):

- `product-discovery`
- `outfit-styling`
- `budget-shopping`

Current limitations:

- Returned product IDs are source `record_id` values. The current feed does not
  guarantee those generated IDs across catalog replacements.
- A restart, another replica, cache eviction, or catalog replacement requires a
  fresh search before a ref can authorize details or cart adds.
- Broad multi-item outfit turns share the distinct taxonomy-scope budget.

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
  should search by item role and keep the initial explanation modest.

Side effects:

- None beyond normal tool-call telemetry.

Failure behavior:

- Returns guidance to search the catalog first when the ref is unknown.
- Returns guidance to search again when the cached ref is absent from the
  active catalog snapshot.

Relevant activated skill roles (behavioral guidance, not a runtime ACL):

- `product-discovery`
- `outfit-styling`
- `budget-shopping`

Current limitations:

- Source IDs in the current feed are generated and are not guaranteed across
  catalog replacements. Lookup is deterministic within the active snapshot.
- Ref authorization is process-local, bounded, and not part of Redis checkpoint
  state; a restart, another replica, or eviction requires a fresh search.

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

Relevant activated skill roles (behavioral guidance, not a runtime ACL):

- `budget-shopping`
- `cart-management`
- `outfit-styling`

Current limitations:

- The memory service uses `display_name` as its cart key. Returned
  `CART_LINE_ID` values are display-name aliases rather than stable
  server-generated IDs, so similarly named products can conflict.
- Durable product IDs, variants, and inventory are future work.

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

Relevant activated skill roles (behavioral guidance, not a runtime ACL):

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

Relevant activated skill roles (behavioral guidance, not a runtime ACL):

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

Relevant activated skill roles (behavioral guidance, not a runtime ACL):

- `cart-management`

Current limitations:

- The idempotency key is generated and echoed in metadata, but the current
  memory service adapter does not enforce deduplication yet.

### `update_cart_items_tool`

Purpose: Change the total quantity of one existing cart line, including
removing it when the requested quantity is `0`.

Inputs:

- `cart_line_id`: A `CART_LINE_ID` from `get_cart_tool`.
- `quantity`: New total quantity. `0` removes the line.

Preconditions:

- Shopper must explicitly ask to change the quantity or remove the item by
  quantity.
- The agent must read the current cart first and must not derive a line ID from
  a product name.

Outputs:

- Authoritative success or structured failure, followed by current cart state
  when available.

Side effects:

- Reads the current cart to validate the line.
- Quantity `0` removes the full current line quantity. A positive quantity
  removes the current line and adds the requested quantity back because the
  memory service has no dedicated update endpoint.

Failure behavior:

- Returns `cart_read_failed` when the cart cannot be read and
  `cart_line_not_found` when the supplied line ID is not current.
- A failed remove stops the operation before the replacement add.

Relevant activated skill roles (behavioral guidance, not a runtime ACL):

- `cart-management`

Current limitations:

- The positive-quantity remove-then-add sequence is not atomic.
- `CART_LINE_ID` remains a display-name alias until the memory service returns
  stable line IDs.

### `get_store_policy_tool`

Purpose: Read controlled store-policy content for a supported topic.

Inputs:

- `topic`: One of `returns`, `shipping`, `sizing`, `payment`, `price_match`, or
  `gift_cards`.

Preconditions:

- Policy questions must use this tool instead of model knowledge.

Outputs:

- The operator-managed policy title and body from
  `chain_server/skills/shopper/store-policy/policies.yaml`.

Side effects:

- Loads the policy file into a process-local read-only cache on first use.

Failure behavior:

- Returns `policy_topic_not_found` for an unavailable topic and
  `policy_load_failed` when the policy file cannot be read. The assistant
  should relay that the policy is unavailable and direct the shopper to the
  retailer's help center.

Relevant activated skill roles (behavioral guidance, not a runtime ACL):

- `store-policy-answers`

Current limitations:

- The bundled policy values are operator placeholders and must be replaced
  before production.

### `check_product_availability_tool`

Purpose: Give a consistent answer when a shopper asks about stock or a
specific product variant.

Inputs:

- `product_ref`: A `PRODUCT_REF` from a prior catalog search in the current
  conversation.
- `variant_hint`: Optional shopper-named size, color, or variant.

Preconditions:

- Use only for an explicit stock, availability, size, color, or variant
  question about a known product ref.

Outputs:

- `availability="unknown"` and a consistent message directing the shopper to
  the product page or checkout for confirmation.

Side effects:

- None. The deliberate stub makes no external call.

Failure behavior:

- None for a well-formed request. The stub never converts catalog presence
  into an inventory claim.

Relevant activated skill roles (behavioral guidance, not a runtime ACL):

- `product-discovery`
- `outfit-styling`
- `budget-shopping`

Current limitations:

- There is no live inventory or variant lookup behind this tool.

## Skill Access Matrix

This matrix documents which shopper commerce tools each activated skill may
guide the agent to use. It is a behavioral map, not a deterministic per-skill
authorization policy. The runtime enforces that at least one registered skill
is selected and fully injected before any of the nine commerce tools can run;
it does not filter those tools according to the selected row. Tool schemas and
wrappers continue to enforce deterministic preconditions independently. Only
skills listed as registered in
[Shopper Agent Skill Registry](SHOPPER_AGENT_SKILL_REGISTRY.md) are eligible for
activation.

| Skill | Tools this skill may guide |
| --- | --- |
| `product-discovery` | `search_catalog_tool`, `get_product_details_tool`, `check_product_availability_tool` |
| `outfit-styling` | `search_catalog_tool`, `get_product_details_tool`, `check_product_availability_tool`, `get_cart_tool`, `view_cart_total_tool`; cart mutation tools only when explicit cart intent is present |
| `budget-shopping` | `search_catalog_tool`, `get_product_details_tool`, `check_product_availability_tool`, `get_cart_tool`, `view_cart_total_tool` |
| `cart-management` | `get_cart_tool`, `view_cart_total_tool`, `add_cart_items_tool`, `remove_cart_item_tool`, `update_cart_items_tool` |
| `store-policy-answers` | `get_store_policy_tool` |

Multi-intent turns should select and inject every needed skill during the one
activation step. For example, a request to "find shoes and a bag for this
outfit under $120 and add the shoes" combines outfit styling, budget shopping,
and cart management, not product discovery. The cart mutation still
happens only after a valid product ref is selected and explicit add intent is
present.

## Not Registered Today

The following capabilities may exist as contracts, docs, or future direction,
but they are not registered tools in the active Deep Agents runtime:

| Capability | Current status |
| --- | --- |
| `load_customer_persona_tool` | Planned. No registered runtime tool. |
| Cross-catalog durable product identity | Planned; requires an upstream stable ID guarantee. |
| Live inventory, variant, and size availability lookup | Not implemented; the registered availability stub always returns `unknown`. |
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
10. Unit coverage proving the tool is registered, is unavailable before
    current-turn activation, and produces or consumes required refs correctly
    after activation.

Mutating tools also need explicit user intent, idempotency design, ownership
checks, and retry behavior. Checkout, payment, order, account, and profile-write
tools require a separate confirmation and authorization design before they are
eligible for the shopper-facing agent.
