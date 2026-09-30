# Tools

What the shopper-serving agent can call. These names are internal: they belong
in code, tests, and agent instructions, never in a reply.

Two separate things decide whether a call runs. A tool exists at all only if it
is in the runtime's `create_deep_agent(..., tools=[...])` registration. It is
callable on a given turn only if one of that turn's activated skills grants it,
and that grant is rechecked against the immutable policy in
`chain_server/src/tools/policy.py` before any handler runs. Skill frontmatter
and that policy must agree exactly, or startup fails.

[Assistant Architecture](ASSISTANT_ARCHITECTURE.md) covers how a turn runs and
where these calls sit inside it; [Skills](SKILLS.md) covers who grants what.

## The registry

| Tool | Risk class | Source of truth | Status |
| --- | --- | --- | --- |
| `activate_shopper_skills_tool` | `internal_control` | Validated static shopper-skill registry | Registered; required first step |
| `describe_catalog_tool` | `read_only_catalog` | Published catalog capabilities | Registered |
| `search_catalog_tool` | `read_only_catalog` | Catalog retriever | Registered |
| `get_product_details_tool` | `read_only_catalog` | Active catalog snapshot; request-local evidence authorizes the ref | Registered |
| `resolve_conversation_products_tool` | `read_only_conversation` | Durable same-conversation presented-product events | Registered |
| `get_cart_tool` | `read_only_cart` | Memory cart service | Registered |
| `view_cart_total_tool` | `computed_read_cart` | Memory cart service plus cached line prices | Registered |
| `add_cart_items_tool` | `mutating_cart` | Memory cart service | Registered |
| `remove_cart_item_tool` | `mutating_cart` | Memory cart service | Registered |
| `update_cart_items_tool` | `mutating_cart` | Memory cart service | Registered |
| `get_store_policy_tool` | `read_only_policy` | Operator-managed static policy file | Registered |
| `check_product_availability_tool` | `read_only_catalog` | **An application stub. No live inventory source** | Registered |
| `check_active_promotions_tool` | `read_only_promotions` | **An application stub. No live promotions source** | Registered |
| `get_weather_forecast_tool` | `read_only_weather` | Visual Crossing Timeline; off by default, and returns a typed failure when disabled, unconfigured, or outside the ~15-day horizon | Registered |

## Risk classes

A review taxonomy rather than something the runtime enforces — the enforcement
is the grant plus the immutable policy.

| Class | Meaning | Rules |
| --- | --- | --- |
| `internal_control` | Selects static instructions; reads and mutates no commerce state | Forced once at turn start; cannot be batched with commerce execution |
| `read_only_catalog` | Reads catalog-domain data, or reports its boundary, without touching shopper state | Granted by the product procedures |
| `read_only_conversation` | Resolves typed refs against products actually presented in this conversation | Zero or many matches require clarification and authorize nothing |
| `read_only_cart` | Reads the authoritative cart for the scoped `cart_id` | Cart management only |
| `computed_read_cart` | Computes over authoritative cart reads | Cart management only; arithmetic happens in code, not model prose |
| `mutating_cart` | Changes cart contents | Requires the cart-management grant, valid refs, and service-side success |
| `read_only_policy` | Reads operator-managed policy content | Never substitute model knowledge for an absent topic |
| `read_only_promotions` | Reports the deployment's promotion signal | Catalog results and price are never markdown evidence |
| `future_high_risk` | Checkout, payment, orders, account changes | Not registered. Needs stronger auth, ownership checks, and a confirmation policy first |

## Skill access matrix

The per-skill authorization contract. The model sees only the union of the
current turn's grants, and every dispatch rechecks it against the immutable
policy, so an ungranted call fails before its handler.

| Skill | Tools granted |
| --- | --- |
| `product-discovery` | `describe_catalog_tool`, `search_catalog_tool`, `get_product_details_tool`, `check_product_availability_tool`, `check_active_promotions_tool`, `resolve_conversation_products_tool` |
| `outfit-styling` | `search_catalog_tool`, `get_product_details_tool`, `check_product_availability_tool`, `check_active_promotions_tool`, `resolve_conversation_products_tool` |
| `catalog-questions` | `describe_catalog_tool`, `search_catalog_tool`, `get_product_details_tool` |
| `cart-management` | `get_cart_tool`, `view_cart_total_tool`, `add_cart_items_tool`, `remove_cart_item_tool`, `update_cart_items_tool`, `resolve_conversation_products_tool`, `search_catalog_tool` |
| `budget-shopping` | None (`tools_granted: []`) |
| `store-policy-answers` | `get_store_policy_tool` |
| `destination-weather` | `get_weather_forecast_tool` |

A multi-intent turn selects every skill it needs in the one activation step.
"Find shoes and a bag for this outfit under $120 and add the shoes" is outfit
styling, budget shopping, and cart management — not product discovery.

## What each tool does

### `activate_shopper_skills_tool`

The forced first step. Takes `skill_names` from the registered enum, and the
model should pick the smallest set that covers the turn. The runtime injects
those files whole and exposes only their combined grants to the next step. It
reads and mutates nothing.

Invalid or unknown skill content exposes no commerce tools at all. A model
response that tries to answer without activating fails the turn with
`skill_activation_failed`, and a commerce call in the same response is rejected
with `skill_activation_required`.

### `describe_catalog_tool`

What the shop holds: total product count, and each category with its own count,
price range, and subcategories. It takes no arguments and searches nothing, so
it cannot become a second retrieval path.

It exposes the capabilities that already build `search_catalog_tool`'s schema
and were otherwise unreadable by the assistant. Without it, a question about
the catalog could only be answered from whatever one search happened to
return — which is how a $199.99 bag came to be called the most expensive item
in a catalog reaching $269.99.

### `search_catalog_tool`

Product discovery over the catalog. Requires product text or an attached image.

| Field | Meaning |
| --- | --- |
| `semantic_query` | Soft ranking direction only; it cannot change the selected taxonomy |
| `shopper_guidance` | One product-agnostic sentence authored before retrieval; empty only for image-only search |
| `requested_product_type` | The shortest product noun or true umbrella from this turn or its direct antecedent; `null` only for image-only search |
| `taxonomy` | Capability-derived category and subcategory; at most one category per call |
| `required_constraints` | Capability-derived hard filters, plus the explicit `unadvertised_requirements` lane |
| `scope_complete` | Whether this search finishes the request |
| `search_mode` | Optional, from the advertised modes |

Every value comes from the published capabilities; there is no alias table and
no keyword router. The rules worth knowing:

- **A must-have the schema cannot enforce fails closed before retrieval.** It
  goes in `unadvertised_requirements` and stops the search, rather than being
  quietly demoted to a preference. This holds even when the model paraphrases
  the shopper's wording.
- **One normalized taxonomy-plus-constraints scope runs once per turn.**
  Repeating it returns `STOP_TOOL_USE` even if `semantic_query` is reworded.
  Three searches per turn, and one repair attempt per scope.
- **A zero-result search proves absence only for its own exact scope,** never
  for another product type or the catalog as a whole.
- A shopper-named type that is not separately advertised may be searched under
  one faithful parent category, with results presented as closest
  alternatives. If no direct type and no faithful parent fit, the assistant
  clarifies instead of calling this tool.

Returns candidates as `PRODUCT_REF` plus name, category, price, and image where
available; those refs become this turn's evidence for detail, availability, and
cart-add calls. Products actually presented to the shopper become a durable
`candidate_set_presented` event; candidates that were not presented do not.

Granted by `product-discovery`, `outfit-styling`, and `cart-management`.
Returned IDs are the feed's own `record_id` values, which the current feed does
not guarantee across catalog replacements.

### `get_product_details_tool`

Deeper facts for a `product_ref` already in this turn's evidence: the fields
the catalog sidecar marks `detail`, including exact composition or care text
where present. It infers nothing that is missing.

Use it before asserting any attribute that search did not return. A per-turn
read cap applies, and at the cap the tool returns `STOP_TOOL_USE` so the agent
answers from what it already read. An unknown ref returns guidance to search
first rather than a guess.

### `resolve_conversation_products_tool`

Resolves products the shopper refers to from earlier product cards in the same
conversation — "the blue one you showed me". Takes a typed batch of exact
selectors (`product_ref`, `display_name`, `category`, `turn_sequence`,
`candidate_set_id`, optional `ordinal`).

Matching is exact after trimming and case normalization; there is no fuzzy or
semantic matching, and it is not a browse tool. Each descriptor comes back
`resolved`, `ambiguous`, or `not_found`, and **only a unique resolution becomes
usable evidence** — ambiguity authorizes nothing and requires clarification.

The runtime permits one batched call per turn, so batch every reference
together; a second call returns `STOP_TOOL_USE` without contacting the memory
service.

### `get_cart_tool`

Reads the authoritative cart. The model supplies nothing; the runtime scopes
the identity. Returns lines as `CART_LINE_ID`, quantity, display name, and
cached unit price.

### `view_cart_total_tool`

Itemized line totals and a subtotal, computed in code from cached line prices,
noting any line whose price is unavailable. It does not reprice against the
live catalog, discounts, tax, shipping, or inventory.

### `add_cart_items_tool`

Adds one or more products in a single call: `product_ref`, `quantity`, and
`expected_display_name` per item. Requires an explicit request to add — a
styling approval is not one — and every ref must already be in this turn's
evidence.

Two guards matter. When the request names products, refs outside that named set
are blocked before any mutation, so a stale remembered ref cannot add something
the shopper did not ask for; and `expected_display_name` must resolve to the
selected ref. Each mutation commits with an owner-scoped idempotency record, so
a retry replays rather than double-adding. Partial success is allowed and must
be reported as partial.

### `remove_cart_item_tool`

Removes quantity from an explicit `CART_LINE_ID`, which must come from a cart
read — never guessed from a product name. The result returns to the model
rather than the shopper, so a compound request like "swap this for that and
tell me the new total" can finish its calls first.

### `update_cart_items_tool`

Changes one line's quantity, its size, or both. `quantity` is the total to end
up with.

A size change is the interesting case: the cart has no size-change operation,
because a size is a separate line. The tool adds the new size and then removes
the old one, **in that order**, so a failure between the two leaves the shopper
an extra line rather than nothing, and the result names the line still to be
removed.

It refuses `quantity: 0` and points at `remove_cart_item_tool`, and refuses a
size the catalog does not sell for that product, naming the sizes it does.

### `get_store_policy_tool`

Reads one of `returns`, `shipping`, `sizing`, `payment`, `price_match`, or
`gift_cards` from operator-managed YAML. Policy questions must use this rather
than model knowledge.

The bundled file is a template of placeholders with `configured: false`, so a
default deployment returns `policy_not_configured` rather than presenting
sample text as real policy. The file sits outside the agent-readable skill
backend and is reachable only through this tool.

### `check_product_availability_tool`

**A stub. It always answers "available".** It makes no external call and there
is no inventory system behind it. Given a known `product_ref` it reports
general availability, confirms a requested size for `apparel` and `footwear`,
and treats every other category as one-size.

It exists so that availability is a question the agent must *ask*, rather than
something it infers from a product appearing in search results. Wiring real
inventory means replacing the body, not the contract.

### `check_active_promotions_tool`

**A stub. It always answers that no promotion is active.** No inputs, no
external call. It exists for the same reason as availability: so that sale
status is never inferred from a price or a search result.

### `get_weather_forecast_tool`

Registered only when `WEATHER_ENABLED=true`, and granted to
`destination-weather` alone. Takes a place the shopper named in the
conversation, plus one ISO date or a complete inclusive range of at most 15
days; with no date it asks rather than assuming today.

It rejects relative phrases, partial or mixed date modes, places the shopper
did not name, and requests for historical or long-range statistical data. Typed
failures separate invalid input, disabled or unconfigured state, an unresolved
place, out-of-horizon dates, auth, rate limits, timeouts, and bad provider
responses — without disclosing the key, the prepared URL, the saved ZIP, the
resolved location, or the provider body.

The first adapter calls the
[Visual Crossing Timeline API](https://www.visualcrossing.com/resources/documentation/weather-api/timeline-weather-api/)
over the existing HTTP dependency; no vendor SDK or MCP server is involved. A
reply that uses a forecast must carry the provider attribution and its link.
That reply is stored with the conversation like any other, so check the
[Visual Crossing storage and sharing terms](https://www.visualcrossing.com/weather-service-terms/)
against your deployment before enabling it. Disabled, the tool makes no request
during startup, health checks, or turns.

## Extension points

This blueprint is a reference implementation, so the surfaces a retailer needs
are already in place even where the system behind them is yours to supply. Each
row below is a deliberate seam: the tool contract, the risk class, and the
skill grants are settled, so connecting your own system means replacing a body
rather than designing a new surface.

| Capability | How to plug in |
| --- | --- |
| Live inventory, variant, and size availability | The registered availability tool holds the contract and makes no I/O. Point its body at your inventory service; its callers and grants stay as they are. |
| Live promotions | The same shape as availability. The registered tool already reports that nothing is configured, so a promotions or pricing service drops straight in. |
| `load_customer_persona_tool` | Planned. Personalization arrives through this seam once you have a profile store to read; no registered runtime tool ships today. |
| Checkout, order, payment, address, or account mutation | Deliberately out of scope and classed `future_high_risk`. Read [Adding a tool](#adding-a-tool) before starting: these need a confirmation and authorization design first. |

## Adding a tool

1. A stable internal name that matches this registry, and a docstring that
   states the action in language the model can act on.
2. A typed input schema for anything nontrivial, and compact output carrying
   the refs that follow-on tools need.
3. A named source of truth and owning service, and an explicit risk class.
4. Structured failure behaviour the final response can ground on, with no
   secrets, private hosts, or customer data in tool output.
5. `return_direct=True` only where the output should bypass the final model
   response. Cart reads, totals, and mutations should return to the model so a
   multi-step request can finish first.
6. Matching entries in the granting skills' `tools_granted` frontmatter and in
   `tools/policy.py`. Startup validation rejects drift in either direction.
7. Unit coverage for registration, pre-activation rejection, model-visible
   allow and deny binding, direct-dispatch rejection from an ungranted skill,
   and the refs the tool produces or consumes.

A mutating tool also needs explicit shopper intent, an idempotency design,
ownership checks, and defined retry behaviour. Checkout, payment, order,
account, and profile-write tools need a separate confirmation and authorization
design before they are eligible for the shopper-facing agent at all.
