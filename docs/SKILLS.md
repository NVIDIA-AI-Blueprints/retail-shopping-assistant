# Skills

A skill is a markdown file with frontmatter. It declares a `role`, an optional
`exclusive_group`, the tools it grants, and the instructions the model follows
while it is active. The grant is the load-bearing part: it is an authorization
boundary, not a hint. A skill that does not grant `add_cart_items_tool` cannot
reach the cart, whatever the model decides.

Registration makes a skill eligible for a turn; it does not mean its
instructions applied to one. Names and paths are internal, and belong in code,
tests, and agent instructions rather than in anything a shopper sees.

[Assistant Architecture](ASSISTANT_ARCHITECTURE.md) covers how a turn runs.
This page is the registry: what exists, what each grants, and how to change one.

## Registered skills

| Skill | Source | Status | Role | Tools granted | Chosen for |
| --- | --- | --- | --- | --- | --- |
| `product-discovery` | `chain_server/skills/shopper/product-discovery/SKILL.md` | Registered | `primary` / `product_procedure` | Catalog description, search, details, availability, promotions, conversation-product resolution | Search, browsing, and filter-driven discovery with no styling intent |
| `outfit-styling` | `chain_server/skills/shopper/outfit-styling/SKILL.md` | Registered | `primary` / `product_procedure` | Search, details, availability, promotions, conversation-product resolution | Building, completing, comparing, or refining a look |
| `catalog-questions` | `chain_server/skills/shopper/catalog-questions/SKILL.md` | Registered | `primary` / `product_procedure` | Catalog description, search, details | Questions about the shop: cheapest, dearest, what departments exist |
| `cart-management` | `chain_server/skills/shopper/cart-management/SKILL.md` | Registered | `standalone` | Cart read, total, add, remove, update, conversation-product resolution, search | Explicit cart reads and mutations, alone or beside a product procedure |
| `budget-shopping` | `chain_server/skills/shopper/budget-shopping/SKILL.md` | Registered | `modifier` | None | A stated price ceiling or bundle budget |
| `store-policy-answers` | `chain_server/skills/shopper/store-policy-answers/SKILL.md` | Registered | `standalone` | Policy lookup | Returns, shipping, sizing, payment, price matching, gift cards |
| `destination-weather` | `chain_server/skills/shopper/destination-weather/SKILL.md` | Registered | `standalone` | Daily forecast | Conditions at a named place and time, when no product is asked for |

## How one is chosen

The first model step can call only `activate_shopper_skills_tool`. The model
picks the smallest set of skills whose descriptions cover the turn; the runtime
validates the names, injects those files whole, and exposes the union of their
grants to the next step. Selection is semantic model work rather than a keyword
router, but everything after it is deterministic.

Three rules constrain the composition:

- `product-discovery`, `outfit-styling`, and `catalog-questions` share the
  `product_procedure` exclusive group, so exactly one is ever active.
- `budget-shopping` is a modifier and joins a primary only when the shopper
  states a budget. It grants nothing of its own.
- The standalone skills need no primary beside them, and a genuine multi-intent
  turn activates each skill it needs once.

The previous turn's selection is handed to the next activation as a read-only
continuity signal. It does not force routing or satisfy the gate; it exists so
that a terse follow-up inside a styling thread stays with styling.

The boundary fails closed, in each of the ways it could otherwise be slipped:

- A commerce call in the same model response as the activation call is
  rejected, because activation takes effect only once its result exists.
- An activation from an earlier turn does not unlock the current one.
- A call outside the granted union is rejected before its handler with
  `SHOPPER_SKILL_TOOL_NOT_GRANTED`.
- Frontmatter grants and `chain_server/src/tools/policy.py` must agree exactly
  or startup fails, so the two cannot drift apart unnoticed.
- One invalid composition gets a typed correction; a second returns a fixed
  clarification without another model call.

The cost is one bounded model step per turn. That is the deliberate price of
guaranteeing that catalog, cart, policy, availability, and promotion work
cannot bypass the instructions meant to govern it.

## The skills

### `product-discovery`

General search, browsing, and filter-driven discovery where no styling
judgment is wanted.

- One focused search per category scope; a call may carry several scopes, each
  with at most one category.
- Maps the shopper's words onto exact advertised taxonomy and filter values.
  It does not invent them, and repeating a scope is a duplicate even when the
  wording changes.
- Never silently weakens a must-have. A requirement the catalog cannot enforce
  is disclosed before the shopper chooses whether to continue with it as a
  preference.
- Treats result names as display names, and reads product details before
  asserting any attribute search did not return.
- Asks the availability and promotions tools rather than reading stock or sale
  status out of catalog results.

### `outfit-styling`

Fashion judgment for building, completing, comparing, balancing, or refining a
look. It stays primary through an active styling thread, including terse
follow-ups that lean on an established anchor.

- Preserves accepted anchors and changes only the piece or quality asked about.
- Coordinates colour, proportion, silhouette, formality, occasion, and texture,
  and ties each candidate back to the anchor or the outfit goal.
- Confirmed anchor attributes guide coordination but do not become hard
  requirements on a complementary piece unless the shopper asks for a match.
- Keeps product facts separate from styling judgment; catalog presence is never
  stock or sale status.
- Cart and budget responsibilities stay with the skills that own them. Styling
  honours a co-active budget and may treat confirmed cart lines as anchors, but
  directs no cart reads or mutations.

### `catalog-questions`

Questions about the shop rather than about a product: the dearest or cheapest
thing, whether anything falls in a price range, what departments exist and how
much each holds.

`describe_catalog_tool` answers these from what the catalog publishes, and the
skill's one rule is that a fact about the shop comes from there and never from
the results of a single search. The failure it exists to prevent: "the most
expensive item in the catalog is the Quintessence Zippered Crossbody Bag at
$199.99", said in a shop that reaches $269.99, because bags were the only
department searched.

A superlative is therefore two steps and ends with the product — the published
shape says which category reaches the bound, then a search of that category at
that bound names the item. A range the shop does not reach is answered from the
published floor without searching at all. It hands over to `product-discovery`
the moment the shopper narrows to a kind of product.

### `cart-management`

Explicit cart reads, additions, removals, and quantity changes.

- Requires explicit mutation intent and tool-provided product or cart-line
  references; a styling approval is not an instruction to add.
- Reads current cart state before a removal or a quantity change.
- Resolves an earlier product before an add only when it is absent from this
  turn's evidence. A missing or ambiguous match authorises nothing.
- Can search for a named product nothing has shown yet, so an add needs no
  product procedure beside it.
- Treats mutation results as authoritative and reports partial failures as
  partial.

### `budget-shopping`

Modifies whichever primary procedure is active when the shopper states a price
ceiling or a bundle budget. It grants no tools of its own.

- Treats the stated ceiling as a hard search constraint.
- Shows running recommendation costs; activate `cart-management` alongside it
  when the turn needs a real cart total.
- Says when a complete set cannot fit rather than quietly hiding the options
  that do not.

### `store-policy-answers`

Controlled answers for the six supported policy topics, read through the policy
tool and never from model knowledge. An unsupported or unconfigured topic is
relayed honestly, with a pointer to the retailer's help centre.

### `destination-weather`

Conditions at a place and time when the turn asks for no product. It is
standalone rather than part of a product procedure because the grant decides
which turns can fetch a forecast at all: while `outfit-styling` held it alone,
"going to Cancun next week, what's the weather like" selected no styling
procedure and so could not see the tool.

- Fetches, then reports the numbers with the city and dates they cover.
- Never describes weather it did not fetch. Given no date, a window past the
  horizon, or a region with no single answer, it names the one thing it is
  missing and asks.
- Granting one tool keeps a weather-only turn far cheaper than a product one.
- Selected beside `outfit-styling` when the turn also asks what to wear or pack
  there; it is the only skill that grants the forecast tool.

## Changing a skill

1. Keep the frontmatter `name` stable unless you mean to change runtime
   behaviour.
2. Change `role`, `exclusive_group`, or `tools_granted` in the same commit as
   `chain_server/src/tools/policy.py`. Startup validation rejects drift in
   either direction, so a half-change fails the deployment rather than the
   review.
3. Prefer catalog-agnostic behaviour rules over hard-coded product names.
4. Restart or redeploy the chain server. The activation registry is
   regenerated from the current files rather than checkpointed, so a running
   process will not pick the change up on its own.
5. Run the skill and activation contract tests, then the smallest affected
   multi-turn scenario and its targeted Judge. Save the full suite and the
   broad Judge cohort for release readiness.

A materially different catalog needs its catalog-dependent style fixtures
regenerated before judging. The skill itself should usually stay stable; the
scenarios and catalog expectations are what need the refresh.

Seasonal framing lives inline in `outfit-styling` rather than in a reference
file, because the shopper harness has no filesystem tools and a pointer to one
would be unreachable. Trend guidance is never catalog truth, and the shopper's
own stated preferences outrank it.

`outfit-styling` is a file-backed skill rather than a subagent, deliberately:
the runtime injects it before the agent receives the shopping tools, and the
agent can then run multi-step tool use under its guidance. Promote it to a
dedicated subagent only if evaluation shows failures that need private
multi-step planning, or if styling needs its own tool budget, memory policy, or
response schema.
