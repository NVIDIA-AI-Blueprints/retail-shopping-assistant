# Shopper Agent Skill Registry

This registry documents the Deep Agents skills registered for the
shopper-serving assistant. Registration makes a skill eligible for per-turn
activation; it does not mean the skill's complete instructions have been
applied to every turn. Skill names and paths are internal implementation
details. They are for engineers, evaluators, and agent instructions, not
shopper-facing UI copy.

## Current Runtime Boundary

The runtime sources of truth are
`chain_server/src/deepagents_runtime.py::DeepAgentsRuntime._create_agent` and
`chain_server/src/skill_activation.py::ShopperSkillActivationMiddleware`. The
assistant uses a `FilesystemBackend` rooted at `chain_server/skills` in virtual
mode. In the container image, `chain_server/Dockerfile` copies that directory
to `/app/skills`.

At turn setup, the runtime validates the registered `SKILL.md` frontmatter and
reads the complete static files server-side. The activation prompt and enum are
generated from that current validated registry. The runtime intentionally does
not enable Deep Agents' checkpointed `SkillsMiddleware` metadata, which could
be stale for a Redis-backed conversation after a deployment changes skill
frontmatter. The complete contents of only the selected skills are injected
deterministically before the commerce-capable model step. Deep Agents
`read_file` remains available through its filesystem middleware for read-only
skill reference files; the model does not need to read an activated `SKILL.md`
again. Write, edit, list, grep, glob, shell, todo, and general-purpose subagent
tools remain disabled for the shopper harness. Customer profile, cart, catalog,
price, inventory, order, and payment truth must stay in application services,
not skill files.

## Registration And Per-Turn Activation

Every shopper turn uses two model phases inside the same Deep Agents run:

1. The activation phase exposes only the internal
   `activate_shopper_skills_tool`, forces that tool choice, and disables
   parallel tool calls. The model selects the smallest registered skill set
   that semantically covers the complete current intent.
2. The runtime validates the selected names, injects the complete selected
   `SKILL.md` contents into the system context, removes the activation tool, and
   exposes the nine shopper commerce tools for the next model step.

Selection is model-owned semantic interpretation over the current conversation
and skill descriptions, not a deterministic keyword router. Loading and prompt
injection are deterministic once names are selected. A multi-intent turn may
activate more than one skill, but `product-discovery` and `outfit-styling` are
alternative primary procedures and must not be selected together.
`budget-shopping` may accompany the applicable primary procedure only when the
shopper states a budget.

The boundary fails closed. If selection or file loading fails, no shopper
commerce tools are exposed. A commerce call issued in the same model response
as activation is rejected, because activation takes effect only after its tool
result is present in the current turn. An activation from an earlier turn does
not unlock the current turn. The runtime also validates the activation-phase
model response, so provider noncompliance cannot silently terminate the turn
with shopper prose instead of an activation call.

This invariant adds one bounded model step to every turn. The static file load
and injection add no model call. The extra step is the deliberate latency and
model-call tradeoff for ensuring that catalog, cart, policy, and availability
work cannot bypass applicable skill instructions.

After the Deep Agent drafts a response from tool calls, the runtime can run a
configurable grounding boundary over the final shopper-facing text. It accepts
only actual tool-role messages, isolates current-turn evidence with the
server-owned request marker, and supplies prior-turn tool evidence separately.
Prior evidence may support a direct reference to an earlier product, but it
cannot prove that a new search or cart mutation ran. Assistant drafts are never
re-ingested as evidence.

For a search-only turn, each successful search carries the model-authored
semantic query as `SEARCH_DIRECTION_EVIDENCE`. Code renders candidate names,
prices, categories, confirmed filters, and that ranking direction
deterministically. Styling output labels the direction as preference rather than
product fact and nominates the first ranked result, or one first result per
requested role. It makes no separate rationale model call. Other tool-backed
responses use the grounding editor to remove unsupported product claims, surface
guarantees, and internal refs. Grounding is enabled by default and can be
disabled with `GROUNDING_REWRITE_ENABLED=false`; the evidence window is
controlled by `GROUNDING_REWRITE_MAX_EVIDENCE_CHARS`.

Final-response extraction skips tool messages, assistant messages that still
contain tool calls, and internal activation markers. If no shopper-facing text
remains, the runtime emits a safe retry response and records
`incomplete_agent_response` rather than exposing internal content.

Each response also exposes operator-facing diagnostics for selected skill-file
paths, ordered tool calls and arguments, rejected or duplicate calls, and final
termination reason. On graph failure, bounded current-turn assistant/tool
messages are captured before checkpoint cleanup.

## Registered Skills

| Skill | Source | Status | Primary entry modes |
| --- | --- | --- | --- |
| `product-discovery` | `chain_server/skills/shopper/product-discovery/SKILL.md` | Registered | Primary procedure for general search, category browsing, filter-driven discovery without styling intent |
| `outfit-styling` | `chain_server/skills/shopper/outfit-styling/SKILL.md` | Registered | Primary procedure for anchor product, no-anchor styling, cart styling, conversational mid-browse |
| `cart-management` | `chain_server/skills/shopper/cart-management/SKILL.md` | Registered | Explicit cart reads, adds, removals, quantity updates |
| `budget-shopping` | `chain_server/skills/shopper/budget-shopping/SKILL.md` | Registered | Modifier for stated price ceilings, budget bundles, cart budget checks |
| `store-policy-answers` | `chain_server/skills/shopper/store-policy-answers/SKILL.md` | Registered | Returns, shipping, sizing, payment, price matching, gift cards |

## `product-discovery`

Purpose: general product search, browsing, and filter-driven discovery without
a styling request. This is the primary procedure for that intent and is not
combined with `outfit-styling`.

- Uses one focused catalog search for each category scope.
- Semantically maps shopper meaning to the exact taxonomy values and
  non-taxonomy constraint properties generated from active catalog
  capabilities. The runtime validates those structured values; it does not
  infer them from shopper prose.
- Uses at most one advertised category per call. For a broad request that names
  no type, `agent_selected_type` may include all advertised subcategories that
  serve one focused semantic role.
- Reports an explicitly requested product type that has no advertised match.
  `no_direct_catalog_match` uses empty taxonomy and no hard constraints and
  performs no retrieval; the skill does not broaden to a parent, omit the type,
  or search an adjacent type until the shopper accepts that direction. An
  unsupported modifier does not erase an advertised type.
- Treats names as display names and reads product details before asserting
  attributes not present in search evidence.
- Never silently weakens a shopper must-have. An unsupported hard requirement
  directly stated for the product is preserved in
  `unadvertised_requirements` and disclosed before the shopper chooses whether
  to continue as a preference.
- On a partial multi-role search, an unadvertised requirement triggers one
  current-context review. A direct product must-have remains blocked; only a
  requirement inferred from broad season, weather, occasion, or style context
  may be removed for the single repair call.
- Keeps subjective style in semantic direction. Repeating taxonomy plus the
  same hard constraints is a duplicate even when `semantic_query` changes.
- Uses the availability tool rather than treating catalog results as inventory.

## `cart-management`

Purpose: explicit cart reads, additions, removals, and quantity changes.

- Requires explicit mutation intent and tool-provided product or cart-line
  references.
- Reads current cart state before removal or quantity updates.
- Treats mutation results as authoritative and reports partial failures.

## `budget-shopping`

Purpose: modify the applicable discovery or styling procedure when the shopper
states a price ceiling or bundle budget.

- Treats the stated ceiling as a hard search constraint.
- Shows running recommendation costs and uses cart tools for actual cart totals.
- Reports when a complete set cannot fit instead of hiding over-budget options.

## `store-policy-answers`

Purpose: controlled answers for the six supported store-policy topics.

- Reads policy content through the registered policy tool, never model
  knowledge.
- Relays unavailable topics honestly and directs the shopper to the retailer's
  help center.

## `outfit-styling`

Purpose: customer-facing fashion styling that can build, complete, validate,
compare, refine, or budget outfits from product context, cart context, uploaded
images, or mid-browse questions. This is the primary procedure for styling
intent and is not combined with `product-discovery`.

Tool boundary:

- Uses catalog search for grounded product recommendations and substitutions.
- Uses product details only for known `PRODUCT_REF` values.
- Treats remembered refs as same-process, active-snapshot evidence. A restart,
  another replica, cache eviction, or catalog replacement requires a fresh
  search; Redis checkpointing does not preserve this cache.
- Product-detail reads are capped per turn; once the cap is reached the agent
  should stop tool calling and answer from evidence already collected.
- Uses cart read and cart total tools for cart styling and budget checks.
- Cart outfit checks should read cart state and avoid unnecessary catalog search
  for items already in the cart.
- Uses cart mutation tools only when the shopper explicitly asks to add or
  remove items and the normal cart tool preconditions are satisfied.
- Multi-item outfit adds use the batched add tool with selected `PRODUCT_REF`
  values, then report exact added and failed items.
- Styling approval does not imply cart consent; the add scope must match the
  items explicitly included in the shopper's add request.
- Ambiguous add scope should produce one concise clarification, not a guessed
  cart mutation.
- Does not own catalog facts, pricing, inventory, cart state, checkout, or
  profile persistence.

Behavior boundary:

- Supports multi-intent requests such as styling plus budget, styling plus
  cart review, image styling plus search, or comparison plus value judgment.
- Selects exact advertised taxonomy values semantically; examples and fashion
  formulas are not catalog taxonomy. Catalog capabilities also generate the
  allowed non-taxonomy constraint properties. Deterministic code validates the
  selected structured values but does not interpret shopper language.
- Keeps each call to one advertised category and one focused product role. When
  the shopper names no concrete type, `agent_selected_type` may include all
  advertised subcategories that serve that role.
- Reports a requested product type with no advertised match before offering an
  adjacent direction. That path uses empty taxonomy and no hard constraints,
  performs no retrieval, and does not broaden, omit, or silently substitute the
  missing type. An unsupported modifier does not erase an advertised type.
- Preserves a directly stated product must-have that the catalog does not
  advertise in `unadvertised_requirements`. Broad season, weather, occasion,
  or style context does not become a product guarantee unless the shopper says
  it directly; a partial multi-role call gets one review to correct that
  distinction before the loop closes.
- Keeps subjective style and anchor facts as semantic styling context unless the
  current request explicitly applies the same value to the target product.
- Treats normalized taxonomy plus hard constraints as the duplicate-search
  identity; semantic paraphrasing alone cannot justify another retrieval.
- Separates catalog facts from styling inference. Shopper wording is context,
  not product evidence, and material or comfort claims should be attributed
  item by item unless the whole outfit is supported by catalog evidence.
- Treats product names as display names, not attribute evidence. Length, color,
  print, material, fit, care, construction, and group-level claims require
  product-detail evidence.
- Keeps internal `PRODUCT_REF` and `CART_LINE_ID` values out of shopper-facing
  responses.
- Avoids grouping leather, rubber, metal, or generic canvas under natural-fiber
  claims; material summaries should stay item-specific.
- Uses product-detail reads before detailed comparison tables or claims about
  material, dimensions, pockets, closures, care, comfort, or outdoor
  practicality.
- Does not use product-detail reads just to enrich the first no-anchor outfit
  recommendation. Initial outfit building should search by item role and
  synthesize from names, prices, categories, images, and styling fundamentals.
- Keeps initial recommendations lightweight: product name, price, role, and one
  styling reason. Detailed specs require product-detail reads and a shopper need
  for that detail.
- Does not claim tax, transaction-specific shipping fees, delivery estimates,
  or real-time stock status. Controlled shipping policy content comes from the
  policy tool.
- Keeps outdoor-practicality language modest unless the catalog explicitly
  supports stronger claims such as grass/gravel stability, water resistance,
  all-day comfort, or weather safety.
- Avoids category-wide superlatives such as maximum breathability or best grip
  unless all relevant catalog options have been checked with product details.
- Asks at most one concise clarifying question when no anchor, occasion, vibe,
  category, budget, image, or cart context exists.
- Keeps shopper-owned wardrobe items separate from catalog products.
- Does not expose skill names, tool names, entry-mode names, or internal
  routing language in shopper responses.
- Final responses pass through the grounding editor when tool evidence exists;
  the editor sees a customer-safe evidence summary rather than raw catalog
  marketing copy. It may soften or remove unsupported material, comfort,
  outdoor, or internal-reference language but should preserve tool-confirmed
  cart actions.

## Tuning Loop

The outfit behavior tuning surface is
`chain_server/skills/shopper/outfit-styling/SKILL.md`. Shared seasonal framing
lives in `chain_server/skills/shopper/trends-current.md`; it is read-only
reference content, not a registered skill or catalog truth. Its frontmatter and
update log own the refresh date and history.

When changing the skill:

1. Keep the frontmatter `name` stable unless changing runtime behavior on
   purpose.
2. Prefer catalog-agnostic behavior rules over hard-coded product names.
3. Validate the skill file with the skill validator.
4. Run unit tests that assert the skill is registered and that applicable turns
   select and inject it before shopper commerce tools are exposed.
5. Restart or redeploy the chain server so the container/process sees the
   current file.
6. Verify the activation registry reflects current frontmatter and descriptions;
   it is regenerated from current files rather than checkpointed per thread.
7. Run the `style_guide` Challenger/Judge evaluation before treating behavior
   changes as ready.

If a new deployment uses a materially different catalog, regenerate or adjust
catalog-dependent style evaluation fixtures before judging. The skill should
usually remain stable; the eval scenarios and catalog expectations are the
pieces that may need refresh.

## Minimal First, Subagent Later

`outfit-styling` is currently a file-backed Deep Agents skill, not a separate
subagent. That is intentional for the first production slice: the runtime
injects the selected complete skill before the main Deep Agent receives the
shopping tools, and that agent can then perform multi-step tool use while the
skill guides decision boundaries and response style.

Promote styling to a dedicated subagent only if evaluation shows repeated
failures that require private multi-step planning beyond the main agent loop,
or if styling needs its own tool budget, memory policy, or response schema.
