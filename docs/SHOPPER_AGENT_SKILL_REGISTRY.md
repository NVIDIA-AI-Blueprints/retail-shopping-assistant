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

At turn setup, the runtime validates each registered `SKILL.md` frontmatter
`name`, `description`, and `response_guidance`, then reads the complete static
files server-side. `description` drives semantic activation;
`response_guidance` is reviewed shopper-facing fallback framing when a search
result has no pre-retrieval `shopper_guidance`. The activation prompt and enum
are generated from that current validated registry. The runtime intentionally does
not enable Deep Agents' checkpointed `SkillsMiddleware` metadata, which could
be stale for an existing checkpoint thread if skill files change while the
process remains alive. The complete contents of only the selected skills are
injected deterministically before the commerce-capable model step. Deep Agents
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
injection are deterministic once names are selected. The prior turn's selected
skill names are supplied to the next activation prompt as a read-only continuity
signal. The model keeps them when the shopper continues the task and may change
them when the task changes; the signal does not force routing, inject a skill,
or satisfy the current turn's activation gate. A multi-intent turn may
activate more than one skill, but `product-discovery` and `outfit-styling` are
alternative primary procedures and must not be selected together.
`budget-shopping` may accompany the applicable primary procedure only when the
shopper states a budget. Within an active outfit-building or style-led
single-piece thread, terse item-only follow-ups remain `outfit-styling` tasks.

Runtime taxonomy validation may bind the longest exact advertised suffix in a
modifier-bearing model phrase (`waterproof boots` to `boots`), but it disables
that shortcut for explicit alternatives containing `and`, `or`, `/`, or `&`.
`closed shoes or boots` therefore stays with model-owned alternative or umbrella
reasoning.

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

Catalog repair is not another skill-selection phase. The server keys repairs by
the full normalized `requested_product_type` phrase, with distinct advertised
siblings protected from being treated as the same scope. Each scope receives
one total repair. A schema correction or a fresh constraint-provenance review
can consume that shared budget; constraint feedback returned by an in-flight
schema repair closes the loop for synthesis rather than opening another repair.
The request replaces the normal prompt and history with a concise, schema-generic
system prompt, the current shopper message, and bounded, sanitized validator
feedback in a separate Human data message. Echoed rejected arguments are
stripped and quoted text is labeled as data. Only `search_catalog_tool` is
exposed and forced. The base runtime prompt, invalid AI/tool history, and earlier
conversation history are absent, while the skill gate appends the complete
active shopper-skill instructions to the concise repair prompt.
For a native tool-transport failure, the requested scope is locked only when
current or recent shopper text grounds it; an ungrounded model-generated scope
may be corrected. Independently valid finite locked fields are restored before
execution, with bounded `restored_fields` diagnostics listing names only. A
change to a grounded free-form scope that cannot be reconstructed safely is
still stripped and recorded with reason `repair_scope_changed`. Malformed or
nonempty free-form `unadvertised_requirements` arguments remain outside that
restoration boundary and close a native schema-invalid call without repair. A
schema-valid, genuinely open `agent_selected_type` request retains the bounded
review for a proposed inferred requirement. When a
runtime semantic open-role schema repair removes such a proposal, runtime
replaces its submitted pre-search guidance with neutral generic guidance for the
selected role. A successful or zero-result search that consumes the final
configured search slot records `SEARCH_BUDGET_EXHAUSTED`; the next model step
removes only `search_catalog_tool`. Product-detail, availability, and cart work
plus honest partial synthesis remain available.

After the Deep Agent drafts a response from tool calls, the runtime can run a
configurable grounding boundary over the final shopper-facing text. It accepts
only actual tool-role messages, isolates current-turn evidence with the
server-owned request marker, and supplies prior-turn tool evidence separately.
Prior evidence may support a direct reference to an earlier product, but it
cannot prove that a new search or cart mutation ran. Assistant drafts are never
re-ingested as evidence.

For a completed successful search-only turn, each search carries the
model-authored semantic query as independent internal `SEARCH_DIRECTION_EVIDENCE`
and required pre-retrieval `shopper_guidance` authored under the active skill.
The runtime presents that product-agnostic guidance without another model call;
static `response_guidance` is the fallback. Candidate results, taxonomy,
filters, semantic query, and drafts are not turned into guidance after retrieval.
Before guidance becomes deterministic shopper-facing evidence, a narrow scrub
replaces documented unsupported outdoor/weather guarantee terms with neutral
selected-role guidance without changing search semantics, taxonomy, hard
constraints, or retrieval. Covered forms include outdoor-surface or
outdoor-walking claims and constructions such as "handle rain," "work well for
outdoor surfaces," or "stay secure for outdoor walking," plus `wet conditions`
and "works well in wet weather/conditions."
Deterministic code then renders every candidate name, price, category, and
search-scoped confirmed-filter group. For multi-role results, it groups each
guidance sentence with the products returned by that same search and
deduplicates candidates by `product_ref`, not display name. Mixed-outcome turns
preserve successful product groups when another scope has no direct match or an
unsupported requirement. A fixed no-direct or unsupported canned response is
used only when that rejection is the sole current-turn business-tool outcome.
An incomplete successful scope receives a neutral
offer to continue with the next requested piece or search scope. Scoped
zero-result evidence retains its exact advertised taxonomy and filters and
cannot support a broader absence claim. Other
tool-backed responses use the grounding editor to remove unsupported product
claims, surface guarantees, and internal refs.
Grounding is enabled by default and can be disabled with
`GROUNDING_REWRITE_ENABLED=false`; the evidence window is controlled by
`GROUNDING_REWRITE_MAX_EVIDENCE_CHARS`.

Final-response extraction skips tool messages, assistant messages that still
contain tool calls, and internal activation markers. If no shopper-facing text
remains, the runtime emits a safe retry response and records
`incomplete_agent_response` rather than exposing internal content.

Each response also exposes operator-facing diagnostics for selected skill-file
paths, ordered tool calls and arguments, rejected or duplicate calls, final
termination reason, bounded product evidence with a truncation flag, and
bounded `catalog_scope_outcomes` for `no_direct_catalog_match` and
`zero_results`. On graph failure, bounded current-turn assistant/tool messages
are captured before checkpoint cleanup. The Judge retains only product
evidence/truncation and those catalog scope outcomes from diagnostics.

## Registered Skills

| Skill | Source | Status | Primary entry modes |
| --- | --- | --- | --- |
| `product-discovery` | `chain_server/skills/shopper/product-discovery/SKILL.md` | Registered | Primary procedure for general search, category browsing, filter-driven discovery without styling intent |
| `outfit-styling` | `chain_server/skills/shopper/outfit-styling/SKILL.md` | Registered | Primary procedure for anchor product, no-anchor styling, cart styling, conversational mid-browse, and terse item-only follow-ups within an active outfit task |
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
  capabilities. The model-facing structural transport schema is revalidated by
  the strict runtime semantic model, so cross-field failures reach the
  capability-aware validator and receive exact corrections. The model owns
  `taxonomy_status`; runtime never semantically rewrites it. Exact advertised
  category/subcategory coherence is owned by the capability contract.
- Supplies required `requested_product_type` provenance on every text search:
  the shortest product noun or true umbrella from the current turn or direct
  antecedent, excluding color, material, fit, occasion, weather, and style
  modifiers. For an open `agent_selected_type` role, the runtime derives that
  duplicate provenance from the selected advertised subcategory. Image-only
  search uses `null`; the field is not taxonomy or ranking text.
- Authors required, nonempty `shopper_guidance` before each taxonomy-scoped
  retrieval under this active skill: one concise product-agnostic sentence
  connecting the selected role to the shopper's stated goal or direct
  antecedent. `image_only` and `no_direct_catalog_match` require empty guidance.
  Guidance cannot name candidates, assert product attributes, or expose search
  mechanics.
- Uses at most one advertised category per call. For a broad request that names
  no type, `agent_selected_type` selects exactly one advertised subcategory as
  the focused starting role. It is forbidden for a role whose type the
  shopper named, including an alternative, confirmation, comparison, or
  follow-up. Invalid open-role provenance is rejected rather than silently
  reinterpreted. The runtime keeps `agent_selected_type` while deriving its
  requested-type provenance.
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
- Every unadvertised requirement on a shopper-stated product scope fails closed
  before retrieval, including when the model uses a synonym rather than the
  shopper's exact wording. The bounded constraint review is reserved for a
  proposed inferred requirement on a genuinely open `agent_selected_type` role
  when its shared repair budget remains. It freezes requested type, taxonomy
  status, taxonomy, completion state, `search_mode`, and every advertised hard
  constraint. Within that preserved hard scope, it may correct only the soft
  `semantic_query`, the reviewed unadvertised-requirement lane, and its
  associated guidance; the requirement is either replaced with the shopper's
  shortest exact wording or removed. Exact wording and unresolved provenance
  fail closed; constraint feedback after a schema repair closes the loop for
  synthesis. Removal scrubs product-attribute guidance. A later valid role
  receives its own single repair opportunity after a successful partial search.
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
intent and is not combined with `product-discovery`. It remains primary for
terse item-only follow-ups that rely on the active outfit or style-led
single-piece goal.

Tool boundary:

- Uses catalog search for grounded product recommendations and substitutions.
- Uses product details only for known `PRODUCT_REF` values.
- Treats remembered refs as same-process, active-snapshot evidence. A restart,
  another replica, cache eviction, or catalog replacement requires a fresh
  search; the graph checkpoint does not contain this separate cache.
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
  allowed non-taxonomy constraint properties. A structural agent-facing schema
  transports the selection to the strict runtime semantic validator so
  cross-field defects receive capability-derived feedback. The model owns
  `taxonomy_status`; runtime never semantically rewrites it. Exact advertised
  category/subcategory coherence is capability-owned.
- Supplies required `requested_product_type` product noun/umbrella provenance on
  text searches. It uses the shortest product noun or true umbrella from the
  current request or direct antecedent, excludes color, material, fit, occasion,
  weather, and style modifiers. For `agent_selected_type`, runtime derives this
  duplicate provenance from the selected advertised subcategory. Image-only
  search uses `null`.
- Authors required, nonempty pre-retrieval `shopper_guidance` under this skill
  for taxonomy-scoped searches, connecting the selected role to the current
  outfit goal or direct antecedent without naming candidates, asserting product
  attributes, or exposing search mechanics. `image_only` and
  `no_direct_catalog_match` require empty guidance.
- Keeps each call to one advertised category and one focused product role. When
  the shopper names no concrete type, `agent_selected_type` selects exactly one
  advertised subcategory as the starting role. Runtime retains
  `agent_selected_type` and derives its requested-type provenance. It is
  rejected for a shopper-named scope rather than silently reinterpreted.
- Reports a requested product type with no advertised match before offering an
  adjacent direction. That path uses empty taxonomy and no hard constraints,
  performs no retrieval, and does not broaden, omit, or silently substitute the
  missing type. An unsupported modifier does not erase an advertised type.
- Places a product must-have that the catalog does not advertise in
  `unadvertised_requirements`. Every such requirement on a shopper-stated
  product scope fails closed before retrieval, including when the model uses a
  synonym rather than the shopper's exact wording. The bounded review is
  reserved for a proposed inferred requirement on a genuinely open
  `agent_selected_type` role when its shared repair budget remains. The review
  freezes requested type, taxonomy status, taxonomy, completion state,
  `search_mode`, and every advertised hard constraint. Within that preserved
  hard scope, it may correct only the soft `semantic_query`, the reviewed
  unadvertised-requirement lane, and its associated guidance; the requirement
  is either replaced with the shopper's shortest exact wording or removed.
  Exact wording and unresolved provenance fail closed; constraint feedback after
  a schema repair closes the loop for synthesis. Removal scrubs product-attribute
  guidance. A later valid
  role receives its own single repair opportunity after a successful partial
  search.
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
  recommendation. Initial outfit building should search by item role, author
  product-agnostic `shopper_guidance` before retrieval, and let deterministic
  response code render names, prices, categories, and confirmed filters.
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
- Non-search-only tool-backed drafts pass through the grounding editor, which
  sees a customer-safe evidence summary rather than raw catalog marketing copy.
  Completed successful search-only responses instead use pre-retrieval
  `shopper_guidance` plus deterministic tool-owned facts.

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
