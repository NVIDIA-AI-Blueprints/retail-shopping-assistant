# Shopper Agent Skill Registry

This registry documents the Deep Agents skills registered for the
shopper-serving assistant. Registration makes a skill eligible for per-turn
activation; it does not mean the skill's complete instructions have been
applied to every turn. Skill names and paths are internal implementation
details. They are for engineers, evaluators, and agent instructions, not
shopper-facing UI copy.

## Current Runtime Boundary

The runtime sources of truth are
`chain_server/src/tool_policy.py` for registry and immutable execution policy,
`chain_server/src/skill_activation.py::ShopperSkillActivationMiddleware` for
per-turn binding, and
`chain_server/src/deepagents_runtime.py::DeepAgentsRuntime._create_agent` for
registered wrapper wiring. The
assistant uses a `FilesystemBackend` rooted at `chain_server/skills` in virtual
mode. In the container image, `chain_server/Dockerfile` copies that directory
to `/app/skills`.

At turn setup, the runtime validates each registered `SKILL.md` frontmatter
`name`, `description`, `response_guidance`, `role`, optional
`exclusive_group`, and `tools_granted`, then reads the complete static files
server-side. Every frontmatter skill/tool pair must match the independent
immutable tool policy exactly; startup fails on drift. `description` drives
semantic activation;
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
   exposes only the union of those skills' `tools_granted` for the next model
   step. Every app-owned shopping dispatch independently rechecks both that
   union and the immutable policy before invoking its handler.

Selection is model-owned semantic interpretation over the current conversation
and skill descriptions, not a deterministic keyword router. Loading and prompt
injection are deterministic once names are selected. The selected names are
persisted with the durable terminal turn and supplied to the next activation
prompt as a read-only continuity signal. The model keeps them when the shopper
continues the task and may change them when the task changes; the signal does
not force routing, inject a skill, or satisfy the current turn's activation
gate. A multi-intent turn may
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
not unlock the current turn. After activation, an ungranted app-owned shopping
call is rejected before its handler with `SHOPPER_SKILL_TOOL_NOT_GRANTED` and
the `skill_tool_not_granted` diagnostic reason. The runtime also validates the
activation-phase model response, so provider noncompliance cannot silently
terminate the turn with shopper prose instead of an activation call.

This invariant normally adds one bounded model step to every turn. An invalid
composition may add one corrective model step; a second invalid composition
returns the deterministic clarification without another model call. The static
file load and injection add no model call. The activation step is the deliberate
latency and model-call tradeoff for ensuring that catalog, cart, policy,
availability, and promotions work cannot bypass applicable skill instructions.

Catalog repair is not another skill-selection phase. The server keys repairs by
the full normalized, model-authored `requested_product_type` phrase. It does not
reconstruct alternatives, negation, ordering, or comparisons from shopper
prose. Each scope receives one total repair. A schema correction
or a fresh constraint-provenance review
can consume that shared budget; constraint feedback returned by an in-flight
schema repair closes the loop for synthesis rather than opening another repair.
The isolated request receives the capability-derived typed
`search_catalog_tool`, compact server-generated Catalog capabilities, the
current shopper message, bounded sanitized validator feedback in a separate
Human data message, and the complete active shopper-skill instructions. Echoed
rejected arguments are stripped and quoted text is labeled as data. Only
`search_catalog_tool` is exposed, but tool choice remains automatic so the
model can signal a clarification by returning no tool call. The server uses that
marker only as branch/control state, discards the model prose, and emits
`Could you clarify the product type or requirement you want me to use?`. The
base runtime prompt, invalid AI/tool history, and earlier conversation history
are absent.
For a native tool-transport failure, the requested scope is locked only when
current or recent shopper text grounds it; an ungrounded model-generated scope
may be corrected. Middleware never restores or rewrites taxonomy, constraints,
requested type, or search mode. It may restore only structural
`scope_complete`, with bounded `restored_fields` diagnostics listing that name.
When strict handler validation has already accepted the advertised constraints,
its feedback may include that finite object and the next call is rejected if it
drifts; the handler does not overwrite it.
A changed shopper-grounded scope is stripped and recorded with reason
`repair_scope_changed`. Malformed or nonempty free-form
`unadvertised_requirements` arguments close a native schema-invalid call without
repair. A
schema-valid, genuinely open request retains the bounded
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
The runtime gives the active skill one final tools-disabled synthesis step, then
grounds that draft against tool-role evidence. Static `response_guidance` and
the pre-retrieval guidance are used by the deterministic fallback when the
draft or editor is unavailable. If the requested outcome depends on a
functional product property absent from evidence, final grounding explicitly
marks it unconfirmed and presents the candidates as the closest catalog or
styling direction rather than as proven suitable; deterministic fallback ends
with the same generic disclosure. Candidate results, taxonomy, filters, semantic
query, and drafts are not turned into evidence after retrieval. Before fallback
guidance becomes shopper-facing text, a narrow scrub
replaces documented unsupported outdoor/weather guarantee terms with neutral
selected-role guidance without changing search semantics, taxonomy, hard
constraints, or retrieval. Covered forms include outdoor-surface or
outdoor-walking claims and constructions such as "handle rain," "work well for
outdoor surfaces," or "stay secure for outdoor walking," plus `wet conditions`
and "works well in wet weather/conditions."
Deterministic fallback code then renders every candidate name, price, category,
and search-scoped confirmed-filter group. For multi-role results, it groups each
guidance sentence with the products returned by that same search and
deduplicates candidates by `product_ref`, not display name. Mixed-outcome turns
preserve successful product groups when another scope has an unsupported
requirement. A fixed unsupported-requirement response is used only when that
rejection is the sole current-turn business-tool outcome.
An incomplete successful scope receives a neutral
offer to continue with the next requested piece or search scope. Scoped
zero-result evidence retains its exact advertised taxonomy and filters and
cannot support a broader absence claim. Other
tool-backed responses use the grounding editor to remove unsupported product
claims, surface guarantees, and internal refs.
The editor receives only the remaining shared model-stage deadline. A timeout
finalizes the turn as failed with `grounding_timeout`; search-only evidence uses
deterministic catalog rendering, while other turns receive a fixed
retry/cart-check response instead of the unverified draft. Editor errors and
empty or whitespace-only output use the same fail-closed response with
`grounding_error`.
Grounding is enabled by default and can be disabled with
`GROUNDING_REWRITE_ENABLED=false`; the evidence window is controlled by
`GROUNDING_REWRITE_MAX_EVIDENCE_CHARS`.

Final-response extraction skips tool messages, assistant messages that still
contain tool calls, and internal activation markers. If no shopper-facing text
remains, the runtime emits a safe retry response and records
`incomplete_agent_response` rather than exposing internal content.

The runtime also retains operator-facing diagnostics for selected skill-file
paths, ordered tool calls and arguments, rejected or duplicate calls, final
termination reason, bounded product evidence with a truncation flag, and
bounded `catalog_scope_outcomes` for `zero_results`. Public query responses
return `{}` for this field by default; trusted operator/evaluation deployments
must explicitly set `EXPOSE_AGENT_DIAGNOSTICS=true`. On graph failure, bounded
current-turn assistant/tool messages are captured before checkpoint cleanup.
The Judge retains only product evidence/truncation and those catalog scope
outcomes from diagnostics.

## Registered Skills

| Skill | Source | Status | Role | Tools granted | Primary entry modes |
| --- | --- | --- | --- | --- | --- |
| `product-discovery` | `chain_server/skills/shopper/product-discovery/SKILL.md` | Registered | `primary` / `product_procedure` | Search, details, availability, promotions, same-conversation product resolution | General search, category browsing, filter-driven discovery without styling intent |
| `outfit-styling` | `chain_server/skills/shopper/outfit-styling/SKILL.md` | Registered | `primary` / `product_procedure` | Search, details, availability, promotions, same-conversation product resolution | Build, complete, or refine a look; coordinate a requested piece with an anchor; use cart evidence only when cart management is also active |
| `cart-management` | `chain_server/skills/shopper/cart-management/SKILL.md` | Registered | `standalone` | Cart read, total, add, remove, update, same-conversation product resolution | Explicit cart reads and mutations, alone or beside a product procedure |
| `budget-shopping` | `chain_server/skills/shopper/budget-shopping/SKILL.md` | Registered | `modifier` | None | Stated price ceilings and budget bundles; combine with cart management for cart-total checks |
| `catalog-questions` | `chain_server/skills/shopper/catalog-questions/SKILL.md` | Registered | `primary` | Questions about the shop | The most or least expensive thing, whether anything falls in a price range, what departments exist |
| `store-policy-answers` | `chain_server/skills/shopper/store-policy-answers/SKILL.md` | Registered | `standalone` | Policy lookup | Returns, shipping, sizing, payment, price matching, and gift cards |

## `product-discovery`

Purpose: general product search, browsing, and filter-driven discovery without
a styling request. This is the primary procedure for that intent and is not
combined with `outfit-styling`.

- Uses one focused catalog search for each category scope.
- Semantically maps shopper meaning to the exact taxonomy values and
  non-taxonomy constraint properties generated from active catalog
  capabilities. The flat model-facing schema contains `semantic_query`,
  `shopper_guidance`, `requested_product_type`, `taxonomy`,
  `required_constraints`, `scope_complete`, and optional `search_mode`. Exact
  taxonomy and hard-filter values come from Catalog capabilities. The schema
  has no model-authored taxonomy relationship, clarification branch, or
  catalog-absence result. The handler applies the strict runtime semantic model,
  so cross-field failures reach capability-aware validation. Exact advertised
  category/subcategory coherence is owned by the capability contract.
- Supplies required `requested_product_type` provenance on every text search:
  the shortest product noun or true umbrella from the current turn or direct
  antecedent, excluding color, material, fit, occasion, weather, and style
  modifiers. For a genuinely open role, it is the one advertised subcategory
  selected for that role. Image-only
  search uses `null`; the field is not taxonomy or ranking text.
- Authors required, nonempty `shopper_guidance` before each taxonomy-scoped
  retrieval under this active skill: one concise product-agnostic sentence
  connecting the selected role to the shopper's stated goal or direct
  antecedent. Image-only search requires empty guidance.
  Guidance cannot name candidates, assert product attributes, or expose search
  mechanics.
- Uses at most one advertised category per call. For a broad request that names
  no type, the model selects exactly one advertised subcategory as the focused
  starting role and names it in `requested_product_type`. That open-role path is
  forbidden when the shopper named the role's type, including an alternative,
  confirmation, comparison, or follow-up. Invalid open-role provenance is
  rejected rather than silently reinterpreted.
- When a shopper-named type is not separately advertised, permits one
  model-selected faithful advertised parent category while retaining the type
  in `requested_product_type` and semantic direction. Results remain closest
  alternatives under their actual catalog categories. If neither a direct type
  nor one faithful parent can be selected, clarification performs no retrieval
  and makes no catalog-absence claim. An unsupported modifier does not erase an
  advertised type.
- Separates request lanes: unresolved product type to clarification, advertised
  type plus unenforceable must-have to
  `unadvertised_requirements`, and preference or styling context to
  `semantic_query`. Product types never enter the requirement lane.
- Treats names as display names and reads product details before asserting
  attributes not present in search evidence.
- Never silently weakens a shopper must-have. An unsupported hard requirement
  directly stated for the product is preserved in
  `unadvertised_requirements` and disclosed before the shopper chooses whether
  to continue as a preference.
- Every unadvertised requirement on a shopper-stated product scope fails closed
  before retrieval, including when the model uses a synonym rather than the
  shopper's exact wording. The bounded constraint review is reserved for a
  proposed inferred requirement on a genuinely open role when its shared repair
  budget remains. It freezes requested type, taxonomy, completion state,
  `search_mode`, and every advertised hard
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
- Uses the promotions tool for explicit sale or promotion status rather than
  treating catalog search or price as markdown evidence.
- Uses the historical resolver only when a needed product is not already
  established in the current turn. A unique exact match becomes request-local
  evidence; missing or ambiguous results require clarification rather than a
  substitute search. Batch all needed references because the runtime permits
  this resolver at most once per turn.

## `cart-management`

Purpose: explicit cart reads, additions, removals, and quantity changes.

- Its instructions require explicit mutation intent and tool-provided product
  or cart-line references. Slice 0 enforces the skill grant and refs, but
  server-owned current-turn mutation intent authorization remains a later
  slice.
- Reads current cart state before removal or quantity updates.
- Resolves an earlier presented product before an add only when the product is
  absent from current-turn evidence. Missing or ambiguous matches do not
  authorize a mutation.
- Treats mutation results as authoritative and reports partial failures.

## `budget-shopping`

Purpose: modify the applicable discovery or styling procedure when the shopper
states a price ceiling or bundle budget.

- Treats the stated ceiling as a hard search constraint.
- Shows running recommendation costs. Activate `cart-management` alongside it
  when the turn also needs an actual cart total; this modifier grants no tools.
- Reports when a complete set cannot fit instead of hiding over-budget options.

## `store-policy-answers`

Purpose: controlled answers for the six supported store-policy topics.

- Reads policy content through the registered policy tool, never model
  knowledge.
- Relays unavailable topics honestly and directs the shopper to the retailer's
  help center.

## `outfit-styling`

Purpose: customer-facing fashion judgment for building, completing, comparing,
balancing, or refining a look. It remains the primary procedure through an
active styling thread, including terse follow-ups that rely on an established
anchor or outfit goal. It is not combined with `product-discovery`.

The skill owns:

- deciding whether to proceed or ask one concise styling clarification;
- preserving accepted anchors and changing only the requested piece or quality;
- coordinating color, proportion, silhouette, formality, occasion, and texture;
- connecting each grounded candidate to the anchor or outfit goal;
- keeping product facts separate from styling judgment; and
- using the seasonal trend reference only as optional framing.

The skill grants catalog search, product details, availability, promotions, and
typed same-conversation product resolution. Search results support name, price,
category, and image availability; other product attributes require detail
evidence. Catalog presence is never treated as stock or sale status.

For a named follow-up role, the skill keeps the anchor as context and searches
only that role. Confirmed anchor attributes guide coordination, but do not
become requirements on a complementary piece unless the shopper explicitly
asks for the same or a matching value. When a needed earlier product is absent
from current-turn evidence, the skill can submit exact descriptors from the
read-only historical-product index. The durable resolver returns 0/1/many;
missing and ambiguous references require one clarification, and only a unique
match can authorize a downstream tool.

Cart and budget responsibilities stay with their owning skills. When
`cart-management` is co-active, confirmed cart lines may be styling anchors;
`outfit-styling` does not direct cart reads or mutations. When
`budget-shopping` is co-active, styling honors the ceiling using confirmed
prices; it does not own cart totals.

The skill does not own catalog taxonomy, tool transport fields, repair loops,
runtime response assembly, cart state, policy, memory, inventory, or checkout.

## Tuning Loop

The outfit behavior tuning surface is
`chain_server/skills/shopper/outfit-styling/SKILL.md`. Shared seasonal framing
lives in `chain_server/skills/shopper/trends-current.md`; it is read-only
reference content, not a registered skill or catalog truth. Its frontmatter and
update log own the refresh date and history.

When changing the skill:

1. Keep the frontmatter `name` stable unless changing runtime behavior on
   purpose.
2. Keep `role`, optional `exclusive_group`, and `tools_granted` aligned with
   `tool_policy.py`; any grant change must update both sources in one change.
3. Prefer catalog-agnostic behavior rules over hard-coded product names.
4. Validate the skill file and exact policy/grant pairs.
5. Run unit tests that assert the skill is registered, applicable turns select
   and inject it, only its grant union is model-visible, and direct dispatch of
   an ungranted tool is rejected.
6. Restart or redeploy the chain server so the container/process sees the
   current file.
7. Verify the activation registry reflects current frontmatter and descriptions;
   it is regenerated from current files rather than checkpointed per thread.
8. Run the focused skill and activation contract tests, then the smallest
   affected scripted multi-turn styling scenario and its targeted Judge.
   Reserve the complete suite and broad Judge cohort for release readiness.

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


## `catalog-questions`

Asks about the catalog rather than about a product: the most or least expensive
thing, whether anything falls in a price range, what departments exist and how
much each holds.

`describe_catalog_tool` answers them from what the catalog publishes -- product
counts and price ranges per category -- and the skill's one rule is that a fact
about the shop comes from there and never from the results of one search. The
failure it exists for: "the most expensive item in the catalog is the
Quintessence Zippered Crossbody Bag at $199.99", said in a shop that reaches
$269.99, because bags were the only department searched.

A superlative is two steps and ends with the product: the shape says which
category reaches the bound, a search of that category at that bound names the
item. A range the shop does not reach is answered from the published floor
without searching at all.

It hands over to `product-discovery` the moment the shopper narrows to a kind
of product.
