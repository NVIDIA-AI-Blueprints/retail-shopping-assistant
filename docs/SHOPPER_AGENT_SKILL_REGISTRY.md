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
   that semantically covers the complete current intent. Whenever it selects
   `event-context`, the same call must bind
   `event_context_next_question`; that field is omitted otherwise. The model
   selects the latter from current and recent shopper conversation:
   `event_location` only when destination is missing and material,
   `event_venue` only after destination is established when venue or setting is
   missing and material, `event_date` only after destination and any material
   venue are established when live weather is enabled and material and a
   bounded date is neither established nor explicitly unavailable, and `none`
   otherwise. An explicitly shopper-stated outdoor patio, beach, garden,
   rooftop, or open-air setting makes enabled live weather material; with
   destination and that setting but no bounded date, select `event_date`. This
   remains model-owned semantic guidance, not deterministic keyword or alias
   parsing. Only the value from a successfully accepted activation authorizes
   an event-context follow-up.
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
shopper states a budget. `event-context` may accompany only `outfit-styling`,
and is selected whenever event destination or venue context is stated or the
response would otherwise ask about or branch on missing destination or venue
context, or when a supported forecast would materially change event guidance.
Generic occasion advice is not a reason to omit it. It alone grants the
read-only weather tool. Its grant combines additively with the tools granted by
`outfit-styling` and any selected standalone skill. Event context never revokes
or narrows product, cart, or policy tools; only its weather capability may be
hidden or execution-blocked when the forecast prerequisites are not
established. Product work remains owned by the selected primary skill. Within an active
outfit-building or style-led single-piece thread, terse item-only follow-ups
remain `outfit-styling` tasks.

For event-context turns, the successful activation result and the
`outfit-styling` procedure repeat the additive boundary also advertised by
catalog search. A reply that only supplies a destination, venue, or date
requested in the prior response is context fulfillment, so prior candidates
remain in play without repeated non-weather product work. Explicit same-turn
comparison, refinement, replacement, search, check, cart, or policy requests
still use their normal selected-skill procedures. This is semantic instruction
for the model, not a deterministic intent rule, skill-selection shortcut, tool
grant, or execution gate.

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
deterministic catalog rendering, protected context-only evidence uses
deterministic event assembly, and current product-detail evidence uses a
deterministic verified-detail renderer containing only current names, prices,
categories, and listed fields, followed by a typed weather outcome when
present. Only a current result named `get_product_details_tool` with the
canonical successful-detail marker at its start qualifies. It preserves
evidence rather than inventing comparative judgment.
Other non-search turns receive a fixed retry/cart-check response instead of the
unverified draft. Ordinary editor errors and empty or whitespace-only output
use the same evidence split with `grounding_error`; invalid protected
context-only output instead falls back deterministically.
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
Raw weather arguments/output are the exception: they remain redacted, while
the weather tool-call record retains only categorical `request_shape`,
`location_source`, `provider_input`, and `outcome`, with no
place, ZIP, date, resolved place, URL, body, or exception. Saved profile ZIP
and failed-turn weather content remain scrubbed.
The Judge retains only product evidence/truncation and those catalog scope
outcomes from diagnostics.

## Registered Skills

| Skill | Source | Status | Role | Tools granted | Primary entry modes |
| --- | --- | --- | --- | --- | --- |
| `product-discovery` | `chain_server/skills/shopper/product-discovery/SKILL.md` | Registered | `primary` / `product_procedure` | Search, details, availability, promotions, same-conversation product resolution | General search, category browsing, filter-driven discovery without styling intent |
| `outfit-styling` | `chain_server/skills/shopper/outfit-styling/SKILL.md` | Registered | `primary` / `product_procedure` | Search, details, availability, promotions, same-conversation product resolution | Build, complete, or refine a look; coordinate a requested piece with an anchor; use cart evidence only when cart management is also active |
| `event-context` | `chain_server/skills/shopper/event-context/SKILL.md` | Registered | `modifier` | Event forecast | Use stated destination/venue context, fetch qualified current event weather, or resolve a missing context branch; use explicit setting over saved ZIP; combine only with outfit styling |
| `cart-management` | `chain_server/skills/shopper/cart-management/SKILL.md` | Registered | `standalone` | Cart read, total, add, remove, update, same-conversation product resolution | Explicit cart reads and mutations, alone or beside a product procedure |
| `budget-shopping` | `chain_server/skills/shopper/budget-shopping/SKILL.md` | Registered | `modifier` | None | Stated price ceilings and budget bundles; combine with cart management for cart-total checks |
| `store-policy-answers` | `chain_server/skills/shopper/store-policy-answers/SKILL.md` | Registered | `standalone` | Policy lookup | Returns, shipping, sizing, payment, price matching, and gift cards |

## `event-context`

Purpose: add the smallest useful event-location, venue, and qualified live
forecast context to occasion-led styling.

- Runs only beside `outfit-styling`; activation without that primary receives
  one typed correction and then fails closed through the existing deterministic
  clarification boundary.
- Gives explicit current-turn destination and venue setting precedence over
  explicit recent context for the same event, with saved ZIP last as a
  tentative candidate to confirm.
- When saved ZIP is the only location candidate, asks whether to plan around
  the shopper's usual area or elsewhere; Guest instead supplies destination or
  venue context from scratch.
- On an explicit plan-before-products turn, missing material context produces
  exactly two short sentences with no headings or lists. It may include only
  the location, venue, or date question selected by activation; `none` permits no
  event-context follow-up. With context complete, it produces one short
  paragraph and asks no further event-context question.
- On an ordinary occasion-only shop-now turn, `outfit-styling` runs one search
  for one grounded requested or core role unless the shopper explicitly asks
  for a complete look or names multiple roles. If location is missing and
  materially changes the next recommendation, activation selects only event
  location alongside the results. A saved-ZIP candidate requires “usual area
  or elsewhere?” framing rather than a bare destination question. Once
  destination is established, activation may instead select one material
  venue/setting question; the plan-first stop rule does not apply.
- Activation owns only `event_context_next_question`. It is required exactly
  with this skill and is one of `event_location`, `event_venue`, `event_date`,
  or `none` under the semantic conditions above; it is omitted without this
  skill. The server does not infer a question from enabled weather or missing
  context.
- The modifier composes additively with `outfit-styling`. Its helper may hide
  and execution-block only `get_weather_forecast_tool` when the selected
  question or missing bounded authority prevents a qualified forecast. It
  never hides product, cart, or policy tools and never closes the overall tool
  loop. Consuming the one weather attempt closes only weather for that turn.
- Protected event decision rendering is selected from evidence rather than an
  activation action. It applies only when event context is active, there is no
  current non-weather business-tool activity, and a current typed weather
  outcome (success or failure) exists. Missing-location/venue or an empty draft
  skips that editor. A separate prior-candidate fallback uses deterministic
  event assembly only when the draft is empty; prior candidates with a nonempty
  draft remain on ordinary grounding only when there is no current weather
  outcome. A comparison that calls only weather remains protected. Other
  protected weather-outcome turns give the narrow decision editor only bounded
  shopper-authored event text and the server-owned deterministic weather
  styling direction. Any current non-weather business-tool activity or evidence
  uses normal grounding, preserving comparisons, product details, cart work,
  and policy answers. After successful weather, that same activity prevents
  response postprocessing from restoring unrelated historical-product names.
- Does not infer that Cancun means beach or that any ZIP or place establishes
  outdoor/indoor setting, terrain, weather, wind, climate, season, dress code,
  local norms, or product performance.
- Grants `get_weather_forecast_tool` for one model-visible attempt on an
  eligible turn with bounded date authority; otherwise the runtime hides and
  execution-blocks it. It uses
  `confirmed_saved_zip`, with both location fields omitted from model
  arguments, only when the deterministic gate accepts a current
  location-neutral statement explicitly naming `my`/`the` usual/home area, a
  bare affirmative immediately after the assistant's usual/home-area question,
  or an immediate strict date-only follow-up to an accepted confirmation. Any
  explicit current place, question, negation, uncertainty, or override rejects
  saved mode. It uses
  `shopper_provided_location` only with one bounded exact named-place, address,
  or postal-code phrase copied from shopper-authored text into `location`.
  For an abbreviation or ambiguous name, `location_query` is required: it must
  preserve that exact phrase as its first component and append only one or two
  comma-separated region/country qualifiers. Keep `location="NYC"` and use
  `location_query="NYC, NY"`; `Springfield, TX` is a valid explicit regional
  assumption. It never carries an unstated ZIP or numeric component and is
  omitted only when `location` is already sufficiently qualified. Semantic equivalence
  remains model-owned rather than deterministic proof and is correctable
  through provider-resolution disclosure. The adapter sends the bounded named
  place directly to Visual Crossing Timeline and uses no synthesized ZIP or
  separate geocoder; Visual Crossing's `resolvedAddress` becomes the reversible
  `resolved_location` assumption. Current
  explicit destination prevents fallback to saved ZIP.
- Treats modal lowercase `may be` as uncertainty while accepting calendar
  `May 5` as a valid date.
- Treats the model limit as one attempt: a schema-invalid weather call consumes
  it and cannot be repaired in that turn. Within a valid call,
  `max_provider_attempts: 2` permits one internal retry only for timeout or
  HTTP 5xx. HTTP 400 maps to generic `weather_request_invalid`; other 4xx,
  connection, and response-validation failures are not retried.
- Requires an exact ISO event date, complete inclusive range, or the typed
  `relative_date=next_week` mode. That mode is allowed only when the shopper
  used the exact phrase `next week`. Exact `<weekday> next week` also requires
  the matching lowercase `weekday`, and the server derives that exact day
  inside the next Monday-through-Sunday window from one captured UTC date.
  Bare `next week` omits `weekday` and derives the full range. Missing,
  mismatched, mixed, negated, or superseded weekday authority fails closed.
  Without a bounded shopper-authored date signal, the runtime hides and
  execution-blocks weather for that turn. A direct date question is permitted
  only when accepted activation selected `event_date`; the server does not
  collect or infer weather-only context otherwise.
  The model resolves
  an unambiguous single-day phrase such as `tomorrow` against that same anchor
  into an exact ISO date. Other ambiguous or unresolved relative dates can
  authorize one date clarification only through accepted `event_date` under
  that same ordering and enabled-and-material rule.
- Uses successful current-turn forecast evidence only. The model-visible
  projection includes the provider-resolved place only for
  `shopper_provided_location`, and final rendering discloses it as the
  forecast-location assumption so any `location_query` qualification is
  reversible. The field is omitted for
  `confirmed_saved_zip`. Prior durable assistant forecast summaries are
  redacted from graph and grounding-editor recent discussion while remaining
  stored and exactly replayable; prior weather tool messages are excluded from
  prior evidence. Raw arguments and output are redacted from diagnostics and
  failed-turn partial output; only the categorical weather summary above
  remains, and saved profile ZIP is scrubbed from diagnostic string keys and
  values. Final rendering appends one exact canonical block
  containing every validated daily date, condition, available low/high
  temperature, precipitation probability/types, Visual Crossing attribution,
  and forecast uncertainty; model prose cannot shorten or selectively omit it.
  Normal grounding removes weather-domain fact language or fact-shaped
  dates/values that lack current weather evidence while preserving ordinary
  grounded styling language. Protected context-only rendering accepts no
  free-form shopper-facing styling. Its exact two-key JSON decision must contain
  an exact shopper-authored venue quote and one or two distinct allowlisted
  adjustment codes. Malformed, ungrounded, extra-key, duplicate, unknown, or
  wrong-cardinality output falls back. The server maps valid codes to fixed
  phrases and deterministically assembles exact prior names, its weather
  direction, only the accepted question, and the typed weather failure or
  canonical success block.
- Treats weather as styling context, never proof of product performance or an
  unstated catalog constraint. There is no new FastAPI, SSE, or UI shape.
- Remains disabled at the provider boundary by default. Before an operator
  enables shopper traffic, the selected Visual Crossing plan's attribution,
  display, storage, and sharing rights must be confirmed for durable final
  assistant summaries and downstream app-model/output-guardrail processing.

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

An established-product comparison is a procedure within this skill, not a new
skill, deterministic intent branch, or rediscovery request. The model submits
all compared prior products in one batched resolver call and, after every
required product resolves uniquely, calls the scalar detail tool once per ref in
separate model steps. The default two-read cap covers one pair; larger
comparisons cannot receive more than two detail reads in the turn. Missing or
ambiguous required products clarify without a substitute search. The response
compares only item-specific confirmed fields, keeps styling judgment separate,
and treats weather as optional additional evidence that never replaces product
facts or proves performance. This orchestration is model-owned; deterministic
code enforces exact refs, limits, and evidence grounding rather than comparison
intent or pair completeness.

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
