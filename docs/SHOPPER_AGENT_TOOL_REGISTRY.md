# Shopper Agent Tool Registry

This registry documents the internal tools available to the shopper-serving
Deep Agent. These names are for engineers, evaluators, and agent instructions.
They are not shopper-facing UI language and should not appear in assistant
responses.

The runtime sources of truth are
`chain_server/src/tool_policy.py` for the immutable tool policy and validated
frontmatter grants,
`chain_server/src/deepagents_runtime.py::DeepAgentsRuntime._create_agent` for
wrapper registration, and
`chain_server/src/skill_activation.py::ShopperSkillActivationMiddleware` for
model binding and dispatch enforcement, with
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
`SKILL.md` contents. Only then does the next model step receive the union of
those skills' declared `tools_granted` from the nine-tool registry. Every
app-owned shopping dispatch independently rechecks the selected skill, grant
union, and immutable policy before invoking its handler.

For primary shopper procedure selection, `product-discovery` and
`outfit-styling` are mutually exclusive. `budget-shopping` is a modifier and is
selected only when the shopper states a budget. Cart or policy skills may still
join the applicable procedure for a genuine multi-intent turn; standalone cart
and policy turns do not require a product primary. A terse item-only follow-up
inside an active outfit-building or style-led single-piece thread remains an
`outfit-styling` task.

The active catalog capabilities generate both the exact taxonomy values and
the non-taxonomy required-constraint properties in the search-tool schema.
The model owns semantic selection through an agent-facing structural transport
schema. Runtime then applies a separate strict semantic search model, so
cross-field failures reach capability-aware validation and receive exact
capability-derived corrections instead of failing before tool execution.
The model owns `taxonomy_status`; runtime never semantically rewrites it.
Capability-owned exact category/subcategory relationships determine whether the
submitted status and selection are coherent.
Every text search also requires `requested_product_type`, the shortest product
noun or true umbrella from the current turn or direct antecedent. It excludes
color, material, fit, occasion, weather, and style modifiers. It is `null` only
for image-only search. Literal validation may bind the longest exact advertised
suffix in a modifier-bearing model phrase (`waterproof boots` to `boots`), but
disables that shortcut for explicit alternatives containing `and`, `or`, `/`,
or `&`. `closed shoes or boots` remains model-owned alternative or umbrella
reasoning. If the current shopper turn contains one unambiguous literal pair of
exact advertised subcategories in the same category, runtime instead validates
that both model-authored branches are preserved under
`member_of_requested_umbrella`. The pair still uses one catalog execution. Its
candidate window covers both branches, and rank-preserving selection keeps one
returned candidate per branch when available before trimming to the configured
result count. Modified, synonymous, ambiguous, and cross-category alternatives
remain model-owned. Each call covers at most one category.
For a broad request that names no type, `agent_selected_type` selects exactly
one advertised subcategory. Runtime retains `agent_selected_type` and derives
the duplicate `requested_product_type` provenance from that selection. It is
rejected for a shopper-named scope rather than silently reinterpreted. When an
agent-selected open-role call is malformed, deterministic validation stops
before retrieval, reports the exact eligible subcategories from the current
capability contract, and returns related cross-field corrections. The model
operating under the active skill still owns semantic selection.

The activation boundary fails closed. Missing or invalid skill content exposes
no commerce tools. A commerce call placed in the same model response as the
activation call is rejected, and activation from a prior turn cannot unlock the
current turn. An activation-phase model response without the required call is
rejected as `skill_activation_failed`, rather than becoming shopper-facing
prose. A post-activation call outside the selected grant union is rejected
before its handler with `SHOPPER_SKILL_TOOL_NOT_GRANTED`. Frontmatter grants and
the independent policy must agree exactly at startup, so unknown or drifted
skill/tool pairs fail closed. Tool schemas and wrappers continue to enforce
deterministic request and state preconditions.

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
metadata, so checkpoint state cannot retain stale skill descriptions or removed
names if skill files change while the process remains alive.

After activation, parallel shopping calls remain disabled. Repair accounting
also rejects any active model response containing more than one shopping tool
call before execution. Repair accounting
uses the full normalized `requested_product_type` phrase; distinct advertised
siblings are protected from being treated as the same scope. Each scope has one
total repair. A schema correction or a fresh constraint-provenance review can
consume that shared budget; constraint feedback returned by an in-flight schema
repair closes the loop for synthesis rather than opening another repair. The
repair request uses a concise, schema-generic system prompt in place of the base
runtime prompt. The skill gate appends the complete active shopper-skill
instructions. Its messages contain only
the current shopper message and bounded, sanitized validator feedback in a
separate Human data message. Echoed rejected arguments are stripped; native
Pydantic feedback is reduced to rejected top-level field names, and free-form
requested-scope text is not replayed. Invalid AI/tool history and earlier
conversation history are absent. Only `search_catalog_tool` remains exposed and
forced, and parallel calls stay disabled. A successful partial search may
continue to another valid role with its own single repair opportunity. For a
native tool-transport failure, the requested scope is locked only when current
or recent shopper text grounds it; an ungrounded model-generated scope may be
corrected. A rejected change to a grounded free-form scope that cannot be
reconstructed safely is removed before execution and recorded in
`agent_diagnostics` with reason `repair_scope_changed`. Runtime
validation separately enforces capability-owned advertised sibling coherence. A
strict request rejection with an independently valid constraint object snapshots
its advertised `required_constraints` privately. The isolated feedback includes
that exact finite object, including an explicit empty object, rather than asking
the repair to reconstruct advertised values. Free-form rejected arguments stay
excluded.
Before execution, runtime restores every independently valid finite lock: the
taxonomy relation, canonical advertised constraints (including an explicit
empty object), explicit valid `scope_complete` and `search_mode`, and
`requested_product_type` when a singleton exact or agent-selected taxonomy
determines it. The model owns only invalid fields. Drift in a restorable lock is
corrected in place; bounded tool-call diagnostics expose only the restored
field names in `restored_fields`. The constraint lock follows accepted
product-phrase normalization, and list-valued constraints compare without
regard to order; omitted optional defaults equal explicit empty values.
A no-direct repair may clear constraints only while remaining no-direct; a
repair that changes to retrieval must retain the original advertised
constraints.
Native enum failures on an
`agent_selected_type` call also include the shopper-named/open-role provenance
rule in the same repair feedback. The correction is explicit: a shopper-named
advertised subtype uses `exact_requested_type`, while a named umbrella or set of
alternatives uses `member_of_requested_umbrella`. If that repair ends in a valid no-direct
outcome, the fixed not-advertised response takes precedence over the earlier
validation failure. A
native failure confined to `required_constraints` includes only finite,
validated taxonomy status and selection in repair feedback, never the free-form
scope, query, or guidance. Scope comparison is private. Relation drift is
restored before the repaired constraint call executes. When native taxonomy
validation fails, independently valid constraints are likewise restored before
execution. A locked boundary that cannot be restored safely remains
comparison-protected and closes under the matching `repair_*_changed` reason.
Malformed or nonempty free-form `unadvertised_requirements` arguments are never
restored; a native schema-invalid call containing one closes without repair. A
schema-valid, genuinely open `agent_selected_type` request retains the bounded
review for a proposed inferred requirement. A completed scope, an exhausted repair budget,
or any tool result
beginning with `STOP_TOOL_USE` removes tools from the following model step and
closes the loop. Successful search-only turns go directly to deterministic
rendering; mixed-tool turns synthesize from evidence already collected. When a
runtime semantic open-role schema repair removes a proposed inferred
requirement, runtime replaces its submitted pre-search guidance with neutral
generic guidance for the selected role. A successful or zero-result search that
consumes the final configured search slot records `SEARCH_BUDGET_EXHAUSTED`;
the next model step removes only `search_catalog_tool`. Product-detail,
availability, and cart tools plus honest partial synthesis remain available.

Every unadvertised requirement on a shopper-stated product scope fails closed
before retrieval, including when the model uses a synonym rather than the
shopper's exact wording. The bounded constraint review is reserved for a
proposed inferred requirement on a genuinely open `agent_selected_type` role
when its shared repair budget remains. It must preserve requested type,
taxonomy status, taxonomy, completion state, `search_mode`, and every advertised
hard constraint. Within that preserved hard scope, it may correct only the soft
`semantic_query`, the reviewed unadvertised-requirement lane, and its associated
`shopper_guidance`; the requirement is either replaced with the shopper's
shortest exact wording or removed. Exact wording and unresolved provenance fail
closed; constraint feedback after a schema repair closes the loop for
synthesis. Removal scrubs the product-attribute claim from `shopper_guidance`.
Only a successful repaired result whose scope remains explicitly
partial may continue with another valid role.

Grounding reads only tool-role messages and partitions current-turn evidence by
the server-owned request marker. Prior-turn tool evidence may resolve a direct
reference but cannot prove that a new search or mutation ran. For search-only
turns, successful search results carry `SEARCH_DIRECTION_EVIDENCE`: the
model-authored semantic query used as an independent internal ranking preference,
plus required pre-retrieval `shopper_guidance` authored under the active skill.
The latter is presented without a final-synthesis model call; static skill
`response_guidance` is the fallback. Results, taxonomy, filters, semantic query,
and drafts are not converted into guidance after retrieval. Before guidance is
serialized as deterministic shopper-facing evidence, the runtime replaces a
sentence containing `waterproof`, `water-resistant`/`water resistant`,
`weather-safe`/`weather safe`, `bug-safe`/`bug safe`, wet surface(s) or
ground(s), `wet conditions`, `grass`, `gravel`, `all-day`/`all day`,
`best-in-category`/`best in category`, `maximally`, outdoor surface(s), or
`outdoor walking` with neutral selected-role guidance. It also covers
`handle`/`handles`, `suitable for`, `work well for`/`works well for`, `secure
for`, or `stay secure for` constructions tied to rain, wet weather, or outdoor
use, including "works well in wet weather/conditions." Search semantics,
taxonomy, hard constraints, and retrieval remain
unchanged. Candidate facts remain deterministic. Multi-role output groups each
guidance sentence and confirmed-filter set only with products from its originating search and
deduplicates candidates by `product_ref`, not display name. Mixed-outcome turns
preserve successful product groups when another scope has no direct match or an
unsupported requirement. A fixed no-direct or unsupported canned response is
used only when that rejection is the sole current-turn business-tool outcome.
Scoped zero-result evidence retains its
exact advertised taxonomy and filters and cannot support a broader absence
claim. If all current-turn business calls are rejected catalog searches and no
current product evidence exists, the runtime returns a fixed retry response
before model-based editing; prior evidence cannot be presented as results from
the rejected search.

The runtime records model-issued calls for the activation control, app-owned,
and built-in tools in `agent_diagnostics`. Calls retain their model-issued order
and structured arguments; deterministic stops identify rejected and duplicate
calls. Search-schema validation failures are recorded as rejected catalog
requests rather than executed searches. A completed activation records each injected
`/shopper/<name>/SKILL.md` path in `skill_files_read`; a later successful
`read_file` of a skill file is recorded there as well. Pre-activation commerce
rejections use `skill_activation_required`; post-activation ungranted calls use
`skill_tool_not_granted`. On graph failure, bounded
current-turn assistant/tool messages are read from the checkpoint before
cleanup. Diagnostics also include at most 24 records and 32,000 serialized
characters of structured current-turn product evidence from successful catalog
search and detail results, with `product_evidence_truncated` marking either
bound, plus at most eight bounded `catalog_scope_outcomes` records for
`no_direct_catalog_match` and `zero_results`. Each search's taxonomy and
confirmed filters stay attached only to its returned products. The evaluator
copies only this evidence list, its truncation flag, and those scope outcomes;
semantic queries, raw tool messages, model reasoning, and all other diagnostics
are discarded. This untrusted, log-sensitive operator/evaluation metadata is
not shopper-facing response text.

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
| `read_only_catalog` | Reads catalog-domain data or reports its availability boundary without shopper state mutation. | Granted by product discovery and outfit styling. |
| `read_only_catalog_cache` | Legacy/cache-only catalog read classification. | No active registered tool uses the cache as product-detail truth. |
| `read_only_cart` | Reads the authoritative cart for the scoped `cart_id`. | Granted only by cart management. |
| `computed_read_cart` | Computes over authoritative cart reads. | Granted only by cart management; do arithmetic in code, not model prose. |
| `mutating_cart` | Changes cart contents. | Slice 0 requires the cart-management grant, valid refs, and service-side success. Skill instructions require explicit shopper intent, but server-owned current-turn intent authorization is a later slice. |
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
  context and only their `tools_granted` union becomes available.

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
- A post-activation tool outside the selected grant union is rejected with
  `SHOPPER_SKILL_TOOL_NOT_GRANTED` and diagnostic reason
  `skill_tool_not_granted`.

Current limitations:

- The activation gate enforces both the two-phase ordering boundary and the
  selected skills' exact tool grants. Explicit mutation-intent authorization is
  not part of Slice 0.
- The required selection phase adds one bounded model step to each turn.

### `search_catalog_tool`

Purpose: Product discovery and recommendation over the catalog.

Inputs:

- `semantic_query` (required): One soft/descriptive product query. Product words
  may be repeated for relevance but are not required; this field is independent
  of taxonomy and never enforces another must-have. Use an empty string only for
  image-only search.
- `shopper_guidance` (required): One concise, product-agnostic shopper-facing
  sentence written before retrieval under the active skill. It connects the
  selected role to the shopper's goal or direct antecedent without naming
  candidates, asserting product attributes, naming types outside the selected
  scope, or exposing internal mechanics. It must be nonempty for every search
  except `image_only` and `no_direct_catalog_match`; those two statuses require
  it to be empty.
- `requested_product_type` (required, nullable): The shortest product noun or
  true umbrella from the current request or direct antecedent. Exclude color,
  material, fit, occasion, weather, and style modifiers. For an open
  `agent_selected_type` role, runtime derives this duplicate provenance from the
  chosen advertised subcategory. This field is provenance, not a catalog enum
  or ranking query. It is `null` only for image-only search. Literal validation
  can recover an advertised suffix from a modifier phrase, but never collapses
  an explicit `and`, `or`, `/`, or `&` alternative phrase to its final noun.
- `taxonomy` (required): `category` and `subcategory` arrays whose allowed values
  are generated from cached catalog capabilities. Use exact advertised values;
  executable text search requires at least one value, while image-only and
  no-direct no-retrieval requests use two empty arrays. A subcategory-only value
  maps to its owning category; an ambiguous owner or incompatible pair is
  rejected. Each call accepts at most one category.
- `taxonomy_status` (required): `exact_requested_type`,
  `member_of_requested_umbrella`, `agent_selected_type`,
  `no_direct_catalog_match`, or `image_only`. `agent_selected_type` is for a
  broad request that names no type; it selects exactly one advertised
  subcategory as the focused starting role. It is forbidden for a role
  whose type the shopper named, including an alternative, confirmation,
  comparison, or follow-up. Invalid open-role provenance is rejected rather
  than silently reinterpreted. Runtime retains `agent_selected_type` and derives
  its requested type from the selected subcategory but never semantically
  rewrites `taxonomy_status`. Exact category/subcategory coherence is validated
  from current catalog capabilities. The no-direct value is only
  for an explicitly requested concrete
  product type with no faithful advertised taxonomy value, uses empty taxonomy
  and no hard constraints, and performs no retrieval.
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
  taxonomy values through the structural transport schema. The strict runtime
  semantic model validates cross-field relationships, then maps valid values to
  catalog fields. The runtime does not maintain taxonomy keyword aliases.
- If an explicitly requested product type has no advertised match, the agent
  uses `no_direct_catalog_match`, which reports the gap without retrieval,
  before asking permission to search an adjacent type. It does not omit the
  type or broaden to its parent category. An unsupported modifier does not erase
  an advertised type, and subjective style stays in `semantic_query`.
- A zero-result retrieval reports only that its exact advertised taxonomy and
  filter scope returned no matches. It cannot establish absence for a different
  product type or the whole catalog.
- A product must-have absent from the generated schema belongs in
  `unadvertised_requirements`. Every such requirement on a shopper-stated
  product scope fails closed before retrieval, even when the model uses a
  synonym. The bounded review is reserved for a proposed inferred requirement
  on a genuinely open `agent_selected_type` role when the shared scope repair
  is still available. It freezes requested type, taxonomy status, taxonomy,
  completion state, `search_mode`, and all advertised hard constraints. Within
  that preserved hard scope, it may correct only the soft `semantic_query`, the
  reviewed unadvertised-requirement lane, and its associated guidance; the
  requirement is either replaced with the shopper's shortest exact wording or
  removed. Removed
  requirements scrub product-attribute guidance; unresolved provenance fails
  closed, and constraint feedback after an in-flight schema repair closes the
  loop for synthesis.
- Capability-derived taxonomy mapping and all other hard-filter validation
  complete before retrieval. Unsupported fields, values, and operators stop the
  search instead of becoming semantic relevance. For a malformed open-role
  `agent_selected_type` call, the repair error enumerates the exact eligible
  advertised subcategories and includes any submitted unadvertised requirement
  plus its associated guidance correction in the same result.
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
- Successful results carry the pre-retrieval `shopper_guidance` separately from
  product evidence. For a completed search-only response, the runtime presents
  that guidance without a final-synthesis model call; static skill
  `response_guidance` is the fallback. Prohibited outdoor/weather guarantee
wording is first replaced with neutral selected-role guidance; this does not
alter the query, taxonomy, constraints, or retrieval. This includes
outdoor-surface or outdoor-walking claims and "handle rain," "work well for
outdoor surfaces," or "stay secure for outdoor walking" constructions, plus
`wet conditions` and "works well in wet weather/conditions." Code separately
lists all returned candidates with their names, prices, and
categories and keeps every
  confirmed-filter group attached only to products from the same search. For
  multi-role results, each guidance sentence is grouped with the products from
  that same search. Candidate groups deduplicate by `product_ref`; display names
  are not identity. A mixed-outcome turn retains successful groups when a later
  scope has no direct match or an unsupported requirement. A partial successful
  result set receives the neutral continuation: "This is a
  partial result set. I can continue with the next requested piece or search
  scope."
- A zero-result response carries `SEARCH_NO_MATCH_GROUNDING_NOTE` with the exact
  advertised taxonomy and confirmed filters searched. That evidence applies
  only to its own scope and cannot establish absence for another product type or
  the whole catalog.

Side effects:

- Caches returned `PRODUCT_REF` values inside the scoped conversation for later
  product details or cart add calls.
- Cached refs let later product-detail or cart calls resolve a previously found
  product without another catalog search. Each shopper turn still runs the
  assistant model.
- The cache holds at most 50 refs per conversation and is process-local. The
  graph checkpoint does not persist or replicate it.
- Records catalog timing and model-usage metadata.

Failure behavior:

- Returns a short tool-readable failure string such as unavailable catalog,
  invalid search request, catalog failure, or no matching products.
- `no_direct_catalog_match` returns before catalog retrieval. Its fixed
  shopper-safe no-match response applies only when that rejection is the sole
  current-turn business-tool outcome.
- Returns without calling catalog search when a required constraint cannot be
  enforced by the active capabilities. An unadvertised must-have on a
  shopper-stated product scope fails closed even when the model uses a synonym.
  Its fixed shopper-safe response applies only when that rejection is the sole
  current-turn business-tool outcome. Only a proposed inferred requirement on
  a genuinely open `agent_selected_type` role may receive one
  constraint-provenance review; a second mismatch or other unresolved
  provenance fails closed.
- A full normalized `requested_product_type` scope receives one total repair,
  enforced by a server-derived key that keeps distinct advertised siblings
  separate. A schema failure or fresh constraint-provenance question can consume
  it. Constraint feedback after an in-flight schema repair closes the loop for
  synthesis. A successful partial search may continue to another valid role
  with its own single repair opportunity; the configured turn cap remains three
  searches. A successful or zero-result third search carries
  `SEARCH_BUDGET_EXHAUSTED`, so no fourth search is possible while non-search
  tools and synthesis remain available.
- A completed scope, unsupported taxonomy or requirement, repeated scope,
  exhausted detail budget, or other `STOP_TOOL_USE` result closes further tool
  use. Search-budget exhaustion removes only `search_catalog_tool`, as described
  above. Completed successful search-only evidence renders deterministically;
  mixed-tool paths may use a tool-disabled synthesis step.
- If the Deep Agents loop fails after catalog search has returned products, the
  runtime clears the failed thread checkpoint and returns a grounded partial
  product summary instead of a generic shopper-facing error.

Skills that grant this tool:

- `product-discovery`
- `outfit-styling`

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

Skills that grant this tool:

- `product-discovery`
- `outfit-styling`

Current limitations:

- Source IDs in the current feed are generated and are not guaranteed across
  catalog replacements. Lookup is deterministic within the active snapshot.
- Ref authorization is process-local, bounded, and separate from graph
  checkpoint state; a restart, another replica, or eviction requires a fresh
  search.

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

Skills that grant this tool:

- `cart-management`

Current limitations:

- `CART_LINE_ID` is the memory service's opaque, non-reusable cart-line ID.
- Stored source product IDs, variants, and inventory are future work; the
  legacy add/remove endpoints still identify stored products by display name.

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

Skills that grant this tool:

- `cart-management`

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

Skills that grant this tool:

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

Skills that grant this tool:

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

- Sends one absolute-quantity `PUT` for the current `CART_LINE_ID`.
- A positive quantity updates the row in one transaction; `0` deletes it.
- Commits the mutation and its idempotency record together. Repeating the same
  key and mutation replays the stored result; conflicting key reuse is rejected
  without a remove-then-add sequence.
- Idempotency records currently persist for the SQLite database lifetime;
  retention and cleanup policy remain follow-up work.

Failure behavior:

- Returns `cart_line_not_found` when the supplied line ID is not current,
  non-retryable `cart_update_failed` for a rejected client request, retryable
  `cart_update_failed` for transport/server failure, and
  `cart_response_invalid` for malformed service output.

Skills that grant this tool:

- `cart-management`

Current limitations:

- The cart service still stores display names rather than source product IDs or
  variants; this does not affect stable line targeting for quantity updates.

### `get_store_policy_tool`

Purpose: Read controlled store-policy content for a supported topic.

Inputs:

- `topic`: One of `returns`, `shipping`, `sizing`, `payment`, `price_match`, or
  `gift_cards`.

Preconditions:

- Policy questions must use this tool instead of model knowledge.
- The operator-managed YAML must set `configured: true` after all placeholders
  are replaced.

Outputs:

- The operator-managed policy title and body from
  `shared/configs/chain_server/store_policies.yaml`.

Side effects:

- Loads the policy file into a process-local read-only cache on first use.

Failure behavior:

- Returns `policy_not_configured` while the bundled template remains disabled,
  `policy_topic_not_found` for an unavailable topic, and `policy_load_failed`
  when the policy file cannot be read or enabled content still contains an
  operator-placeholder marker. The assistant should relay that the policy is
  unavailable and direct the shopper to the retailer's help center.

Skills that grant this tool:

- `store-policy-answers`

Current limitations:

- The bundled policy values are operator placeholders and `configured` defaults
  to `false`, so default deployments never present them as store policy. The
  file is outside the agent-readable skill backend and is accessible only
  through this controlled tool.

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

Skills that grant this tool:

- `product-discovery`
- `outfit-styling`

Current limitations:

- There is no live inventory or variant lookup behind this tool.

## Skill Access Matrix

This matrix is the deterministic per-skill authorization contract. The model
sees only the union of the current selected skills' grants, and every app-owned
shopping dispatch rechecks the grant against the independent immutable policy.
Unknown or ungranted calls fail closed before their handlers. Tool schemas and
wrappers continue to enforce deterministic request and state preconditions.
Only skills listed as registered in
[Shopper Agent Skill Registry](SHOPPER_AGENT_SKILL_REGISTRY.md) are eligible for
activation.

| Skill | Tools granted |
| --- | --- |
| `product-discovery` | `search_catalog_tool`, `get_product_details_tool`, `check_product_availability_tool` |
| `outfit-styling` | `search_catalog_tool`, `get_product_details_tool`, `check_product_availability_tool` |
| `budget-shopping` | None (`tools_granted: []`) |
| `cart-management` | `get_cart_tool`, `view_cart_total_tool`, `add_cart_items_tool`, `remove_cart_item_tool`, `update_cart_items_tool` |
| `store-policy-answers` | `get_store_policy_tool` |

Multi-intent turns should select and inject every needed skill during the one
activation step. For example, a request to "find shoes and a bag for this
outfit under $120 and add the shoes" combines outfit styling, budget shopping,
and cart management, not product discovery. The cart mutation still
happens only after a valid product ref is selected. The skill instructions
require explicit add intent; deterministic execution-time intent authorization
is planned for a later slice.

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
10. Matching entries in the granting skills' `tools_granted` frontmatter and
    `tool_policy.py`; exact startup validation must reject drift in either
    direction.
11. Unit coverage proving registration, pre-activation rejection, model-visible
    allow/deny binding, direct-dispatch rejection for ungranted skills, and
    required ref production or consumption after an allowed activation.

Mutating tools also need explicit user intent, idempotency design, ownership
checks, and retry behavior. Checkout, payment, order, account, and profile-write
tools require a separate confirmation and authorization design before they are
eligible for the shopper-facing agent.
