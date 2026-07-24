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

The active Deep Agents runtime registers eleven app-owned shopper commerce tools
plus one internal activation control tool. Every turn begins in an activation
phase where the model sees only `activate_shopper_skills_tool`, its use is
forced, and parallel tool calls are disabled. After the model semantically
selects the smallest skill set for the complete current intent, the runtime
validates those names and deterministically injects the full selected
`SKILL.md` contents. Only then does the next model step receive the union of
those skills' declared `tools_granted` from the eleven-tool registry. Every
app-owned shopping dispatch independently rechecks the selected skill, grant
union, and immutable policy before invoking its handler.

For primary shopper procedure selection, `product-discovery` and
`outfit-styling` are mutually exclusive. `budget-shopping` is a modifier and is
selected only when the shopper states a budget. Cart or policy skills may still
join the applicable procedure for a genuine multi-intent turn; standalone cart
and policy turns do not require a product primary. A terse item-only follow-up
inside an active outfit-building or style-led single-piece thread remains an
`outfit-styling` task.

The model-facing catalog tool accepts one flat executable search. Its fields
are `semantic_query`, `shopper_guidance`, `requested_product_type`, `taxonomy`,
`required_constraints`, `scope_complete`, and optional `search_mode`. Catalog
capabilities generate the exact taxonomy values, hard-filter properties and
enum values, typed numeric ranges, and search-mode enum. The schema has no
model-authored taxonomy relationship, clarification branch, or catalog-absence
result. It omits cross-field validators. Runtime applies the existing strict
semantic search model, so invalid individual values fail at the tool boundary
while cross-field failures reach capability-aware validation.
Capability-owned exact category/subcategory relationships determine whether the
selection is coherent.
Every text search also requires `requested_product_type`, the shortest product
noun or true umbrella from the current turn or direct antecedent. It excludes
color, material, fit, occasion, weather, and style modifiers. It is `null` only
for image-only search. Literal validation may bind the longest exact advertised
suffix in a modifier-bearing model phrase (`waterproof boots` to `boots`), but
disables that shortcut for explicit alternatives containing `and`, `or`, `/`,
or `&`. `closed shoes or boots` remains model-owned alternative or umbrella
reasoning. The model owns alternative, comparison, ordering, and negation
semantics. A typed selection of multiple advertised subcategories in one
category uses one catalog execution. Its candidate window covers the complete
selection, and rank-preserving selection keeps one returned candidate per
selected subcategory when available before trimming to the configured result
count. The runtime does not extract that selection from shopper prose. Each call
covers at most one category.
For a broad request that names no type, the model selects exactly one advertised
subcategory and names it in `requested_product_type`. That open-role path is
rejected for a shopper-named scope rather than silently reinterpreted. When an
open-role call is malformed, deterministic validation stops
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
uses the full normalized, model-authored `requested_product_type` phrase. It
does not reconstruct shopper alternatives or equate connector and ordering
changes. Each scope has one total repair. A schema correction
or a fresh constraint-provenance review can
consume that shared budget; constraint feedback returned by an in-flight schema
repair closes the loop for synthesis rather than opening another repair. The
isolated request receives the capability-derived typed search tool, compact
server-generated Catalog capabilities, the current shopper message, bounded
sanitized validator feedback in a separate Human data message, and the complete
active shopper-skill instructions. Echoed rejected arguments are stripped; native
Pydantic feedback is reduced to rejected top-level field names, and free-form
requested-scope text is not replayed. Invalid AI/tool history and earlier
conversation history are absent. Only `search_catalog_tool` remains exposed,
tool choice stays automatic so the model can signal clarification by returning
no tool call, and parallel calls stay disabled. A no-tool repair is only
branch/control state: the server marks it, discards the model prose, and emits
`Could you clarify the product type or requirement you want me to use?`. A
successful partial search may
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
The repaired call must preserve capability-validated advertised constraints;
the strict handler rejects drift instead of overwriting model output. Repair
middleware never restores or rewrites taxonomy, constraints, requested type, or
search mode. It may restore only the independently valid structural
`scope_complete` flag, with that field name recorded in bounded
`restored_fields` diagnostics.
Open-role validation failures also include the shopper-named/open-role
provenance rule in the same repair feedback. A shopper-named role retains the
shopper's noun or umbrella; a genuinely open role selects and names one
advertised subtype. The same repair may signal a clarification by returning no
tool call; only the fixed server-authored clarification reaches the shopper. A
native failure confined to `required_constraints` receives sanitized field
feedback plus the typed search tool and compact Catalog capabilities, never the
free-form scope, query, or guidance. Scope comparison is private. Middleware
does not reconstruct or overwrite rejected catalog values; a changed
shopper-grounded scope closes as `repair_scope_changed`. Malformed or nonempty
free-form `unadvertised_requirements` arguments are never restored. A native
schema-invalid call containing one closes without repair. A
schema-valid, genuinely open request retains the bounded
review for a proposed inferred requirement. A completed scope, an exhausted repair budget,
or any tool result
beginning with `STOP_TOOL_USE` removes tools from the following model step and
closes the loop. Closed turns get one tools-disabled synthesis from evidence
already collected. Successful search-only drafts then pass through the
grounding editor, with deterministic rendering as fail-closed fallback. When a
runtime semantic open-role schema repair removes a proposed inferred
requirement, runtime replaces its submitted pre-search guidance with neutral
generic guidance for the selected role. A successful or zero-result search that
consumes the final configured search slot records `SEARCH_BUDGET_EXHAUSTED`;
the next model step removes only `search_catalog_tool`. Product-detail,
availability, and cart tools plus honest partial synthesis remain available.

Every unadvertised requirement on a shopper-stated product scope fails closed
before retrieval, including when the model uses a synonym rather than the
shopper's exact wording. The bounded constraint review is reserved for a
proposed inferred requirement on a genuinely open role when its shared repair
budget remains. It must preserve requested type, taxonomy, completion state,
`search_mode`, and every advertised
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
A closed search gets one tools-disabled synthesis under that skill and then the
grounding editor. Static skill `response_guidance` and pre-retrieval guidance
support deterministic fallback. Results, taxonomy, filters, semantic query,
and drafts are not converted into evidence after retrieval. Before fallback
guidance is serialized as shopper-facing text, the runtime replaces a
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
preserve successful product groups when another scope has an unsupported
requirement. A fixed unsupported-requirement response is used only when that
rejection is the sole current-turn business-tool outcome.
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
`zero_results`. Each search's taxonomy and
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
| `get_product_details_tool` | `read_only_catalog` | Active catalog snapshot; request-local evidence authorizes the ref | Registered |
| `resolve_conversation_products_tool` | `read_only_conversation` | Durable same-conversation presented-product events | Registered |
| `get_cart_tool` | `read_only_cart` | Memory cart service | Registered |
| `view_cart_total_tool` | `computed_read_cart` | Memory cart service plus cached line prices | Registered |
| `add_cart_items_tool` | `mutating_cart` | Memory cart service | Registered |
| `remove_cart_item_tool` | `mutating_cart` | Memory cart service | Registered |
| `update_cart_items_tool` | `mutating_cart` | Memory cart service | Registered |
| `get_store_policy_tool` | `read_only_policy` | Operator-managed static policy file | Registered |
| `check_product_availability_tool` | `read_only_catalog` | Application availability contract; no live inventory source | Registered |
| `check_active_promotions_tool` | `read_only_promotions` | Application promotions contract; no live promotions source | Registered |

## Risk Classes

| Class | Meaning | Rules |
| --- | --- | --- |
| `internal_control` | Selects and activates static behavioral instructions; it does not read or mutate commerce state. | Forced once at turn start; cannot be batched with commerce execution. |
| `read_only_catalog` | Reads catalog-domain data or reports its availability boundary without shopper state mutation. | Granted by product discovery and outfit styling. |
| `read_only_catalog_cache` | Legacy/cache-only catalog read classification. | No active registered tool uses the cache as product-detail truth. |
| `read_only_conversation` | Resolves typed product references against durable products actually presented in this conversation. | Granted only by product discovery, outfit styling, and cart management; 0 or many matches require clarification. |
| `read_only_cart` | Reads the authoritative cart for the scoped `cart_id`. | Granted only by cart management. |
| `computed_read_cart` | Computes over authoritative cart reads. | Granted only by cart management; do arithmetic in code, not model prose. |
| `mutating_cart` | Changes cart contents. | Slice 0 requires the cart-management grant, valid refs, and service-side success. Skill instructions require explicit shopper intent, but server-owned current-turn intent authorization is a later slice. |
| `read_only_policy` | Reads controlled operator-managed policy content. | Never substitute model knowledge when a topic is absent. |
| `read_only_promotions` | Reports the deployment's promotion signal without treating catalog results or price as markdown evidence. | Granted by product discovery and outfit styling; currently no active promotion is configured. |
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

- `semantic_query` (required): Soft ranking direction only; it cannot change or
  repair the selected taxonomy.
- `shopper_guidance` (required): One concise, product-agnostic sentence authored
  under the active skill before retrieval. It is empty only for image-only
  search.
- `requested_product_type` (required, nullable): The shortest product noun or
  true umbrella from the current turn or direct antecedent. For a genuinely
  open role, it is the one advertised subcategory selected for that role. It is
  `null` only for image-only search.
- `taxonomy` (required): Capability-derived `category` and `subcategory`
  values. Each call accepts at most one category.
- `required_constraints` (required): Capability-derived non-taxonomy
  hard-filter properties and values, plus the explicit
  `unadvertised_requirements` lane.
- `scope_complete` (required): Whether this search plus existing current-turn
  evidence completes the current shopper request.
- `search_mode` (optional): A capability-advertised retrieval mode.

Preconditions:

- Requires either product text or an attached image.
- The agent semantically maps the shopper's request to exact advertised
  taxonomy values, hard-filter properties and enum values, and typed numeric
  range shape through the capability-derived flat schema.
  The strict handler model validates cross-field relationships, then maps valid
  values to catalog fields. The runtime does not maintain taxonomy keyword
  aliases.
- If an explicitly requested product type cannot be mapped faithfully, the
  agent clarifies. It does not omit the type, broaden to its parent, substitute
  an adjacent type, or claim catalog absence. An unsupported modifier does not
  erase an advertised type, and subjective style stays in `semantic_query`.
- A zero-result retrieval reports only that its exact advertised taxonomy and
  filter scope returned no matches. It cannot establish absence for a different
  product type or the whole catalog.
- A product must-have absent from the generated schema belongs in
  `unadvertised_requirements`. Every such requirement on a shopper-stated
  product scope fails closed before retrieval, even when the model uses a
  synonym. The bounded review is reserved for a proposed inferred requirement
  on a genuinely open role when the shared scope repair is still available. It
  freezes requested type, taxonomy, completion state, `search_mode`, and all
  advertised hard constraints. Within
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
  call, the repair error enumerates the exact eligible
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
  product evidence. For a completed search-only response, the runtime runs one
  tools-disabled synthesis under the active skill and grounds the draft against
  tool-role evidence. Static skill `response_guidance` and pre-retrieval guidance
  support deterministic fallback. Prohibited outdoor/weather guarantee wording
is first replaced with neutral selected-role guidance in that fallback; this does not
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
  scope has an unsupported requirement. A partial successful
  result set receives the neutral continuation: "This is a
  partial result set. I can continue with the next requested piece or search
  scope."
- A zero-result response carries `SEARCH_NO_MATCH_GROUNDING_NOTE` with the exact
  advertised taxonomy and confirmed filters searched. That evidence applies
  only to its own scope and cannot establish absence for another product type or
  the whole catalog.

Side effects:

- Adds returned `PRODUCT_REF` values to request-local evidence for later detail,
  availability, or cart-add calls in the same turn.
- If the terminal response sends those products as ordered product cards,
  durable finalization derives a `candidate_set_presented` event and updates the
  compact same-conversation reference index. Search candidates that are not
  presented do not enter that memory.
- Records catalog timing and model-usage metadata.

Failure behavior:

- Returns a short tool-readable failure string such as unavailable catalog,
  invalid search request, catalog failure, or no matching products.
- When faithful taxonomy cannot be selected, the assistant asks one concise
  clarification directly instead of calling this tool.
- Returns without calling catalog search when a required constraint cannot be
  enforced by the active capabilities. An unadvertised must-have on a
  shopper-stated product scope fails closed even when the model uses a synonym.
  Its fixed shopper-safe response applies only when that rejection is the sole
  current-turn business-tool outcome. Only a proposed inferred requirement on
  a genuinely open role may receive one
  constraint-provenance review; a second mismatch or other unresolved
  provenance fails closed.
- A full normalized `requested_product_type` scope receives one total repair.
  Alternative, comparison, ordering, and negation semantics remain model-owned;
  deterministic repair does not reconstruct them from shopper prose. A schema
  failure or fresh constraint-provenance question can consume the repair.
  Constraint
  feedback after an in-flight schema repair closes the loop for synthesis. A
  successful partial search may continue to another valid role
  with its own single repair opportunity; the configured turn cap remains three
  searches. A successful or zero-result third search carries
  `SEARCH_BUDGET_EXHAUSTED`, so no fourth search is possible while non-search
  tools and synthesis remain available.
- A completed scope, unsupported taxonomy or requirement, repeated scope,
  exhausted detail budget, or other `STOP_TOOL_USE` result closes further tool
  use. Search-budget exhaustion removes only `search_catalog_tool`, as described
  above. Completed turns receive one tools-disabled synthesis from collected
  evidence. Search-only drafts then pass through grounding, with deterministic
  rendering as fail-closed fallback. The grounding editor receives only the
  remaining shared model-stage deadline. A timeout finalizes as failed with
  `grounding_timeout`; non-search turns receive a fixed retry/cart-check response
  instead of the unverified draft. Other editor failures and empty or
  whitespace-only successful editor responses use the same fail-closed response
  with `grounding_error`.
- If the Deep Agents loop fails after catalog search has returned products, the
  runtime clears the failed thread checkpoint and returns a grounded partial
  product summary instead of a generic shopper-facing error.

Skills that grant this tool:

- `product-discovery`
- `outfit-styling`

Current limitations:

- Returned product IDs are source `record_id` values. The current feed does not
  guarantee those generated IDs across catalog replacements.
- Durable resolution is exact and same-conversation only. Catalog revision is
  recorded when available but not yet used to invalidate an old reference; an
  absent or changed active product still requires a fresh search.
- Broad multi-item outfit turns share the distinct taxonomy-scope budget.

### `get_product_details_tool`

Purpose: Read deeper facts for a known product established in current-request
evidence by search or unique historical resolution.

Inputs:

- `product_ref`: A `PRODUCT_REF` from current-turn search or a unique result from
  `resolve_conversation_products_tool`.

Preconditions:

- The ref must exist in request-local product evidence.
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
- Product images from current request evidence are added to the response map
  when available.

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
- Returns guidance to search again when an evidence-backed ref is absent from the
  active catalog snapshot.

Skills that grant this tool:

- `product-discovery`
- `outfit-styling`

Current limitations:

- Source IDs in the current feed are generated and are not guaranteed across
  catalog replacements. Lookup is deterministic within the active snapshot.
- Historical authorization requires one unique durable same-conversation
  resolution in the current request. The resolver does not perform fuzzy
  matching or enforce catalog-revision freshness.

### `resolve_conversation_products_tool`

Purpose: Resolve one or more products the shopper refers to from earlier
product-card output in the same conversation.

Inputs:

- `references`: A nonempty typed batch. Each descriptor has a caller label
  `reference_id` and one or more exact selectors: `product_ref`, `display_name`,
  `category`, `turn_sequence`, or `candidate_set_id`.
- `ordinal`: Optional one-based position; valid only with a turn sequence or
  candidate-set ID.

Preconditions:

- Use only when the needed product is not already established by current-turn
  search or another unique resolution.
- Use exact values exposed in the read-only historical-product index. Do not
  submit free-form prose or use the tool to browse.
- The runtime permits one batched call per turn. A second call returns
  `STOP_TOOL_USE` without contacting the memory service.
- The active skill must be `product-discovery`, `outfit-styling`, or
  `cart-management`.

Outputs:

- One independent `resolved`, `ambiguous`, or `not_found` result per descriptor.
- A resolved result includes exactly one product plus its candidate-set ID,
  turn sequence, one-based position, and recorded catalog revision.
- Ambiguous and missing results explicitly require shopper clarification. Up to
  five matches are returned for clarification while `match_count` retains the
  full count.

Side effects:

- Only a unique resolved product is added to request-local evidence, where
  detail, availability, or cart-add tools can use it.
- No persistent state is mutated. The memory service performs no catalog,
  embedding, or model call.

Failure behavior:

- Service or payload failure returns a clarification instruction; the runtime
  does not guess or silently search for a substitute.
- Zero or multiple matches do not authorize any product.

Skills that grant this tool:

- `product-discovery`
- `outfit-styling`
- `cart-management`

Current limitations:

- Matching is exact after trimming and case normalization; no fuzzy or semantic
  matching is implemented.
- Resolution is limited to durable presented-product events in one conversation.
- Preferences, sentiment, active anchors, cross-conversation history, and
  stale-catalog-revision invalidation are not implemented.

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
- Source product IDs are persisted for serving-path adds. Variants and inventory
  remain future work.

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
- Each product must be in current-request evidence from search or one unique
  durable same-conversation resolution.
- The agent must pass `PRODUCT_REF` values, not display names.
- The selected `PRODUCT_REF` must resolve to the intended
  `expected_display_name` when provided.
- Cart mutation scope must match the explicit add request. Styling approval or
  product selection is not enough to add an item unless the shopper also asks
  to add, buy, or put it in the cart.
- When the current add request contains exact evidence-backed product names or
  sufficiently specific abbreviated evidence-backed names, the tool blocks requested
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
- Uses a runtime-generated idempotency key backed by the memory service's
  owner-scoped mutation ledger.
- Duplicate refs in one call are aggregated before mutation.

Failure behavior:

- Unknown refs and service failures are reported per item.
- Expected-name mismatches and explicit-scope mismatches are blocked before any
  cart mutation in that call; the model can retry with the correct resolved ref
  or ask a clarification.
- Partial success is allowed and must be described accurately in the final
  assistant response.

Skills that grant this tool:

- `cart-management`

Current limitations:

- Variant, inventory, and checkout validation are not part of this tool.
- Mutation replay records currently persist for the SQLite database lifetime.

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
- Uses a runtime-generated idempotency key backed by the memory service's
  owner-scoped mutation ledger.

Failure behavior:

- Returns a clear missing-line or mutation failure message and must not be
  described as success.

Skills that grant this tool:

- `cart-management`

Current limitations:

- Mutation replay records currently persist for the SQLite database lifetime.

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
- Mutation replay records currently persist for the SQLite database lifetime;
  retention and cleanup policy remain follow-up work.

Failure behavior:

- Returns `cart_line_not_found` when the supplied line ID is not current,
  non-retryable `cart_update_failed` for a rejected client request, retryable
  `cart_update_failed` for transport/server failure, and
  `cart_response_invalid` for malformed service output.

Skills that grant this tool:

- `cart-management`

Current limitations:

- Variant-level cart identity remains future work; this does not affect stable
  line targeting for quantity updates.

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

Purpose: Give a deterministic answer when a shopper asks whether a known
product or requested size is available.

Inputs:

- `product_ref`: A `PRODUCT_REF` from a prior catalog search in the current
  conversation.
- `variant_hint`: Optional shopper-named size wording; the field name is retained
  for compatibility.

Preconditions:

- Use only for an explicit stock, availability, or size
  question about a known product ref.

Outputs:

- `availability="in_stock"` for a known product.
- Without a variant hint, confirms general availability.
- With a hint, confirms that size for `apparel` and `footwear`;
  other catalog categories are treated as one-size.

Side effects:

- None. The deliberate stub makes no external call.

Failure behavior:

- An unknown or expired `PRODUCT_REF` stops before the helper runs and asks for
  a fresh catalog search.

Skills that grant this tool:

- `product-discovery`
- `outfit-styling`

Current limitations:

- There is no live inventory or variant lookup behind this tool.

### `check_active_promotions_tool`

Purpose: Give a deterministic answer when a shopper explicitly asks whether a
sale, discount, or promotion is active.

Inputs:

- None. The current deployment signal is global.

Preconditions:

- Use for explicit sale or promotion status, not ordinary low-price or budget
  browsing and not store price-matching policy.

Outputs:

- `active=false` with a message that no active sale or promotion is currently
  configured through the assistant.

Side effects:

- None. The deliberate stub makes no catalog or external-service call.

Skills that grant this tool:

- `product-discovery`
- `outfit-styling`

Current limitations:

- There is no live promotions service behind this tool. Product prices and
  catalog search results do not establish sale status.

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
| `product-discovery` | `search_catalog_tool`, `get_product_details_tool`, `check_product_availability_tool`, `check_active_promotions_tool`, `resolve_conversation_products_tool` |
| `outfit-styling` | `search_catalog_tool`, `get_product_details_tool`, `check_product_availability_tool`, `check_active_promotions_tool`, `resolve_conversation_products_tool` |
| `budget-shopping` | None (`tools_granted: []`) |
| `cart-management` | `get_cart_tool`, `view_cart_total_tool`, `add_cart_items_tool`, `remove_cart_item_tool`, `update_cart_items_tool`, `resolve_conversation_products_tool` |
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
| Live inventory, variant, and size availability lookup | Not implemented; the registered no-I/O stub reports deterministic availability for known conversation product refs. |
| Live promotions lookup | Not implemented; the registered no-I/O stub reports that no active promotion is configured through the assistant. |
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
