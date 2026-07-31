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

The active Deep Agents runtime registers twelve app-owned shopper tools
plus one internal activation control tool. Every turn begins in an activation
phase where the model sees only `activate_shopper_skills_tool`, its use is
forced, and parallel tool calls are disabled. After the model semantically
selects the smallest skill set for the complete current intent, the runtime
validates those names and deterministically injects the full selected
`SKILL.md` contents. Only then does the next model step receive the union of
those skills' declared `tools_granted` from the twelve-tool registry. Every
app-owned shopping dispatch independently rechecks the selected skill, grant
union, and immutable policy before invoking its handler.

For primary shopper procedure selection, `product-discovery` and
`outfit-styling` are mutually exclusive. `budget-shopping` is a modifier and is
selected only when the shopper states a budget. The `event-context`
modifier may accompany only `outfit-styling`. It is selected only when physical
context is part of the current styling subject: supplied or changed
destination/date/venue/weather context, a direct weather-aware request, an
answer to its pending question, or an explicit continuation of that established
event, trip, or weather-planning subject. Hypothetical weather relevance does
not activate it for otherwise location-independent styling. It alone adds the
read-only weather tool to the union. Its activation also requires
`event_context_next_question` for the only permitted event-context follow-up;
that field is omitted for every activation without `event-context`. The same
activation may bind one currently valid `weather_receipt_id` only with
`event_context_next_question=none` and an unchanged exact event location/date
scope. A correction, uncertainty, or refresh request omits it. Event
context is additive: it never removes product, cart, or policy grants supplied
by the other selected skills and never closes their normal tool loop. Cart or
policy skills may still join the applicable procedure for a genuine
multi-intent turn; standalone cart and policy turns do not require a product
primary. A terse item-only follow-up inside an active outfit-building or
style-led single-piece thread remains an `outfit-styling` task.

Before activation, the model-visible receipt index contains only receipt
ID/type, shopper location/date scope, and `valid_until`; normalized forecast
evidence is not present. Full evidence stays server-side and can enter
grounding only after an accepted binding. Memory evaluates expiry atomically at
durable turn start, and that accepted receipt set remains the validity snapshot
for the request without a second mid-turn wall-clock check.

The activation input contains the typed current weather scope. For an existing
scope, a separate request-local tools-disabled resolver compares the current
query with the exact completed shopper turns named by its component source
identities. Contract v5 supplies those turns in a dedicated lane independent of
summary compaction and raw-tail limits. The resolver
must emit one forced typed control call. This internal schema channel is neither
a business tool nor a subagent, and it emits only the semantic relation.
Activation alone proposes one atomic selection that copies the scope revision
and chooses `retain`, `set`, or `clear` independently for location and date.
Deterministic compilation accepts
`set` only from current-turn authority. Invalid, unavailable, timed-out, or
unclear semantic output fails closed for prior authority but cannot veto a
validated current-turn `set`/`set` replacement. A missing location/date question is
persisted as a typed pending binding. Only an exact-handle
`answers_pending` relation may authorize retaining its counterpart, and only
when activation sets the named missing component. It means a pending-only
answer; a reply that also changes or withdraws the counterpart is
`same_subject`. Runtime and memory carry and atomically verify the exact handle
through a server-only completion control; without a usable
resolver result, every retain is cleared and prior-dependent weather is
blocked. A validated current-turn `set`/`set` replacement remains independent
authority and may require weather. Prior raw
turns and summary prose cannot authorize the provider adapter.

The model-facing catalog tool accepts one flat executable search. Its fields
are `semantic_query`, `shopper_guidance`, `requested_product_type`, `taxonomy`,
`required_constraints`, `scope_complete`, and optional `search_mode`. Catalog
capabilities generate the exact taxonomy values, hard-filter properties and
enum values, typed numeric ranges, and search-mode enum. The schema has no
model-authored taxonomy relationship, clarification branch, or catalog-absence
result. The model owns how shopper language maps to the requested role,
semantic direction, taxonomy, and constraints. Runtime validates only
structural shape and capability-owned invariants: exact field values,
text-versus-image requirements, one-category cardinality,
category/subcategory ownership, hard filters, retrieval mode, duplicate hard
scopes, and turn limits. It does not classify the shopper's product wording or
infer a semantic relationship between `requested_product_type` and taxonomy.
Every text search also requires `requested_product_type`, the shortest product
noun or true umbrella from the current turn or direct antecedent. It excludes
color, material, fit, occasion, weather, and style modifiers. It is `null` only
for image-only search. The model owns exact, umbrella, open-role, alternative,
comparison, ordering, negation, and parent-category semantics. A typed
selection of multiple advertised subcategories in one category uses one catalog
execution. Its candidate window covers the complete selection, and
rank-preserving selection keeps one returned candidate per selected subcategory
when available before trimming to the configured result count. The runtime does
not extract or validate that meaning from shopper prose. Each call covers at
most one category. A category-only text search emits neutral evidence that
records the model-authored requested role and the advertised category searched;
it does not claim the role is unavailable or prove a parent relationship.

The activation boundary fails closed. Missing or invalid skill content exposes
no commerce tools. A commerce call placed in the same model response as the
activation call is rejected, and activation from a prior turn cannot unlock the
current turn. An activation-phase model response without the required call is
rejected as `skill_activation_failed`, rather than becoming shopper-facing
prose. An invalid skill composition returns a typed reason and receives one
correction attempt. If the corrected composition is also invalid, the
middleware returns a deterministic clarification without another model call or
any shopping-tool execution. Multiple activation calls in one response execute
none and return the generic clarification immediately. A post-activation call outside the selected grant union is rejected
before its handler with `SHOPPER_SKILL_TOOL_NOT_GRANTED`. Frontmatter grants and
the independent policy must agree exactly at startup, so unknown or drifted
skill/tool pairs fail closed. Tool schemas and wrappers continue to enforce
deterministic request and state preconditions.

This design normally adds one activation model step to every turn and permits
one additional correction only for an invalid composition. Deterministic
skill-file injection itself adds no model call. The runtime also excludes default Deep
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

After activation, parallel shopping calls remain disabled. Catalog repair is a
separate bounded structural phase with one opportunity total per turn. It does
not normalize `requested_product_type` into a repair key, classify shopper
prose, lock a semantic product scope, or reject a correction because its role
noun changed. The isolated request receives the capability-derived typed search
tool, compact server-generated Catalog capabilities, the current shopper
message, bounded sanitized validator feedback in a separate Human data message,
and the complete active shopper-skill instructions. Echoed rejected arguments
are stripped; native Pydantic feedback is reduced to rejected top-level field
names. Invalid AI/tool history and earlier conversation history are absent.
Only `search_catalog_tool` remains exposed, tool choice stays automatic so the
model can signal clarification by returning no tool call, and parallel calls
stay disabled. A no-tool repair is branch/control state: the server marks it,
discards the model prose, and emits `Could you clarify the product type or
requirement you want me to use?`.

The model may correct semantic fields, including `requested_product_type` and
taxonomy. Runtime validates the repaired call independently against structural
and capability invariants and never rewrites those fields. Independently valid
finite structural fields—advertised `required_constraints`, `scope_complete`,
and `search_mode`—may be preserved across the repair; bounded
`restored_fields` diagnostics identify any restoration. Malformed or nonempty
free-form `unadvertised_requirements` closes without repair or semantic
provenance review. Once the repair is used, a later invalid catalog call closes
to tools-disabled synthesis. A successful partial repaired search may continue
with later valid work, but no second repair is available in that turn.
Completed scopes and `STOP_TOOL_USE` outcomes retain their existing bounded
termination behavior. A successful or zero-result search that consumes the
final configured search slot records `SEARCH_BUDGET_EXHAUSTED`; the next model
step removes only `search_catalog_tool`. Product-detail, availability, and cart
tools plus honest partial synthesis remain available.

Grounding reads only tool-role messages and partitions current-turn evidence by
the server-owned request marker. Prior-turn tool evidence may resolve a direct
reference but cannot prove that a new search or mutation ran. For search-only
turns, successful search results carry `SEARCH_DIRECTION_EVIDENCE`: the
model-authored semantic query used as an independent internal ranking preference,
plus required pre-retrieval `shopper_guidance` authored under the active skill.
A closed search gets one tools-disabled synthesis under that skill and then the
grounding editor. Static skill `response_guidance` and pre-retrieval guidance
support deterministic fallback. If the requested outcome depends on a
functional product property absent from evidence, grounding discloses the gap
and frames candidates as the closest catalog or styling direction rather than
as proven suitable; deterministic fallback makes the same generic disclosure.
Results, taxonomy, filters, semantic query,
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

Weather is a stricter exception: diagnostics replace every raw
`get_weather_forecast_tool` argument object with `{"redacted": true}` and omit
its result. The call record retains only categorical `request_shape`,
`location_source`, `provider_input`, and `outcome`; it never includes a place,
ZIP, date, resolved place, URL, body, or exception.
Receipt handling exposes only a categorical lifecycle value such as
`promotion_prepared` or `bound`; receipt IDs, exact scope, and evidence are not
diagnostic fields.
Failed-turn partial graph capture also replaces weather calls and output with
redacted placeholders, and diagnostics recursively scrub saved profile ZIP
from string keys and values. A grounded final forecast summary
remains ordinary durable assistant text and can be exactly replayed, but a
recognized prior summary is replaced with a redaction marker in later graph
and grounding-editor recent discussion. The bounded receipt projection is a
separate structured lane and becomes grounding evidence only after explicit
current-turn binding.

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
| `get_weather_forecast_tool` | `read_only_weather` | Current Visual Crossing forecast through the provider-neutral client; successful normalized evidence may enter the bounded typed-receipt projection | Registered |

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
| `read_only_weather` | Reads bounded live forecast evidence for a qualified typed location/date scope without changing shopper or commerce state. | Granted only by event context beside outfit styling; an atomic resolution that completes the scope or explicit refresh opens one zero-argument model-visible attempt, while other unchanged turns block it; a valid call allows at most two provider attempts for timeout/5xx only; current success or one explicitly bound valid exact-scope receipt may ground weather. |
| `future_high_risk` | Checkout, payment, orders, account changes. | Not registered. Requires stronger auth, idempotency, ownership checks, and confirmation policy before use. |

## Registered Tools

### `activate_shopper_skills_tool`

Purpose: Select the registered shopper skills whose complete instructions must
govern the current turn.

Inputs:

- `skill_names`: A non-empty list drawn from the registered skill-name enum.
  The runtime removes duplicates, and the model should choose the smallest set
  whose descriptions cover the complete current intent.
- Event-context controls are capability-scoped. If `skill_names` omits
  `event-context`, the boundary removes
  `event_context_next_question`, `weather_scope`, `weather_refresh`, and
  `weather_receipt_id` before nested validation. They are inert and cannot
  mutate weather state, grant the weather tool, or reject the primary
  activation.
- `event_context_next_question`: Required exactly when `event-context` is
  selected and omitted otherwise. The activation model normally selects one
  of `event_location`, `event_venue`, `event_date`, or `none` from the current
  and recent shopper conversation; when the current shopper turn contains a
  bounded date accepted by the shared date-authority parser, the per-turn enum
  excludes `event_date`. A date only in prior raw turns remains available to
  the semantic procedure and weather tool but cannot narrow activation. Use
  `event_location` only when destination is
  missing and material. Use `event_venue` only after destination is established
  when venue or setting is missing and material. Use `event_date` only after
  destination and any material venue are established, live weather is enabled
  and material, and a bounded date is neither established nor explicitly
  unavailable. Use `none` otherwise. An explicitly shopper-stated outdoor
  patio, beach, garden, rooftop, or open-air setting makes enabled live weather
  material; with destination and that setting but no bounded date, select
  `event_date`. Skill selection, location, venue, materiality, and intent
  remain model-owned semantic guidance; the dynamic enum is typed argument
  consistency, not an intent router or keyword routing layer. A destination
  does not establish beach,
  outdoor/indoor setting, or terrain. For example, `Cancun` with no setting
  selects `event_venue`; after the shopper says `on the beach`, enabled live
  weather is material and `event_date` is selected when the date remains
  neither established nor explicitly unavailable.
  A durable pending question is stamped with the shopper turn that originated
  it. When the shopper continues product work without answering it, choose
  `none` rather than repeating that already-asked question; the binding remains
  available for a later direct answer.
- `weather_receipt_id`: Optional and dynamically restricted to IDs in the
  current valid receipt projection. It is accepted only with `event-context`
  and `event_context_next_question=none`, and only when the shopper is
  continuing the exact same event location/date scope without requesting a
  refresh. Omit it for a correction, uncertainty, or different scope.
- `weather_refresh`: Boolean, default `false`. Set it only when the shopper
  explicitly asks for a fresh forecast for an unchanged complete scope.
  Comparisons and other turns leave it false. It requires `event-context`,
  `event_context_next_question=none`, no `weather_scope`, and no receipt.

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
- When `event-context` is selected, that marker also includes the
  model-visible additive-boundary reminder: a reply that only supplies the
  destination, venue, or date requested in the prior response fulfills context
  without repeating non-weather product work, while explicit same-turn
  comparison, refinement, replacement, search, check, cart, or policy work
  continues normally. This semantic reminder does not alter grants or dispatch
  authorization.
- On the next model step, the complete selected files are injected into system
  context and only their `tools_granted` union becomes available.

Side effects:

- Activates static instructions for the current request only.
- Adds the injected paths to `agent_diagnostics.skill_files_read`.
- Accepted `event_location` or `event_venue` hides and execution-blocks
  weather. Missing location or date authority may also deny weather, but does
  not revoke any business tool from the selected skills' additive grant union
  or close the primary skill's normal tool loop.
- A valid `weather_scope` copies the singleton revision and resolves both
  components through explicit `retain`, `set`, or `clear` actions. Current-turn
  provenance is required for every `set`, and an accepted missing location/date
  question becomes the durable pending binding.
  A valid `weather_receipt_id` is
  mutually exclusive with that update, must exactly match the effective scope,
  binds that one receipt for grounding, and hides a new weather call. Every
  unbound receipt remains non-evidence.
- Supplies the accepted `event_context_next_question` as the only
  event-context question boundary to final response handling. The server does
  not infer a question from weather configuration or missing context.
- A scope resolution that produces a complete scope requires the zero-argument
  weather call before accepting prose. For an unchanged complete scope, only
  explicit `weather_refresh=true` requires it; comparisons and other turns
  block weather.
- Makes no catalog, cart, policy, availability, or other external service call.

Failure behavior:

- Unknown, empty, or invalid skill content fails closed and exposes no shopper
  commerce tools.
- A model response that tries to finish without activation fails the turn with
  final termination reason `skill_activation_failed`.
- A model response that tries to finish while the server-required weather call
  is pending fails closed; it cannot substitute prose for the tool result.
- Same-batch commerce calls are rejected with
  `skill_activation_required`; a successful activation from an earlier request
  does not satisfy the current turn.
- A post-activation tool outside the selected grant union is rejected with
  `SHOPPER_SKILL_TOOL_NOT_GRANTED` and diagnostic reason
  `skill_tool_not_granted`.
- When `event-context` is selected, a missing or invalid
  `event_context_next_question` receives the typed activation correction path
  and runs no shopping tool. When it is not selected, all extra event-context
  question, scope, refresh, and receipt controls are discarded before nested
  validation, so they do not consume a correction.
- With `event-context` selected, an unavailable receipt ID or invalid
  receipt/question composition receives the same one-correction fail-closed
  activation treatment.

Current limitations:

- The activation gate enforces both the two-phase ordering boundary and the
  selected skills' exact tool grants. Explicit mutation-intent authorization is
  not part of Slice 0.
- The required selection phase normally adds one bounded model step to each
  turn. An invalid composition may add one corrective model step.

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
- Its model-visible description repeats the event-context additive boundary:
  do not call catalog search when a reply only supplies the destination, venue,
  or date requested in the prior response; retain established candidates
  instead. If that same reply explicitly asks for comparison, refinement,
  replacement, new product work, a check, cart work, or policy help, follow the
  normal selected-skill procedure. This guides model tool choice and is not a
  deterministic handler-side intent rule or denial gate.
- The agent semantically maps the shopper's request to exact advertised
  taxonomy values, hard-filter properties and enum values, and typed numeric
  range shape through the capability-derived flat schema.
  Runtime validates structural shape, exact capability values, and
  category/subcategory ownership, then maps valid values to catalog fields. It
  does not maintain taxonomy keyword aliases or compare shopper wording with
  the submitted scope.
- The model owns whether one category-only scope faithfully represents the
  requested role. When it selects that shape, the tool emits neutral structured
  evidence containing the requested role and advertised category searched.
  The evidence does not assert that the role is unavailable, prove the category
  is its parent, or relabel returned products. If the model cannot choose a
  faithful advertised scope, the agent clarifies without retrieval or a
  catalog-absence claim.
- A zero-result retrieval reports only that its exact advertised taxonomy and
  filter scope returned no matches. It cannot establish absence for a different
  product type or the whole catalog.
- A product must-have absent from the generated schema belongs in
  `unadvertised_requirements`. Any nonempty lane fails closed before retrieval.
  Runtime does not inspect shopper wording to decide whether the requirement
  was explicit or inferred and does not open a constraint-provenance review;
  that semantic distinction belongs to the active skill and model before the
  call.
- Capability-derived taxonomy mapping and all other hard-filter validation
  complete before retrieval. Unsupported fields, values, operators, and
  category/subcategory combinations stop the search instead of becoming
  semantic relevance.
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
  support deterministic fallback. Unconfirmed functional properties remain
  explicit in both grounded synthesis and deterministic fallback, which frames
  the products as the closest direction rather than as proven suitable.
  Prohibited outdoor/weather guarantee wording
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
  invalid search request, or catalog failure. An exact-scope no-product result
  is recorded only as `zero_results`.
- When the model cannot select a faithful advertised scope, the assistant asks
  one concise clarification directly instead of calling this tool.
- Returns without calling catalog search when a required constraint cannot be
  enforced by the active capabilities. Any nonempty
  `unadvertised_requirements` lane fails closed; its fixed shopper-safe response
  applies only when that rejection is the sole current-turn business-tool
  outcome.
- The turn receives one structural repair total. The repair model may correct
  semantic fields, but runtime performs no shopper-prose classification or
  semantic scope lock. After that repair is consumed, a later invalid catalog
  call closes to synthesis. Later valid role searches may continue within the
  configured cap of three successful searches. A successful or zero-result
  third search carries `SEARCH_BUDGET_EXHAUSTED`, so no fourth search is
  possible while non-search tools and synthesis remain available.
- A completed scope, unsupported taxonomy or requirement, repeated scope,
  exhausted detail budget, or other `STOP_TOOL_USE` result closes further tool
  use. Search-budget exhaustion removes only `search_catalog_tool`, as described
  above. Completed turns receive one tools-disabled synthesis from collected
  evidence. Search-only drafts then pass through grounding, with deterministic
  rendering as fail-closed fallback. The grounding editor receives only the
  remaining shared model-stage deadline. A timeout finalizes as failed with
  `grounding_timeout`; a structurally selected context-only event turn uses
  deterministic event assembly, while current product-detail evidence uses a
  deterministic verified-detail renderer containing only current names, prices,
  categories, and listed fields, followed by a typed weather outcome when
  present. Only a current result named `get_product_details_tool` whose content
  starts with the canonical successful-detail marker qualifies. It does not
  invent comparative judgment. Other non-search turns receive a fixed
  retry/cart-check response instead of the unverified draft.
  Other ordinary editor failures and empty or whitespace-only successful editor
  responses use the same evidence split with `grounding_error`; invalid
  structured context-only event output instead falls back deterministically.
- The protected event decision renderer is selected only when `event-context`
  is active, no current non-weather business-tool activity occurred, and a
  current typed weather outcome (success or failure) or explicitly bound valid
  receipt exists. Missing
  location/venue or an empty draft skips its decision editor. A separate
  prior-candidate fallback uses deterministic event assembly only when the
  draft is empty. Product comparison with current resolution/details remains
  on ordinary grounding and may use a bound receipt silently. Other protected
  weather-evidence turns give that
  editor only bounded shopper-authored event text and the server-owned
  deterministic weather styling direction. Any attempted current non-weather
  business tool keeps the response on normal evidence grounding.
  After successful weather, the same current non-weather activity prevents
  response postprocessing from restoring unrelated historical-product names.
  For a successful event-context search, edited text must preserve at least one
  exact returned product or deterministic grounded rendering restores the
  missing candidates.
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
- For an explicit two-product comparison, the model calls this scalar tool once
  per uniquely resolved ref in separate model steps before answering. The
  default cap of two fits one pair.
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
- Established-product comparison does not itself authorize catalog search.
  Weather may be additional event evidence but cannot replace either detail
  read or establish product performance.
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
- Calls are scalar and sequential. More than two products cannot all receive
  detail reads under the default cap; a later call returns `STOP_TOOL_USE`.

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
- For a comparison, submit every compared prior product together in this one
  batched call. Do not resolve one member and search again for another.
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
- Under the comparison procedure, any required ambiguous or missing member
  produces one concise clarification and no substitute search.

Skills that grant this tool:

- `product-discovery`
- `outfit-styling`
- `cart-management`

Current limitations:

- Matching is exact after trimming and case normalization; no fuzzy or semantic
  matching is implemented.
- Resolution is limited to durable presented-product events in one conversation.
- Submitting every comparison member together and requiring a complete pair is
  model-owned skill procedure. The deterministic resolver reports each
  descriptor independently and authorizes only its unique matches.
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

### `get_weather_forecast_tool`

Purpose: Read one bounded live daily forecast for event or non-event styling
after the typed current location/date scope is complete.

Inputs:

- None. The model-visible schema is empty. The request-bound adapter reads the
  effective `CurrentWeatherScope` compiled during activation. Location/date
  values cannot be supplied or changed in this call.

Preconditions:

- `event-context` and `outfit-styling` are both active.
- Activation has already bound `event_context_next_question`. Only the value
  from the successfully completed activation is trusted as the response's
  event-context question boundary. `event_location` or `event_venue` denies
  this weather tool. Missing location or date authority may also deny weather,
  but no such denial removes product, cart, or policy capabilities granted by
  the other selected skills or closes their normal tool loop.
- `confirmed_saved_zip` is allowed only when the server's narrow deterministic
  gate accepts a current location-neutral statement explicitly naming
  `my`/`the` usual/home area, a bare affirmative immediately after the
  assistant's usual/home-area question, or an immediate strict date-only
  follow-up to an accepted confirmation.
- `shopper_provided_location` is allowed only with the exact bounded phrase
  supplied in the current shopper turn. A current explicit destination overrides and
  forbids fallback to saved ZIP. `location` remains that provenance authority
  even when `location_query` supplies the provider-facing place assumption.
- Any explicit current place, question, negation, uncertainty, or
  location-override cue rejects saved mode. A grounded shopper-authored place
  can instead authorize `shopper_provided_location`.
- Modal lowercase `may be` is an uncertainty cue, while calendar `May 5`
  remains a valid date.
- `relative_date=next_week` is allowed only when the shopper used the exact
  phrase `next week`. Exact `<weekday> next week` requires a matching
  `weekday`; the wrapper derives that one exact day inside the next
  Monday-through-Sunday window from the turn's captured UTC date. Bare `next
  week` requires no `weekday` and derives the full range. Omitted, invented,
  mismatched, mixed, negated, or superseded weekday authority fails closed.
  Without a complete effective location/date scope, the runtime hides and
  execution-blocks this tool for the turn. A direct date question is permitted
  only when accepted activation selected `event_date`; the server does not
  infer it from enabled weather or missing date authority.
  The model resolves an unambiguous
  single-day phrase such as `tomorrow` against that same anchor into an exact
  ISO date. Server UTC rather than caller/shopper local time is an explicit
  current limitation. Other ambiguous or unresolved relative dates, an unconfirmed saved
  area, and location-independent requests do not authorize a call.
- The request-bound server guard permits one zero-argument model-visible
  attempt when this activation's transition produces a complete scope or
  `weather_refresh=true` explicitly requests fresh evidence for an unchanged
  complete scope. Other unchanged turns deny the tool. Within a scope-valid call,
  `max_provider_attempts: 2` permits one internal retry only after timeout
  or HTTP 5xx. HTTP 400 maps to generic `weather_request_invalid`; other 4xx,
  connection, and response-validation failures are not retried.

Outputs:

- On success, bounded normalized daily current-turn evidence containing the
  provider, fetch time, requested window, forecast days, and Visual Crossing
  attribution. A same-ID successful call/result pair is eligible for atomic
  receipt promotion only if the turn finalizes completed.
- Optional receipt-promotion conflicts are preserved through the memory HTTP
  client and retried once without promotion. Scope revision, resolution, or
  status conflicts instead terminalize the turn as failed without the disputed
  scope write.
- For `shopper_provided_location`, the model-visible projection includes the
  provider-resolved place and deterministic final rendering discloses it as the
  forecast-location assumption, making any `location_query` qualification
  reversible. That field is omitted for
  `confirmed_saved_zip`; provider timezone remains omitted in both modes.
  Current successful rendering appends exactly one canonical block containing the resolved
  exact date for `<weekday> next week` or the Monday-through-Sunday range for
  bare `next week`, every validated day's date, condition, available low/high
  temperature, precipitation probability/types,
  [Weather Data Provided by Visual Crossing](https://www.visualcrossing.com/),
  and the warning that forecasts can change and should be rechecked closer to
  the event. Model-authored prose cannot shorten or selectively omit it. A
  later product-comparison turn that binds this exact-scope receipt uses the
  weather direction silently and does not repeat exact forecast facts or this
  block.
- When current non-weather business-tool activity exists, the response follows
  normal business-evidence grounding. In that path, editor sentences containing
  weather-domain fact language or fact-shaped dates/values are removed while
  ordinary grounded styling language remains. Successful-weather
  postprocessing does not restore unrelated historical-product names, and an
  editor failure preserves current verified detail facts plus the typed weather
  output.
- The protected event decision renderer is selected structurally only when
  `event-context` is active, no current non-weather business-tool activity
  occurred, and a current typed weather outcome (success or failure) or
  explicitly bound valid receipt exists.
  Missing location/venue or an empty draft skips that editor. A separate
  prior-candidate fallback uses deterministic event assembly only when the
  draft is empty. A comparison with current product resolution/details stays
  on ordinary grounding. Other protected weather-evidence turns must return
  exactly one JSON object with exactly `venue_quote` and `adjustments`.
  `venue_quote` must be a trimmed,
  single-line, 1–80-character exact case-insensitive substring of bounded
  shopper-authored event text. `adjustments` must contain one or two distinct
  values from `streamlined_accessories`, `lower_profile_footwear`,
  `polished_unfussy_finish`, and `adaptable_finishing`. A null/missing or
  non-shopper quote, malformed JSON, extra key, wrong cardinality, duplicate,
  or unknown code falls back.
- The server maps valid adjustment codes to fixed phrases, escapes the
  canonical shopper quote, and deterministically assembles exact names from the
  newest historical candidate set, the fixed venue sentence when valid, its
  weather styling direction, only the accepted next question, and a current
  typed weather failure or current canonical success block. The editor receives no candidate
  evidence or free-form draft and authors no shopper-facing prose.
- On failure, a sanitized typed outcome without provider body, raw exception,
  request URL, key, requested place/ZIP, dates, or resolved location.

Side effects:

- Consuming the one weather attempt does not revoke or close the loop for
  normally granted product, cart, or policy tools. Their independent tool
  policies, limits, and synthesis behavior remain in force.
- No application-state mutation. When `WEATHER_ENABLED=true`, one qualified
  call may make at most two attempts against the configured
  [Visual Crossing Timeline API](https://www.visualcrossing.com/resources/documentation/weather-api/timeline-weather-api/)
  endpoint through the existing HTTP dependency; no vendor SDK or MCP server
  is required.
- A validated success may be proposed to memory for completed-turn
  finalization. Memory applies the configured TTL, exact-scope supersession,
  and four-receipt cap atomically with turn output and replay identity.

Failure behavior:

- Disabled/configuration, validation, authentication, rate-limit, timeout,
  availability, horizon, and provider-response failures produce bounded
  shopper-safe behavior and no weather claim. HTTP 400 never proves the
  shopper's place is wrong. Existing candidates remain visible with
  conditional styling/recheck guidance, and lookup failure alone never asks
  for state, region, country, or a finer location.
- Raw tool arguments and output are redacted from diagnostics and failed-turn
  partial graph messages. Only the categorical weather summary named above
  remains. Saved profile ZIP is recursively scrubbed from diagnostic string
  keys and values.
- Failure, blocked/failed finalization, raw provider request/response data, the
  prepared provider endpoint URL, key, exception, and saved ZIP digits never
  enter the receipt projection. Validated evidence retains the pinned public
  attribution URL.

Skills that grant this tool:

- `event-context` only, with its required `outfit-styling` composition.

Current limitations:

- Current successful forecast evidence is authoritative first. Otherwise only
  one explicitly bound, unexpired exact-scope `weather_forecast.v1` receipt
  may be reused. Unbound receipts, prior assistant summaries, and prior weather
  tool messages are non-evidence; the summaries remain exactly replayable but
  are redacted from later graph/editor discussion.
- Weather can guide general styling judgment but cannot establish climate,
  venue, dress code, local norms, product warmth, waterproofing,
  breathability, comfort, safety, surface performance, or an unstated catalog
  constraint.
- There is no weather-specific FastAPI route, SSE event, request/response
  field, or UI component.
- `WEATHER_ENABLED=false` is the default. Before enabling shopper traffic, the
  operator must confirm that the selected Visual Crossing plan permits the
  intended attribution, display, storage, and sharing, including durable final
  assistant summaries and downstream app-model/output-guardrail processing.
  Review the current
  [pricing and edition terms](https://www.visualcrossing.com/weather-data-pricing/)
  and [service terms](https://www.visualcrossing.com/weather-service-terms/);
  this repository does not claim which plan is selected.

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
| `event-context` | `get_weather_forecast_tool` (only with `outfit-styling`) |
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
