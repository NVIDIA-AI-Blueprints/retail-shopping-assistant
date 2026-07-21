# Shopper Agent Architecture

This document is the short architectural map of the serving shopper agent. It
shows where product truth is published, how skills control model behavior, and
which tools connect the agent to application services.

## Architecture at a Glance

![Shopper Deep Agent architecture](images/shopper-agent-architecture.svg)

[Open the full-size SVG](images/shopper-agent-architecture.svg).

For the concise executive flow, worked example, and prioritized follow-up work,
see the [Shopper Agent Leadership Note](SHOPPER_AGENT_LEADERSHIP_NOTE.md).

## Architectural Boundaries

| Boundary | Owns | Does not own |
| --- | --- | --- |
| Published catalog | Product records, taxonomy, filter values, field roles, prices, details, and retrieval results | Shopper intent, styling judgment, cart state, or inventory |
| Deep Agents runtime | Semantic intent, skill selection, deterministic selected-skill tool grants, tool selection, and styling judgment | Product facts, policy facts, or cart truth |
| Memory service | Ordered durable shopper/assistant turns, exact finalized replay, bounded recent-turn reads, presented-product events and compact index, deterministic same-conversation reference resolution, authoritative cart state, and atomic mutation replay records | Catalog facts, model reasoning, preferences, sentiment, active anchors, or cross-conversation memory |
| Graph checkpointer | Request-scoped working graph/tool state within one chain-server process | Durable transcript storage, cross-turn shopper memory, cross-replica context, or product-ref authorization |

A model-authored semantic query is an internal **ranking preference**, not a
product fact or shopper-facing explanation. Only catalog tool evidence can
establish catalog facts. Conversation memory may guide judgment, but it cannot
override current tool results. Caller-supplied persona data is not injected into
model context.

## 1. Published Catalog Data Foundation

The catalog starts from two operator-published files:

- [Product JSONL](../shared/data/enriched_products.jsonl) supplies product rows
  and all observed values.
- [Field-role sidecar](../shared/data/enriched_products.schema.yaml) identifies
  record fields, ordered taxonomy fields, and whether each field supports hard
  filtering, semantic retrieval, or product details. It does not duplicate
  catalog values.

One validated snapshot supplies capabilities, product details, and the data
indexed in Milvus. `GET /capabilities` publishes the exact current taxonomy,
filter values, numeric ranges, field coverage, and supported retrieval modes.
The chain server caches the first successful contract and uses it both to
generate `search_catalog_tool` inputs and to validate every structured search.

The shopper model owns the semantic mapping from the request to advertised
values. Its tool uses a structural transport schema; the runtime revalidates the
payload with a separate strict semantic search model. Cross-field failures
therefore reach the capability-aware catalog validator and produce exact
corrections instead of failing before tool execution. Deterministic code verifies
and maps valid values before retrieval. The catalog service contains no
chat/completion LLM and does not interpret shopper language, expand queries, or
choose filters. Its only model inference is text or image embedding generation;
candidate fusion, hard filtering, COSINE relevance normalization, deduplication,
and final ordering are deterministic.

Each search covers at most one catalog category and one focused product role.
Every text search carries `requested_product_type`: the shortest product noun or
true umbrella from the shopper's current turn or direct antecedent, excluding
color, material, fit, occasion, weather, and style modifiers. This field records
provenance, not taxonomy or ranking text, and is `null` only for `image_only`.
Literal validation can bind the longest exact advertised suffix in a
modifier-bearing model phrase (`waterproof boots` to `boots`). It disables that
shortcut for explicit alternatives containing `and`, `or`, `/`, or `&`, so `closed
shoes or boots` remains model-owned alternative or umbrella reasoning.
When the current shopper turn contains one unambiguous literal pair of exact
advertised subcategories from the same category, deterministic validation
requires the model-authored requested type and taxonomy to retain both branches
with `member_of_requested_umbrella`. This is exact capability matching, not
semantic interpretation. The valid scope executes once with a pair-wide
candidate window; rank-preserving selection keeps one returned candidate per
branch when available and trims to the configured result count. Modified,
synonymous, ambiguous, or cross-category alternatives remain model-owned.
The `semantic_query` remains independent soft ranking direction. The model owns
advertised taxonomy selection, including `taxonomy_status`; runtime never
semantically rewrites that status. Capability-owned exact category/subcategory
relationships validate the submitted status and selection.
`agent_selected_type` is rejected for a
shopper-named scope and is valid for a genuinely open role only when it selects
exactly one advertised subcategory. For that open role, the runtime derives the
duplicate `requested_product_type` provenance from the selected subcategory and
retains `agent_selected_type`. If an agent-selected open-role call is malformed,
deterministic validation stops before retrieval and reports the exact eligible
advertised subcategories from the current capability contract. It returns
related constraint and `shopper_guidance` defects in that same repair result.
This is bounded schema feedback, not semantic routing: the model operating under
the active skill still selects the role.
The duplicate identity is normalized taxonomy plus hard constraints, not
semantic wording. A concrete requested type with no faithful advertised value
uses the no-retrieval path with empty taxonomy and no hard constraints; an
unsupported modifier does not erase an otherwise advertised type.

The detailed contracts and implementation live in:

- [Catalog architecture](CATALOG_REFACTOR_PLAN.md)
- [Commerce contracts](COMMERCE_CONTRACTS.md)
- [Catalog loader](../catalog_retriever/src/catalog.py)
- [Capability publisher](../catalog_retriever/src/capabilities.py)
- [Catalog retriever](../catalog_retriever/src/retriever.py)
- [Chain capability cache](../chain_server/src/catalog_capabilities.py)

## 2. One Shopper Turn

1. The runtime scopes the request and starts a durable memory-service turn before
   guardrail, model, or tool work. That transaction returns bounded finalized
   raw turns, a compact historical-product index, the authoritative cart, and an
   opaque execution `attempt_id`. The raw turns replace the legacy rolling
   context blob. LangGraph working state is isolated to this request under
   a collision-safe pair of conversation ID and request ID.
2. The first model step can call only `activate_shopper_skills_tool`. It selects
   the smallest registered skill set for the current intent. The prior turn's
   selected skill names are included only as a read-only continuity signal for
   this fresh semantic decision. They do not force routing, inject the old skill,
   or authorize commerce tools. The conversation still establishes intent: a
   terse item-only follow-up inside an active outfit-building or style-led
   single-piece thread remains an `outfit-styling` task.
3. The runtime validates the selection and injects the complete selected
   `SKILL.md` files. Each file declares `role`, optional `exclusive_group`, and
   `tools_granted`; only the union of those grants becomes model-visible on the
   next step. Activation therefore cannot be bypassed or batched with shopping
   work. Each skill's required frontmatter also supplies declarative
   `response_guidance`: reviewed shopper-facing fallback framing for a search
   call whose tool evidence contains no `shopper_guidance`.
4. The model chooses only from the selected skills' granted tools. Every
   app-owned shopping dispatch independently rechecks both the frontmatter grant
   union and the immutable execution policy before its handler can run. Before
   a detail, availability, or cart operation uses a product from an earlier
   turn, the selected discovery, styling, or cart skill may call the typed
   historical resolver once with one or more exact descriptors from the compact
   index. Its 0/1/many result requires clarification for zero or many; only one
   match enters request-local product evidence. Resolution is deterministic and
   makes no catalog, embedding, or separate model call. Before
   retrieval, every search
   except `image_only` and `no_direct_catalog_match` includes one nonempty,
   product-agnostic `shopper_guidance` sentence authored under the active skill;
   those two statuses require empty guidance. Catalog requests pass through the
   capability-derived schema and deterministic validation. Tool calls are
   sequential and duplicate taxonomy-plus-hard-constraint scopes are rejected.
   Repair accounting uses
   the full normalized `requested_product_type` phrase rather than only its last
   noun, and distinct advertised siblings cannot substitute for one another.
   Each scope receives one total repair. A schema correction or a fresh
   constraint-provenance review can consume that shared budget; constraint
   feedback returned by an in-flight schema repair closes the loop for synthesis
   rather than opening another repair. The repair is isolated: a concise,
   schema-generic system prompt replaces the base runtime prompt, while the
   skill gate appends the complete active shopper-skill instructions. Only
   `search_catalog_tool` is exposed and forced, and parallel calls are disabled.
   Active-skill responses containing more than one shopping tool call are
   rejected before execution, so repair state always belongs to one call.
   Its messages contain only the current shopper
   message and bounded, sanitized validator feedback in a separate Human data
   message. Echoed rejected arguments are stripped; native Pydantic feedback is
   reduced to rejected top-level field names, and unbounded requested-scope text
   is never copied into the repair message. Invalid AI/tool history and the rest
   of the conversation are absent. For a native tool-transport failure, runtime locks the requested
   scope only when current or recent shopper text grounds it; an ungrounded
   model-generated scope may be corrected. A change to a grounded free-form
   locked scope that cannot be reconstructed safely is removed before execution
   and recorded in `agent_diagnostics` with reason `repair_scope_changed`. For
   any strict request failure with independently
   valid constraints, runtime snapshots the capability-validated advertised
   `required_constraints` privately and includes that exact finite object,
   including an explicit empty object, in the isolated feedback. This bounded
   capability-derived object is the exception to excluding free-form rejected
   arguments. Before execution, runtime restores every independently valid
   finite lock: the taxonomy relation, canonical advertised constraints
   (including an explicit empty object), explicit valid `scope_complete` and
   `search_mode`, and `requested_product_type` when a singleton exact or
   agent-selected taxonomy determines it. The model owns only invalid fields.
   Drift in a restorable lock is corrected in place; bounded tool-call
   diagnostics expose only the affected field names in `restored_fields`. A
   no-direct repair may clear
   constraints only if it remains no-direct; changing to retrieval requires the
   original advertised constraints.
   The model remains responsible for correcting taxonomy and
   `taxonomy_status`; accepted modifier removal cannot bypass the lock, and
   list-valued filters compare canonically. When native enum validation rejects an
   `agent_selected_type` call, the same bounded feedback includes whether the
   role was shopper-named or genuinely open, allowing the one repair to correct
   both schema and provenance errors. Shopper-named advertised subtypes repair
   to `exact_requested_type`; named umbrellas and alternatives repair to
   `member_of_requested_umbrella`. A no-direct terminal outcome reached after
   that repair retains the specific not-advertised response. A native
   failure confined to `required_constraints` carries only the finite, validated
   taxonomy status and selection into the isolated repair while continuing to
   exclude free-form scope, query, and guidance text. Scope comparison remains
   private. Runtime restores relation drift before the repaired constraint call
   executes. A native taxonomy failure likewise restores independently valid
   constraints before execution. A locked boundary that cannot be restored
   safely remains comparison-protected and closes under the matching
   `repair_*_changed` reason. Empty optional defaults and reordered filter
   lists compare canonically. A native
   schema-invalid call containing malformed or nonempty free-form
   `unadvertised_requirements` arguments closes without repair; a schema-valid,
   genuinely open `agent_selected_type` request retains the bounded review for
   a proposed inferred requirement. Every unadvertised requirement on a
   shopper-stated product scope fails closed before retrieval, including when
   the model uses a synonym rather than the shopper's exact wording. The bounded
   constraint review is reserved for a proposed inferred requirement on a
   genuinely open `agent_selected_type` role when the shared scope budget
   remains. It freezes requested type, taxonomy status, taxonomy, completion
   state, `search_mode`, and all advertised hard constraints. Within that
   preserved hard scope, it may correct only the soft `semantic_query`, the
   reviewed unadvertised-requirement lane, and its associated
   `shopper_guidance`; the requirement is either replaced with the shopper's
   shortest exact wording or removed. Exact wording, unresolved provenance, and
   constraint feedback after a schema repair fail safe and close the loop.
   Removal scrubs the product-attribute claim from `shopper_guidance`. When a
   runtime semantic open-role schema repair removes its
   proposed inferred requirement, runtime replaces the submitted pre-search
   guidance with neutral generic guidance for the selected role. A successful
   partial search may continue with another valid role and its own single repair
   opportunity. The configured turn cap remains three searches. A successful or
   zero-result search that consumes the final slot records
   `SEARCH_BUDGET_EXHAUSTED`; the next model step removes only
   `search_catalog_tool`. Product-detail, availability, and cart tools plus
   honest partial synthesis remain available.
5. Tool-role messages are the evidence boundary. For completed successful
   search-only turns, the runtime runs one final tools-disabled synthesis under
   the active skill, then grounds that draft against tool-role evidence. If the
   draft or editor is unavailable, pre-retrieval `shopper_guidance` and static
   skill `response_guidance` feed deterministic fallback. Before fallback
   guidance is serialized, a narrow scrub replaces documented unsupported
   outdoor/weather guarantee terms with neutral selected-role guidance. It does
   not change the semantic query, taxonomy, constraints, or executed search.
   Covered forms include outdoor-surface or outdoor-walking claims and
   constructions such as "handle rain," "work well for outdoor surfaces," or
   "stay secure for outdoor walking," plus `wet conditions` and "works well in
   wet weather/conditions."
   Results, filters, and the assistant draft are not copied into guidance after
   retrieval.
   Deterministic fallback code separately renders every returned candidate with name,
   price, category, and only the confirmed filters from that candidate's search.
   Multi-role output groups each guidance sentence with the products from its
   originating search and deduplicates candidates by `product_ref`, not display
   name. Mixed-outcome turns preserve successful product groups when another
   scope has no direct match or an unsupported requirement. A fixed no-direct
   or unsupported canned response applies only when that rejection is the sole
   current-turn business-tool outcome. Incomplete successful evidence gets a
   neutral offer to continue with the next requested piece or search scope.
   Zero-result evidence
   retains its exact search scope and cannot establish broader absence. Other
   tool-backed drafts pass through the grounding boundary before becoming
   shopper-facing text. When all current-turn business calls are rejected
   catalog searches and no current product evidence exists, a fixed retry
   response bypasses model-based editing so prior evidence cannot be recast as
   results from the rejected search.
6. The runtime finalizes the durable turn as `completed`, `blocked`, or `failed`
   on every terminal path. An exact retry of a finalized request replays stored
   assistant text, products, retrieved images, and diagnostics without another
   model/tool turn or finalize call. A start failure runs no agent work. A
   finalize must echo the current attempt token. Only the latest-sequence
   abandoned turn can reopen, and doing so rotates that token; a late finalize
   from the stale attempt is rejected and becomes a safe superseded-attempt
   response with no stale products or images. Other finalize failures do not
   replace an already grounded response; diagnostics record
   `memory_finalize_error`, and the request checkpoint is preserved.
   Memory-service operations are transactional, and database sessions are
   request-scoped and returned to the SQLAlchemy pool after every request.

   Finalization derives one `candidate_set_presented` event only from the
   ordered product cards in the terminal replay output, then rebuilds the compact
   product-reference projection in the same transaction. After that durable
   commit succeeds, the runtime deletes the request-scoped MemorySaver thread.
   `CHECKPOINT_STORE=memory` is the only supported mode. The checkpointer remains
   process-local but is no longer shopper memory; it survives only a finalize
   failure and otherwise ends with the request.

The durable transcript contains raw shopper/assistant text plus bounded replay
output and ordered event envelopes. It does not contain raw media, model
reasoning, or the full graph/tool transcript. Product-card output now populates
durable `candidate_set_presented` events and a bounded compact product-reference
index. The projection keeps the newest complete candidate sets within 16,384
serialized characters. A typed batch resolver matches exact product ref,
display name, category, turn, candidate set, and one-based position within the
current conversation. It is enforced at most once per turn and returns
`resolved`, `ambiguous`, or `not_found`; only a unique result becomes
request-local evidence. Active anchors and effective preferences remain
reserved. Fuzzy/embedding lookup, preference or sentiment extraction,
cross-conversation lookup, and stale-catalog-revision handling are not included.

The resolved agent dependency boundary remains `deepagents==0.6.12`,
`langchain==1.3.11`, `langgraph==1.2.7`, and `langgraph-sdk==0.4.2`.
Services that resolve `orjson` pin `3.11.5`, the last upstream release limited
to the project's Apache-2.0/MIT policy. Redis checkpoint packages are not
installed.

Final-response extraction ignores tool messages, assistant tool-call messages,
and internal activation markers. If no shopper-facing answer remains, the
runtime returns a safe retry response and records `incomplete_agent_response`.
Operator diagnostics also include bounded `catalog_scope_outcomes` for
no-direct and zero-result scopes.

Product refs used for detail, availability, and cart-add calls live only in the
current request's evidence set. Current-turn search adds them directly; a
unique durable historical resolution can add an earlier presented product.
Ambiguous or missing references never authorize a downstream tool. The current
slice records catalog revision metadata but does not reject stale revisions.

Cart transaction safety is owned by the memory service. Adds use catalog
`product_id`; removes and quantity updates use opaque `cart_line_id`. Each
mutation commits with its owner-scoped idempotency record, so an identical retry
replays and conflicting key reuse cannot change the cart.

The serving implementation is split across the
[Deep Agents runtime](../chain_server/src/deepagents_runtime.py),
[conversation-memory client](../chain_server/src/conversation_memory.py),
[conversation-product boundary](../chain_server/src/conversation_products.py),
[shopper tool policy](../chain_server/src/tool_policy.py),
[skill activation boundary](../chain_server/src/skill_activation.py), and
[tool-loop controller](../chain_server/src/tool_loop_control.py). The durable
SQLite boundary is implemented by the memory service's
[conversation API](../memory_retriever/src/conversations.py),
[product-reference resolver](../memory_retriever/src/product_references.py),
[models](../memory_retriever/src/models.py), and
[migrations](../memory_retriever/src/migrations.py).

## 3. Skills and Their Tools

Five shopper skills are registered. Their frontmatter grants are an
authorization boundary: the model sees only the union for the current selected
skills, and dispatch checks the same grant against an independent immutable
policy. Tool schemas and wrappers still enforce refs, filters, service state,
and mutation preconditions.

| Skill | Role | Use | Granted tools |
| --- | --- | --- | --- |
| [`product-discovery`](../chain_server/skills/shopper/product-discovery/SKILL.md) | `primary`, `product_procedure` | Non-styling search, browse, filters, and product facts | `search_catalog_tool`, `get_product_details_tool`, `check_product_availability_tool`, `resolve_conversation_products_tool` |
| [`outfit-styling`](../chain_server/skills/shopper/outfit-styling/SKILL.md) | `primary`, `product_procedure` | Build, complete, compare, balance, or refine a look | `search_catalog_tool`, `get_product_details_tool`, `check_product_availability_tool`, `resolve_conversation_products_tool` |
| [`budget-shopping`](../chain_server/skills/shopper/budget-shopping/SKILL.md) | `modifier` | Add budget procedure when the shopper states a price ceiling or bundle budget | None; combine with the applicable product or cart skill |
| [`cart-management`](../chain_server/skills/shopper/cart-management/SKILL.md) | `standalone` | Cart reads, adds, removals, and quantity changes, alone or beside product work | `get_cart_tool`, `view_cart_total_tool`, `add_cart_items_tool`, `remove_cart_item_tool`, `update_cart_items_tool`, `resolve_conversation_products_tool` |
| [`store-policy-answers`](../chain_server/skills/shopper/store-policy-answers/SKILL.md) | `standalone` | Returns, shipping, sizing, payment, price matching, and gift cards | `get_store_policy_tool` |

`product-discovery` and `outfit-styling` are mutually exclusive primary
procedures. `budget-shopping` modifies the applicable primary only when the
shopper states a budget. A genuine multi-intent turn activates every needed
skill once; for example, styling under a budget with an explicit add request
uses `outfit-styling`, `budget-shopping`, and `cart-management`.

The styling skill owns fashion procedure and clarification: anchors, color,
proportion, silhouette, formality, occasion, texture, and concise explanation.
The catalog publishes the advertised taxonomy and filter contract. The
capability-aware runtime validates and maps submitted constraints before the
catalog service performs deterministic retrieval and filtering. Cart reads and
mutations are available to styling only through a co-active `cart-management`
skill.

Slice 0 proves selected-skill tool authorization, not explicit mutation intent.
Selecting `cart-management` currently grants its cart tools; deterministic refs
and service preconditions still apply, but a server-owned current-turn intent
authorization object is a later slice.

[trends-current.md](../chain_server/skills/shopper/trends-current.md) is a
read-only seasonal reference used by `outfit-styling`; it is not a registered
skill and is not catalog truth.

## 4. Tool Ownership

| Tool group | Tools | Source of truth |
| --- | --- | --- |
| Catalog | `search_catalog_tool`, `get_product_details_tool` | Active published catalog snapshot |
| Conversation products | `resolve_conversation_products_tool` | Durable same-conversation `candidate_set_presented` events in the memory service |
| Availability boundary | `check_product_availability_tool` | Known-ref, category-aware no-I/O application stub; live inventory remains unavailable |
| Cart | `get_cart_tool`, `view_cart_total_tool`, `add_cart_items_tool`, `remove_cart_item_tool`, `update_cart_items_tool` | Memory service cart |
| Policy | `get_store_policy_tool` | Operator-managed policy YAML |

`activate_shopper_skills_tool` is an internal control tool, not a commerce
tool. It is forced at turn start and selects static behavior instructions; it
does not read or mutate catalog, cart, or policy state.

For exact input schemas, risk classes, and failure behavior, use the
[Shopper Agent Tool Registry](SHOPPER_AGENT_TOOL_REGISTRY.md). For skill
selection and tuning details, use the
[Shopper Agent Skill Registry](SHOPPER_AGENT_SKILL_REGISTRY.md).
