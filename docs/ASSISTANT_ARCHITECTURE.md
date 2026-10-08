# Assistant Architecture

How one shopper turn is served, and which layer is allowed to decide what.
This is the map. [Skills](SKILLS.md) and [Tools](TOOLS.md) are the registries,
[Catalog Architecture](CATALOG_ARCHITECTURE.md) covers ingestion and the
published contract, and [API Reference](API.md) covers the wire format.

## At a glance

![Assistant architecture](images/shopper-agent-architecture.svg)

[Open the full-size SVG](images/shopper-agent-architecture.svg).

## Who owns what

The design is a set of ownership boundaries. Most rules in this document follow
from one of them.

| Boundary | Owns | Does not own |
| --- | --- | --- |
| Published catalog | Product records, taxonomy, filter values, prices, retrieval results | Shopper intent, styling judgment, cart state, inventory |
| Deep Agents runtime | Semantic intent, skill selection, tool selection, styling judgment | Product facts, policy facts, cart truth |
| Memory service | Durable turns, cart truth, presented-product index, the shopper registry | Catalog facts, model reasoning, cross-conversation memory |
| Graph checkpointer | Request-scoped graph state inside one process | Durable transcript, cross-turn memory, cross-replica context |
| Guardrails (optional, on by default) | Content safety and topic checks on input and output | Product facts, or authorization of any tool |
| Weather (optional, off by default) | Forecast evidence for a place the shopper named | Looking up the saved ZIP on its own, or any persistence |

Both optional boundaries are configured in
[Deployment](DEPLOYMENT.md): guardrails under Guardrails, weather under Weather
Tool. Disabled, neither registers a tool and neither is called at startup or on
a health check.

## One shopper turn

1. **Scope and start.** The runtime opens a durable turn in the memory service
   before any guardrail, model, or tool work. The start transaction returns
   bounded recent turns, the compact historical-product index, the previous
   turn's selected skills, the authoritative cart, and an `attempt_id`.
2. **Activate skills.** The first model step may call only
   `activate_shopper_skills_tool`, and selects the smallest skill set that fits
   the intent. An invalid composition gets one typed correction; a second ends
   the turn with a fixed clarification and exposes no shopping tool.
3. **Inject the selected skills.** Their `SKILL.md` files are injected whole.
   The union of their `tools_granted` is what becomes visible to the model on
   the next step, so activation cannot be skipped or batched with real work.
4. **Dispatch tools.** Every dispatch independently rechecks the grant against
   an immutable policy before a handler runs. Using a product from an earlier
   turn requires a typed resolver call first, which is deterministic and makes
   no model or catalog call; only a unique match becomes usable evidence.
5. **Search.** Requests pass a capability-derived schema and deterministic
   validation. Three searches per turn, and one repair attempt per scope, in an
   isolated context that sees only the current message and sanitized validator
   feedback. A repair may correct the request or signal that clarification is
   needed; it may not invent taxonomy or constraints.
6. **Ground the response.** Tool-role messages are the evidence boundary. A
   final tools-disabled draft is checked against that evidence, and a
   deterministic renderer takes over if the draft or the editor is unavailable.
   Properties the catalog cannot confirm — fit, comfort, durability, weather
   suitability — must be disclosed as unverified rather than asserted.
7. **Finalize.** Every terminal path records `completed`, `blocked`, or
   `failed`. An exact retry of a finalized request replays the stored text,
   products, and images without another model turn.

## Invariants

These hold regardless of model behavior, and are the rules to preserve when
changing the runtime.

- **Only catalog tool evidence establishes a product fact.** A model-authored
  semantic query is a ranking preference. Conversation memory may guide
  judgment but cannot override current tool results.
- **A product ref must already be in the current request's evidence.** Current
  search adds refs directly; an earlier product needs a unique durable
  resolution. Ambiguous or missing refs never authorize a downstream tool.
- **A skill grant is checked twice** — once when building the model's visible
  toolset, once at dispatch against immutable policy.
- **Shopper profiles are soft guidance only.** The runtime renders one block
  with type, behavior, and saved ZIP. It cannot grant a tool or establish a
  budget, requirement, cart intent, or product fact, and explicit shopper
  instructions outrank it. Caller-supplied persona objects are never injected.
- **The checkpointer is not memory.** It is request-scoped and process-local,
  survives only a finalize failure, and is deleted once the durable commit
  succeeds. `CHECKPOINT_STORE=memory` is the only supported mode.
- **Timeouts fail closed.** The graph and the grounding editor share one
  configurable 45-second model-stage deadline, and the editor gets only the
  time left. A timeout clears undelivered products and returns a fixed retry
  response rather than an unverified draft.
- **Cart mutations are idempotent.** Adds use catalog `product_id`; removes and
  quantity changes use an opaque `cart_line_id`. Each mutation commits with an
  owner-scoped idempotency record, so a retry replays instead of double-acting.
- **A blocked turn stays durable** for replay and audit, and is excluded from
  the model's context.

## Skills

Seven skills are registered. A skill is a markdown file whose frontmatter
declares its `role`, an optional `exclusive_group`, and the tools it grants —
which makes the grant an authorization boundary rather than a hint.

| Skill | Role | For |
| --- | --- | --- |
| `product-discovery` | primary, `product_procedure` | Search, browse, filters, and product facts |
| `outfit-styling` | primary, `product_procedure` | Building, completing, comparing, or refining a look |
| `catalog-questions` | primary, `product_procedure` | Questions about the shop rather than a product |
| `budget-shopping` | modifier | Adds budget procedure when a price ceiling is stated |
| `cart-management` | standalone | Cart reads, adds, removals, quantity changes |
| `store-policy-answers` | standalone | Returns, shipping, sizing, payment, gift cards |
| `destination-weather` | standalone | Conditions at a named place and time |

The three `product_procedure` primaries are mutually exclusive, so exactly one
is active per turn. A genuine multi-intent turn activates each skill it needs
once: styling under a budget with an add request uses `outfit-styling`,
`budget-shopping`, and `cart-management` together.

[Skills](SKILLS.md) lists every skill with its granted tools and the tuning
workflow. It is checked against the code by tests, so treat it as the registry
and this table as orientation.

## Tools

| Group | Source of truth |
| --- | --- |
| Catalog search, details, description | The active published catalog snapshot |
| Conversation products | Durable `candidate_set_presented` events in the memory service |
| Cart | The memory service cart |
| Store policy | Operator-managed policy YAML |
| Availability | An application stub. **Live inventory is not wired**, so availability is not a real answer |
| Promotions | An application stub, reporting no active promotion |
| Weather (only when enabled) | A provider-neutral client; Visual Crossing is the first adapter |

`activate_shopper_skills_tool` is an internal control tool, not a commerce
tool. It is forced at turn start and selects instructions; it reads and mutates
nothing.

These contracts are app-owned and deliberately independent of ACP, UCP, and any
future commerce protocol. Support for one belongs in a thin adapter around
these tools, not as extra fields inside the product and cart models.

[Tools](TOOLS.md) has each tool's input schema, risk class, and failure
behavior, and is likewise test-checked against the code.

## Where it lives

The turn is implemented in the chain server's
[runtime](../chain_server/src/runtime/runtime.py), with the
[tool policy](../chain_server/src/tools/policy.py),
[activation boundary](../chain_server/src/tools/skill_gate.py), and
[tool-loop controller](../chain_server/src/tools/loop_control.py) enforcing the
grants. Durability is the memory service's
[conversation API](../memory_retriever/src/conversations.py),
[product-reference resolver](../memory_retriever/src/product_references.py),
[shopper registry](../memory_retriever/src/shopper_profiles.py), and
[migrations](../memory_retriever/src/migrations.py).

To see a turn actually behave this way, [Observability](OBSERVABILITY.md) shows
which skill ran, what the model was told, and why a tool was refused.
