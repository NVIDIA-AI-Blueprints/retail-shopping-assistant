# Shopper Agent Leadership Note: Flow and Next Steps

## Executive Summary

The shopper agent now has clean boundaries:

- the **published catalog** owns product truth;
- **conversation state** supplies continuity but cannot override current data;
- **skills** provide versioned behavior for an intent;
- **typed tools** are the only path to catalog, cart, availability, or policy
  actions; and
- deterministic validation rejects unsupported taxonomy, hard filters, and
  product evidence, while selected skills now grant only their declared tools.

This gives us a scalable way to add skills and tools without rebuilding the
agent. Slice 0 establishes the tool-binding foundation. The next investment is
the regression boundary and explicit server-owned turn authorization. Durable
turn storage now also preserves products actually presented and supports exact,
same-conversation historical resolution without making the graph checkpoint a
memory database.

## Current Request Flow

```mermaid
flowchart LR
    A[Shopper request] --> B[Turn state and identity]
    B --> C[Start durable turn; load recent turns and cart]
    C --> D[Load recent turns and presented-product index]
    D --> E[Activate skills and bind grants]
    E --> F[Model proposes typed tool call]
    F --> G[Authorize and validate]
    G --> H[Catalog, memory, cart, or policy owner]
    H --> I[Grounded response and durable finalize]
```

1. **Scope and start the turn.** The API creates one request identity containing
   session, conversation, cart, and request identifiers. Before guardrails,
   model, or tool work, the memory service transaction creates the ordered
   durable turn and returns an opaque attempt token, or exactly replays a
   matching finalized request.
2. **Load continuity.** A new turn start returns bounded model-context-eligible
   raw shopper/assistant turns, a compact index of products actually presented,
   and the authoritative cart from SQLite. Blocked turns stay durable for exact
   replay and audit but never enter this context. Those turns replace the legacy
   rolling context blob. LangGraph creates separate process-local working state
   for only this request under a collision-safe pair of conversation ID and
   request ID.
3. **Load the data contract.** The chain server reuses its process-lifetime
   catalog-capability snapshot. That contract advertises the exact taxonomy,
   hard filters, ranges, and retrieval modes available from the current catalog.
4. **Activate behavior first.** The model's first step can only select shopper
   skills. The runtime injects the complete selected `SKILL.md`; only the union
   of those skills' declared tool grants is exposed on the following model step.
5. **Propose one typed action.** The model chooses a catalog, conversation-
   product, cart, availability, or policy tool and supplies its structured
   arguments. Skills are tool permissions and behavioral instructions, but do
   not access databases. The historical resolver is conditional: it runs only
   when a needed product is absent from current-request evidence, batches all
   needed references, and is enforced at most once per turn. Its compact index
   keeps the newest complete candidate sets within 16,384 serialized characters.
6. **Validate before execution.** Runtime middleware checks the request against
   the selected-skill grant and immutable tool policy, then applies advertised
   catalog capabilities, refs, service state, turn limits, and duplicate scopes.
   A catalog scope gets at most one isolated repair;
   independently valid finite fields are preserved, while semantic corrections
   remain the model's responsibility.
7. **Call the owning service.** Catalog tools call the catalog retriever, which
   applies hard filters and ranks Milvus candidates. Cart tools call the memory
   service. A typed reference batch resolves deterministically against that
   conversation's durable presented-product events: exactly one match is usable,
   while zero or many require clarification. Policy lookup reads disabled-by-default operator content. The
   availability tool is a deterministic no-I/O stub for known product refs; it
   applies a fixed sized-versus-one-size category rule.
8. **Finalize and return grounded evidence.** The response boundary uses only
   current tool evidence for new results or mutations. It finalizes the durable
   turn with the current attempt token as completed, blocked, or failed with
   replay output and ordered event envelopes. Ordered product cards produce one
   `candidate_set_presented` event and refresh the compact reference index in
   that same transaction. After a successful commit, the runtime deletes the
   request checkpoint, then emits the response and diagnostics. Only the latest abandoned turn can reopen, and its token rotates;
   a stale finalizer receives a safe response with no stale products. A generic
   finalize outage keeps the grounded response and request checkpoint and
   adds an operator diagnostic.

## Where Data and State Live

| Store | What it owns | Current lifecycle |
| --- | --- | --- |
| Product JSONL + schema sidecar | Product records, taxonomy roles, filter roles, prices, and details | Operator-published catalog snapshot |
| Milvus | Vector candidates for the active catalog snapshot | Rebuilt or reused from the catalog fingerprint |
| Memory-service SQLite | Ordered raw shopper/assistant turns, exact finalized replay, presented-product events and compact index, typed same-conversation resolution, authoritative cart, product/cart-line identity, and atomic mutation replay | Named-volume persistence for one memory-service replica; retention is operator-owned |
| LangGraph `MemorySaver` | Working graph messages and tool state keyed by a collision-safe conversation/request pair | Process-local; deleted after successful durable finalize and retained only on finalize failure |
| Turn `State` | Query, media, context, cart, current evidence, timings, and diagnostics | Transient for one request |
| Shopper skills | Reviewed behavioral instructions, response framing, roles, and tool grants | Versioned repository files; no customer or product state |

Memory is guidance, not truth. A remembered statement that an item was added or
is available cannot override the current cart or a current catalog result.
Raw turns and products actually presented now survive a chain-server restart
through the memory service. The compact product index is read-only model context;
the full event payload is the resolver's authority. A unique typed match becomes
request-local evidence for details, availability, or cart add. Missing or
ambiguous references require clarification. New-conversation requests such as
“show me the bag from last week” remain unsupported. Preferences, sentiment,
active anchors, fuzzy matching, embeddings, and stale-revision handling are not
implemented.

## Example: “What Bottoms Go With That Beige Top?”

Assume the previous turn returned a group of tops confirmed by the catalog's
`primary_color=beige` filter.

1. The earlier top cards were finalized as one ordered
   `candidate_set_presented` event. The next turn receives their compact names,
   refs, positions, and turn coordinates even after a chain-server restart.
2. Skill activation selects `outfit-styling`; it does not select
   `product-discovery` merely because the shopper named one product role.
3. If “that beige top” identifies exactly one earlier card, the styling skill
   calls the typed resolver. One match adds that top to request-local evidence;
   zero or multiple matches stop for one concise clarification.
4. The model reads the advertised catalog taxonomy. In the current snapshot,
   `skirts` is the only published child that is a kind of bottom; pants,
   trousers, and shorts are not advertised. Dresses are excluded because a
   dress is not a bottom.
5. The model calls `search_catalog_tool` once with:

   - requested type: `bottoms`;
   - taxonomy relation: member of the requested umbrella;
   - taxonomy: `apparel/skirts`;
   - semantic direction: skirts that balance the beige top; and
   - hard constraints: none, unless the shopper also asked for beige or a
     same-color look.

6. Deterministic validation confirms that every selected taxonomy value is an
   advertised child of the requested role. The catalog then ranks only products
   inside that hard skirt scope.
7. The response presents grounded skirt candidates and explains their role in
   the beige-top outfit. It does not claim that the skirts themselves are beige
   and does not introduce dresses as substitutes.

This is the intended division of labor: the skill supplies styling procedure,
the model maps intent to the published contract, validation enforces the
contract, and the catalog supplies product truth.

## What Comes Next

### 1. Add explicit turn authorization against the locked regressions

Slice 3 closes the mutation-retry cases with one memory-service transaction
boundary for add, remove, and quantity update. Slice 0 proves that only
`cart-management` can expose cart mutators; it does not yet prove that the
shopper requested a particular mutation. That narrower authorization boundary,
invented-constraint assurance, and live quality validation of ambiguous
historical references remain separate work.

### 2. Harden durable product identity only when needed

The minimal resolver deliberately stores only cards actually shown and performs
exact same-conversation matching. The next identity work is catalog-revision
invalidation or a stable upstream product ID if catalog replacement becomes a
real operating requirement. Do not add fuzzy matching, embeddings, inferred
preferences, or a transcript summarizer pre-emptively.

### 3. Select a production durable memory store

Request-scoped `MemorySaver` is no longer shopper memory, so a shared graph
backend is not the immediate durability gate. The production gate is the
single-replica SQLite service. Choose an Apache-2.0/MIT-compatible shared,
multi-writer database before scaling memory-service replicas, with explicit
retention and deletion policy for transcript and product-reference events.

### 4. Extend the long-conversation quality gate

Add fixed scenarios for returning to earlier candidates, ambiguous pronouns,
ordinal references, rejected searches followed by references, restart behavior,
and catalog-fingerprint changes. Reconcile catalog-sensitive Golden answers
with the active published snapshot so safety improvements are not scored as
inventory errors.

### 5. Add new domains through the same boundaries

- New **behavior** becomes a skill.
- New **action or read capability** becomes a typed tool with a clear data owner.
- New **durable fact or state** belongs in that owner's service, not in a skill
  prompt or process-local cache.

The fixed representative-shopper registry now supplies one trusted, typed
source, but Slice 1 does not bind it to conversation context. Any later
representative-profile binding, user-owned persona support, live inventory, and
additional policy content should use explicit trusted data contracts; none
should be inferred from conversation text.

## Recommendation

The styling boundary is focused, cart idempotency is memory-service owned, and
durable turn start/finalize/replay plus exact presented-product resolution are
built for one SQLite replica. Deliver cart-only intent authorization, catalog-
revision handling when required, and a shared production memory-service store
as separate work. The catalog remains unchanged. The trap is expanding the
minimal exact resolver into inferred preferences, fuzzy matching, or keyword
taxonomy patches; those would make each new skill harder to extend safely.

For implementation detail, see the
[Shopper Agent Architecture](SHOPPER_AGENT_ARCHITECTURE.md),
[Skill Registry](SHOPPER_AGENT_SKILL_REGISTRY.md), and
[Tool Registry](SHOPPER_AGENT_TOOL_REGISTRY.md).
