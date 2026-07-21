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
the regression boundary and explicit server-owned turn authorization; the
structured reference ledger and durable transcript follow on that safe base.

## Current Request Flow

```mermaid
flowchart LR
    A[Shopper request] --> B[Turn state and identity]
    B --> C[Load conversation and cart]
    C --> D[Activate skills and bind grants]
    D --> E[Model proposes typed tool call]
    E --> F[Authorize and validate]
    F --> G[Catalog, cart, or policy owner]
    G --> H[Grounded response and state save]
```

1. **Scope the turn.** The API creates one request identity containing session,
   conversation, and cart identifiers. The transient turn state contains the
   current query, media, recent context, authoritative cart, results, timing,
   and operator diagnostics.
2. **Load continuity.** The memory service reads the bounded conversation text
   and cart from SQLite. LangGraph separately loads the conversation's graph
   messages from process-local `MemorySaver` using `conversation_id` as the
   thread ID.
3. **Load the data contract.** The chain server reuses its process-lifetime
   catalog-capability snapshot. That contract advertises the exact taxonomy,
   hard filters, ranges, and retrieval modes available from the current catalog.
4. **Activate behavior first.** The model's first step can only select shopper
   skills. The runtime injects the complete selected `SKILL.md`; only the union
   of those skills' declared tool grants is exposed on the following model step.
5. **Propose one typed action.** The model chooses a catalog, cart, availability,
   or policy tool and supplies its structured arguments. Skills are tool
   permissions and behavioral instructions, but do not access databases.
6. **Validate before execution.** Runtime middleware checks the request against
   the selected-skill grant and immutable tool policy, then applies advertised
   catalog capabilities, refs, service state, turn limits, and duplicate scopes.
   A catalog scope gets at most one isolated repair;
   independently valid finite fields are preserved, while semantic corrections
   remain the model's responsibility.
7. **Call the owning service.** Catalog tools call the catalog retriever, which
   applies hard filters and ranks Milvus candidates. Cart tools call the memory
   service. Policy lookup reads disabled-by-default operator content. The
   availability tool is a deterministic no-I/O stub for known product refs; it
   applies a fixed sized-versus-one-size category rule.
8. **Return and remember grounded evidence.** The response boundary uses only
   current tool evidence for new results or mutations. It emits diagnostics,
   persists a bounded transcript through the memory service, and checkpoints
   the graph for the next turn.

## Where Data and State Live

| Store | What it owns | Current lifecycle |
| --- | --- | --- |
| Product JSONL + schema sidecar | Product records, taxonomy roles, filter roles, prices, and details | Operator-published catalog snapshot |
| Milvus | Vector candidates for the active catalog snapshot | Rebuilt or reused from the catalog fingerprint |
| Memory-service SQLite | Bounded conversation text, authoritative cart, product/cart-line identity, and atomic mutation replay | Service-owned persistence; shared only through that service |
| LangGraph `MemorySaver` | Exact graph messages and tool state keyed by `conversation_id` | Process-local; lost on restart and not shared across replicas |
| Product-ref cache | Recently returned product refs used by details and cart adds | Bounded and process-local; valid only for the active catalog snapshot |
| Turn `State` | Query, media, context, cart, current evidence, timings, and diagnostics | Transient for one request |
| Shopper skills | Reviewed behavioral instructions, response framing, roles, and tool grants | Versioned repository files; no customer or product state |

Memory is guidance, not truth. A remembered statement that an item was added or
is available cannot override the current cart or a current catalog result.
Same-process, same-conversation follow-ups generally work today. A restart,
another replica, or a cross-session request such as “show me the bag from last
week” is not yet a supported guarantee.

## Example: “What Bottoms Go With That Beige Top?”

Assume the previous turn returned a group of tops confirmed by the catalog's
`primary_color=beige` filter.

1. The new request and recent discussion establish an active styling thread and
   a beige-top candidate group as the direct anchor.
2. Skill activation selects `outfit-styling`; it does not select
   `product-discovery` merely because the shopper named one product role.
3. The model reads the advertised catalog taxonomy. In the current snapshot,
   `skirts` is the only published child that is a kind of bottom; pants,
   trousers, and shorts are not advertised. Dresses are excluded because a
   dress is not a bottom.
4. The model calls `search_catalog_tool` once with:

   - requested type: `bottoms`;
   - taxonomy relation: member of the requested umbrella;
   - taxonomy: `apparel/skirts`;
   - semantic direction: skirts that balance the beige top; and
   - hard constraints: none, unless the shopper also asked for beige or a
     same-color look.

5. Deterministic validation confirms that every selected taxonomy value is an
   advertised child of the requested role. The catalog then ranks only products
   inside that hard skirt scope.
6. The response presents grounded skirt candidates and explains their role in
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
invented-constraint assurance, and ambiguous historical references remain
separate work.

### 2. Add a structured conversation-reference ledger

Store a compact record for each turn containing candidate groups, product refs,
product roles, shared confirmed filters, shopper selections, and explicit
no-result or unsupported outcomes. Use it to give the model unambiguous
reference context before skill and tool selection.

The immediate correctness rule is simple: if the direct antecedent produced no
product, “that item” is unresolved. Ask a concise clarification instead of
jumping silently to an older candidate. This addresses the observed case where
“that skirt” followed a failed denim-skirt request but resolved to an earlier
maxi skirt.

Because the catalog is expected to remain stable, previously grounded product
evidence can be reused without a catalog recheck while its catalog fingerprint
still matches. A changed fingerprint requires a fresh search.

### 3. Select a production shared checkpointer

`MemorySaver` is appropriate for development but loses graph history on restart
and splits conversations across replicas. Choose an Apache-2.0/MIT-compatible
shared implementation or an internal checkpoint service, then add namespace,
retention, and fail-fast configuration tests at the existing checkpointer
factory boundary.

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

Typed persona support, live inventory, and additional policy content should be
added only after their trusted data contracts exist. They should not be inferred
from conversation text.

## Recommendation

The styling boundary is focused and cart idempotency is now memory-service
owned. Deliver cart-only intent authorization, durable turn/event storage, and
historical reference resolution as separate slices. The catalog remains
unchanged. The trap is adding keyword taxonomy mappings or increasingly
specific prompt rules: that would couple language interpretation to catalog
structure and make each new skill harder to extend safely.

For implementation detail, see the
[Shopper Agent Architecture](SHOPPER_AGENT_ARCHITECTURE.md),
[Skill Registry](SHOPPER_AGENT_SKILL_REGISTRY.md), and
[Tool Registry](SHOPPER_AGENT_TOOL_REGISTRY.md).
