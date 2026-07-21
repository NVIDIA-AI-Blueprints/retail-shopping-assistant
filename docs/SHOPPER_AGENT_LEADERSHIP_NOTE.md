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
raw turn storage now provides the audit/replay foundation; the structured
historical-reference resolver remains a separate Slice 5.

## Current Request Flow

```mermaid
flowchart LR
    A[Shopper request] --> B[Turn state and identity]
    B --> C[Start durable turn; load recent turns and cart]
    C --> D[Activate skills and bind grants]
    D --> E[Model proposes typed tool call]
    E --> F[Authorize and validate]
    F --> G[Catalog, cart, or policy owner]
    G --> H[Grounded response and durable finalize]
```

1. **Scope and start the turn.** The API creates one request identity containing
   session, conversation, cart, and request identifiers. Before guardrails,
   model, or tool work, the memory service transaction creates the ordered
   durable turn and returns an opaque attempt token, or exactly replays a
   matching finalized request.
2. **Load continuity.** A new turn start returns bounded finalized raw
   shopper/assistant turns and the authoritative cart from SQLite. Those turns
   replace the legacy rolling context blob. LangGraph separately loads the
   conversation's exact graph/tool messages from process-local `MemorySaver`
   using `conversation_id` as the thread ID; this key remains unchanged in
   Slice 4.
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
8. **Finalize and return grounded evidence.** The response boundary uses only
   current tool evidence for new results or mutations. It finalizes the durable
   turn with the current attempt token as completed, blocked, or failed with
   replay output and ordered event envelopes, then emits the response and
   diagnostics. Only the latest abandoned turn can reopen, and its token rotates;
   a stale finalizer receives a safe response with no stale products. A generic
   finalize outage keeps the grounded response and conversation checkpoint and
   adds an operator diagnostic.

## Where Data and State Live

| Store | What it owns | Current lifecycle |
| --- | --- | --- |
| Product JSONL + schema sidecar | Product records, taxonomy roles, filter roles, prices, and details | Operator-published catalog snapshot |
| Milvus | Vector candidates for the active catalog snapshot | Rebuilt or reused from the catalog fingerprint |
| Memory-service SQLite | Ordered raw shopper/assistant turns, exact finalized replay, bounded recent-turn reads, authoritative cart, product/cart-line identity, and atomic mutation replay | Named-volume persistence for one memory-service replica; retention is operator-owned |
| LangGraph `MemorySaver` | Exact graph messages and tool state keyed by `conversation_id` | Process-local; lost on restart and not shared across replicas |
| Product-ref cache | Recently returned product refs used by details and cart adds | Bounded and process-local; valid only for the active catalog snapshot |
| Turn `State` | Query, media, context, cart, current evidence, timings, and diagnostics | Transient for one request |
| Shopper skills | Reviewed behavioral instructions, response framing, roles, and tool grants | Versioned repository files; no customer or product state |

Memory is guidance, not truth. A remembered statement that an item was added or
is available cannot override the current cart or a current catalog result.
Raw turns now survive a chain-server restart through the memory service, but
exact graph/tool state and product refs do not. Same-process,
same-conversation follow-ups generally work today. Another replica or a
cross-session request such as “show me the bag from last week” is not yet a
supported guarantee.

The event vocabulary and projection columns created with the durable schema do
not yet interpret preferences, anchors, selections, or product references.
They are reserved for Slice 5 and are not active model context.

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

### 2. Build the Slice 5 structured historical resolver

Slice 4 stores ordered raw turns, replay output, and event envelopes. Slice 5
must interpret grounded candidate groups, product refs, product roles, shared
confirmed filters, shopper selections, and explicit no-result or unsupported
outcomes into a bounded projection. Only then should request-scoped checkpoints
replace the current conversation-scoped MemorySaver.

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

The styling boundary is focused, cart idempotency is memory-service owned, and
durable raw turn start/finalize/replay is built for one SQLite replica. Deliver
cart-only intent authorization, Slice 5 historical reference resolution, and a
shared production graph/memory design as separate work. The catalog remains
unchanged. The trap is treating raw transcript persistence as resolved product
reference semantics or adding keyword taxonomy mappings and increasingly
specific prompt rules; both would make each new skill harder to extend safely.

For implementation detail, see the
[Shopper Agent Architecture](SHOPPER_AGENT_ARCHITECTURE.md),
[Skill Registry](SHOPPER_AGENT_SKILL_REGISTRY.md), and
[Tool Registry](SHOPPER_AGENT_TOOL_REGISTRY.md).
