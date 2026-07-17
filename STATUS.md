# Project Status

Updated: 2026-07-17

## Current Milestone

The schema-driven catalog refactor is implemented:

- the active feed is a 205-product JSONL catalog with a field-role sidecar;
- all 205 current product names existed in the prior 218-row CSV; the migration
  removes 12 unique legacy names, collapses one duplicate row, and primarily
  improves how existing products can be discovered and verified;
- startup builds one validated snapshot for indexing, capabilities, filtering,
  product details, and index-rebuild fingerprinting;
- raw Milvus COSINE scores are normalized to `[0, 1]` relevance scores before
  applying the configured similarity threshold;
- catalog values and category-specific field availability are derived from the
  active rows rather than hard-coded in application logic;
- the catalog publishes fields, values, ranges, coverage, taxonomy scopes, and
  retrieval modes through `GET /capabilities`;
- the chain server caches the first successful capability response for its
  process lifetime, reuses the full object for deterministic validation, and
  sends the LLM a compact projection rather than refetching the API each turn;
  and
- every agent catalog call requires one `semantic_query`, a `taxonomy` envelope,
  and `required_constraints`. The allowed `taxonomy.category` and
  `taxonomy.subcategory` values are generated from the cached catalog
  capabilities rather than application taxonomy. Text search requires at least
  one of those values; only image-only search may leave both arrays empty;
- the optional search-mode enum is generated from the catalog's advertised
  retrieval modes, and an explicit unknown or unsupported mode stops before
  retrieval instead of becoming an automatic default;
- the chain maps those generic taxonomy roles to the catalog-advertised field
  names, infers owning categories for a valid subcategory-only selection,
  rejects incompatible category/subcategory selections, and applies the result
  as deterministic hard filters; and
- one normalized taxonomy scope may execute only once per shopper turn. A
  repeated same-scope call is stopped even when its semantic wording changes;
  distinct scopes remain bounded by `max_catalog_searches_per_turn`.

The catalog service performs embeddings, filtering, and deterministic retrieval
only. It makes no chat/completion call or shopper-language interpretation.

The concise source of truth is
[Schema-Driven Catalog Architecture](docs/CATALOG_REFACTOR_PLAN.md).

## Verification

- Offline unit suite: 670 passed.
- Current capability API response: 147,792 bytes; compact LLM projection:
  5,769 bytes, down from 12,962 bytes before this change while retaining
  taxonomy keys, every current enum value, and semantic/filter roles.
- UI production build: compiled successfully.
- UI lint: 0 errors and 3 pre-existing warnings.
- Docker Compose configuration: valid.
- The latest live shopping evaluation and timing comparison are preserved in
  the canonical local quality archive rather than versioned as source.
- The latest 48-turn run improved Judge average from 3.0000 to 3.1667 and the
  score-4-or-better count from 20 to 22 versus the removed multi-query WIP. All
  48 turns completed and were timed. Mean / median / p95 were 15.638s / 12.439s
  / 33.997s; one provider 429 plus its 60-second retry produced the 70.836s
  maximum. There were no request or Judge errors. One Deep Agents recursion
  failure returned grounded fallback products, and five memory operations
  approached the 10-second timeout.
- “Show me stylish clutches.” improved from 2/5 in 26.883s to 4/5 in 6.812s.
  The catalog received one text query with `category=bags` and
  `subcategory=clutches`. “Any closed shoes or boots?” also scored 4/5 and
  returned both requested branches.
- The run is promoted to the canonical local `current_wip` archive. Against
  Staging, Judge average is 3.1667 versus 3.2083 with one fewer score>=4; mean
  and median are slower, while p95 and maximum are more than 23 seconds lower.

The comparison remains qualified because the catalog treatment changed. Two
low Judge scores remain confirmed stale-reference cases: existing beige skirts
became discoverable through explicit color metadata, and a pre-existing pastel
top contradicts an old “none available” Golden.

## Remaining Quality Risk

The corrected contract guarantees validation and enforcement of the scope and
must-haves the agent supplies. It cannot prove that a language model copied
every strict phrase from shopper language into `required_constraints`. For
example, if the agent omitted “only cotton,” deterministic validation would
have no omitted value to reject. Guaranteeing that separately would require
another interpretation/review model call or fixed natural-language rules; both
were intentionally excluded to avoid added latency and hard-coded language
logic. The prompt and required tool fields are the chosen minimal boundary.

After the live run, offline-only fail-closed checks were added for duplicate
taxonomy values, partially incompatible category/subcategory sets, and taxonomy
fields that are not scalar enum hard filters. These generic checks add no model
call and do not alter the valid requests observed in the 48-turn run.

The single grounded recursion fallback, five slow memory operations, and the
provider-retry tail remain reliability observations rather than hidden reruns.

The raw Judge score should not be described as an unqualified catalog-quality
gain until catalog-dependent Goldens are reconciled with the active inventory.
