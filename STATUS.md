# Project Status

Updated: 2026-07-16

## Current Milestone

The schema-driven catalog refactor is implemented:

- the active feed is a 205-product JSONL catalog with a field-role sidecar;
- all 205 current product names existed in the prior 218-row CSV; the migration
  removes 12 unique legacy names, collapses one duplicate row, and primarily
  improves how existing products can be discovered and verified;
- startup builds one validated snapshot for indexing, capabilities, filtering,
  product details, and index-rebuild fingerprinting;
- catalog values and category-specific field availability are derived from the
  active rows rather than hard-coded in application logic;
- the catalog publishes fields, values, ranges, coverage, taxonomy scopes, and
  retrieval modes through `GET /capabilities`;
- the chain server caches the first successful capability response for its
  process lifetime, reuses the full object for deterministic validation, and
  sends the LLM a compact projection rather than refetching the API each turn;
  and
- soft preferences remain semantic, while every must-have is validated as a
  required constraint and unsupported strict requirements stop the search.

The concise source of truth is
[Schema-Driven Catalog Architecture](docs/CATALOG_REFACTOR_PLAN.md).

## Verification

- Offline unit suite: 664 passed.
- Current capability API response: 147,792 bytes; compact LLM projection:
  5,769 bytes, down from 12,962 bytes before this change while retaining
  taxonomy keys, every current enum value, and semantic/filter roles.
- UI production build: compiled successfully.
- UI lint: 0 errors and 3 pre-existing warnings.
- Docker Compose configuration: valid.
- Fixed shopping evaluation: 48/48 turns completed with no request or Judge
  errors.
- The post-compaction WIP Judge average is 3.2292/5 versus 3.2083/5 on the
  cached Staging baseline.
- The WIP score-4-or-better rate is 22/48 versus 23/48 on Staging.
- Mean, median, p95, and maximum latency all improved: 8.475s / 7.141s /
  25.167s / 29.367s versus Staging's 14.922s / 8.863s / 57.367s / 94.609s.
- Four internal memory-persistence calls timed out, although all shopper
  responses and Judge scores completed; this remains a reliability caveat.

The comparison remains qualified because the catalog treatment changed. Two
low Judge scores remain confirmed stale-reference cases: existing beige skirts
became discoverable through explicit color metadata, and a pre-existing pastel
top contradicts an old “none available” Golden.

## Remaining Quality Risk

Twelve turns still scored 2/5. The recurring risks are completeness for broad
budget requests and context-sensitive matching across follow-up turns. The
prior 117.286-second light-summer failure is resolved in this run: it returned
products in 10.261 seconds and scored 3/5. The four memory-persistence timeouts
also need follow-up outside this catalog-refactor scope.

The catalog architecture and its qualified quality evidence are ready to
preserve in a feature commit. The raw Judge score should not be described as an
unqualified catalog-quality gain until catalog-dependent Goldens are reconciled
with the active inventory.
