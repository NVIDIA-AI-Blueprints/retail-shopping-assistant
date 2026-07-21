# Schema-Driven Catalog Architecture

**Status:** Implemented.

The catalog retriever loads
`shared/data/enriched_products.jsonl` together with
`shared/data/enriched_products.schema.yaml`. The JSONL supplies products and
values; the sidecar supplies field meaning. Catalog ingestion, capabilities,
filter validation, execution, and product lookup are deterministic. No LLM runs
inside the catalog service for interpretation, query expansion, or reranking.
Its only model inference is configured text/image embedding generation. The
shopper-facing LLM uses the catalog's advertised capabilities to express
intent; deterministic code decides whether that intent can be enforced.

## At a Glance

| Stage | Input | Output | Owner |
| --- | --- | --- | --- |
| Ingest | JSONL rows + field-role sidecar | One validated `CatalogSnapshot` | Catalog service |
| Index | Snapshot semantic documents and images | Fingerprinted Milvus collections | Catalog service |
| Advertise | The same snapshot | `GET /capabilities`: fields, values, ranges, coverage, and taxonomy scopes | Catalog service |
| Discover | First successful capability response, cached for the chain-server process lifetime | Full validation contract plus a compact prompt projection | Chain server |
| Interpret | Shopper language + advertised contract | One required `semantic_query` + product noun/umbrella `requested_product_type` provenance + required `taxonomy` envelope + required `required_constraints` | Shopper LLM |
| Enforce | Structured intent + the same capabilities | Taxonomy roles mapped to advertised fields plus validated hard filters, or a refusal | Deterministic chain code |
| Retrieve | Singleton agent text query/image + validated filters | Product refs; optional deterministic details lookup | Catalog service |

The agent depends on advertised capabilities, but capabilities are not trusted
as executable instructions. The chain validates the agent's structured intent,
and the catalog validates the resulting request again.

The agent tool schema requires `semantic_query`, `requested_product_type`,
`taxonomy`, and `required_constraints`. For every text search,
`requested_product_type` is the shortest product noun or true umbrella from the
shopper's current turn or direct antecedent, excluding color, material, fit,
occasion, weather, and style modifiers. For `agent_selected_type`, it is the
chosen advertised role noun. It is provenance rather than taxonomy or ranking
text and is `null` only for image-only search. The taxonomy envelope has stable
`category` and `subcategory` roles, but their enum values are generated from the
cached capability contract. `required_constraints` stays generic for
non-taxonomy must-haves. The chain keeps the full capability object for
deterministic mapping and validation, while the system prompt receives a compact
projection of other filter, semantic, and detail roles. Generated values are
never baked into application code.

Each search accepts at most one category. When a broad request names no concrete
type, `agent_selected_type` selects exactly one advertised subcategory as the
focused starting role. It is forbidden for any role whose
product type the shopper named, including an alternative, confirmation,
comparison, or follow-up. `no_direct_catalog_match` uses empty
taxonomy and no hard constraints and performs no retrieval. It is decided from
the requested type alone: an unsupported modifier does not erase an advertised
type, while subjective style remains semantic direction.

Explicit image or hybrid intent is also validated against advertised retrieval
modes and requires an attached image. It stops before retrieval rather than
silently becoming a text search.

For example, the generated projection for this catalog includes concise lines
like these rather than the full capability response:

```text
Retrieval modes: text, image, hybrid
Hard filters (enum values are exact; numbers use min/max):
- neckline: enum; values boat, collared, crew, ..., v_neck; semantic yes
- price: number; range 39.9 to 269.99; semantic no
Semantic/detail fields (not hard filters):
- care: text; detail yes
Taxonomy-specific field availability (category > subcategory; use exact values from Hard filters above):
- category=apparel
  - subcategory=dresses
    filters: closure, garment_length, neckline, pattern, price, primary_color, silhouette, sleeve_length
```

The field names, values, ranges, and scopes in this projection are generated
from the capability object; the example is not application configuration.

The first successful capability fetch is cached for the chain-server process
lifetime. An unsuccessful initial fetch is not cached, so a later request can
retry. After a successful fetch there is no per-turn refresh. The catalog
service still validates every request against its active snapshot and fails
closed rather than accepting stale or unsupported filters.

Example shopper request:

> Show me beige skirts under $100, preferably cotton.

If the current capabilities advertise `subcategory`, `primary_color`, and
`price` as filters while `composition` is semantic-only, the agent emits:

```json
{
  "semantic_query": "cotton skirt",
  "requested_product_type": "skirts",
  "taxonomy_status": "exact_requested_type",
  "taxonomy": {
    "category": ["apparel"],
    "subcategory": ["skirts"]
  },
  "required_constraints": {
    "primary_color": ["beige"],
    "price": {"max": 100}
  }
}
```

“Preferably cotton” remains soft semantic ranking. If the shopper instead says
“must be cotton,” the agent also places `composition` in
`required_constraints`; deterministic validation then refuses the search
because the active catalog cannot enforce free-text composition as a hard
filter. Nothing silently weakens the shopper's must-have.

There are no fashion category or filter values in the deterministic catalog or
request-building code. New categories and values come from JSONL rows. Field
names and roles are declared once in the sidecar; adding an entirely new field
requires declaring its type and uses, not adding category branches. The current
wire contract supports one or two ordered taxonomy levels; a deeper hierarchy
would require a contract change, not category-specific rules.

Implementation map:

- ingest and semantic documents: `catalog_retriever/src/catalog.py`;
- generated capabilities: `catalog_retriever/src/capabilities.py`;
- capability/detail API and request models: `catalog_retriever/src/main.py`;
- embedding retrieval, filtering, and deterministic ranking:
  `catalog_retriever/src/retriever.py`;
- lifecycle caching and compact prompt rendering: `chain_server/src/catalog_capabilities.py`;
- intent validation: `chain_server/src/catalog_request.py`;
- plan-to-request mapping: `chain_server/src/catalog_execution.py`; and
- agent tool wiring: `chain_server/src/deepagents_runtime.py`.

## What Changed From the Old Catalog

| Concern | Before | Now |
| --- | --- | --- |
| Source | `products_extended.csv` | `enriched_products.jsonl` plus `enriched_products.schema.yaml` |
| Bundled rows | 218 | 205 |
| Product identity | Implicit row/name behavior | Required unique, URL-safe `record_id` |
| Field meaning | CSV columns plus configured filter registry | Sidecar-declared record, taxonomy, filter, semantic, and detail roles |
| Values and category scope | Limited/global discovery | Observed from rows globally and within category/subcategory scopes |
| Embedding text | Primarily product prose | Name + taxonomy + semantic attributes + enriched description, with legacy description fallback |
| Agent contract | Static/general search assumptions | Catalog-lifecycle fields, values, ranges, scopes, and retrieval modes |
| Product facts | Search-result text | Search refs followed by deterministic `GET /products/{product_id}` details |

This migration adds no new product names relative to the legacy CSV. All 205
current names already existed there. Twelve unique legacy names are absent from
the current feed, and one duplicate legacy row is collapsed. The important
functional change is therefore not “more SKUs”; it is that normalized taxonomy
and attributes make existing products newly searchable, hard-filterable, or
available as grounded details.

That can legitimately change expected answers. For example, a product that was
present but had no explicit color can now satisfy an exact color filter. Whenever
bundled row membership or searchable attributes change, catalog-dependent
Goldens and fixtures must be checked against the new snapshot. Conversational
behavior expectations should remain stable; obsolete inventory-absence claims
should not be preserved merely to keep an old score.

## Design

At service startup, `load_catalog()` builds one validated `CatalogSnapshot`:

```text
JSONL + role sidecar
        |
        v
 CatalogSnapshot
   |      |       |
   v      v       v
indexes  capabilities  product details
```

That same snapshot backs text/image indexing, `GET /capabilities`, generic
hard-filter execution, and `GET /products/{product_id}`. No runtime path
reopens and reinterprets the source independently.

The loader:

- reads JSONL one record per line and reports malformed input with its line;
- requires unique source product IDs, nonempty names and taxonomy values,
  finite non-negative prices, descriptions, and images when image search is
  enabled;
- preserves every source field;
- builds deterministic semantic documents;
- discovers enum/list values, numeric ranges, taxonomy nodes, and coverage;
- advertises unknown fields as `unclassified`; and
- fails startup rather than serving a partially valid catalog.

## Sidecar Contract

The sidecar maps core roles and declares field type and uses:

```yaml
record:
  product_id: record_id
  name: name
  description: enriched_description
  fallback_description: description
  image: image
  price: price

taxonomy:
  fields: [category, subcategory]

fields:
  category:
    type: enum
    uses: [filter, semantic, detail]
  subcategory:
    type: enum
    uses: [filter, semantic, detail]
  neckline:
    type: enum
    uses: [filter, semantic, detail]
  care:
    type: text
    uses: [semantic, detail]
```

The sidecar never contains catalog values, value aliases, or category-specific
applicability rules. A new value or category therefore needs only new JSONL
rows. An entirely new field needs one sidecar role declaration before it can
be searched or filtered.

Mapped or declared source field names must not use the index-owned names `pk`,
`text`, `vector`, or `catalog_fingerprint`; ingestion rejects those mappings
before building an index so product IDs and metadata remain round-trippable.

Supported field types are `enum`, `enum_list`, `number`, and `text`. Supported
uses are:

- `filter`: enforceable hard constraint;
- `semantic`: included in text embedding documents; and
- `detail`: returned by deterministic product lookup.

Hard filters may use `enum`, `enum_list`, or `number`. Text fields are semantic
or detail evidence; declaring a text hard filter fails schema validation rather
than introducing an undefined matching rule.

Every ordered taxonomy field must be a scalar `enum` with `filter` use because
the agent's generic taxonomy envelope is enforced as a hard browse scope. The
sidecar still supplies the field names, and ingestion still derives every
category and subcategory value from the rows.

For the current catalog, `composition` and `care` are semantic/detail text,
not exact filters. All colors, patterns, taxonomy values, and category-specific
facets are observed from the JSONL.

## Semantic Documents

Each product produces this stable passage:

```text
name: {name}
taxonomy: {category} > {subcategory}
attributes:
- {searchable attribute, sorted}: {value}
summary: {enriched_description or description fallback}
```

Missing attributes are omitted. Underscores in enum values are humanized for
embedding text. List values are deduplicated and sorted. Product ID, price,
image, URL, and source-row bookkeeping never enter the text embedding.

Image pixels are embedded into a separate image collection. Public query
requests accept raw text and raw image data, never client-generated vectors.

## Capabilities and Querying

`GET /capabilities` returns:

- product count and supported retrieval modes;
- a compatibility `filters` projection;
- authoritative field roles, coverage, observed enum/list values, and numeric
  ranges; and
- nested category/subcategory nodes with scoped filters and semantic fields.

This communicates facts such as “dresses have observed neckline values” or
“boots have shaft height” without `if category == ...` logic. A new category
automatically gets a new node from the same grouping pass.

Filter mechanics are generic:

- AND between different fields;
- OR between requested values within one enum or list field;
- any-overlap matching for `enum_list`; and
- lower/upper bounds for numbers.

For this 205-product catalog, the default vector candidate window covers the
whole active snapshot before hard filters and final `top_k` trimming. Unknown
filters, values, taxonomy values, and operators return explicit validation
errors.

The chain server fetches capabilities on first use and caches the first
successful response for its process lifetime. Every agent search supplies
exactly one semantic query, required pre-retrieval product-agnostic
`shopper_guidance`, required product noun/umbrella
`requested_product_type` provenance, a required taxonomy envelope, and required
non-taxonomy constraints.
Each call accepts at most one category. Executable text search requires at least
one advertised category or subcategory; `requested_product_type` is `null` and
both taxonomy arrays are empty for image-only search, while the arrays are also
empty for the
`no_direct_catalog_match` no-retrieval result. Generic taxonomy roles map
through `taxonomy.category_field` and `taxonomy.subcategory_field`;
subcategory-only selections infer their owning category and fail if multiple
owners would cross the one-category boundary. When both roles are present,
every selected subcategory must belong to the selected category; incompatibility
fails before retrieval.

When the current shopper turn contains one unambiguous literal pair of exact
advertised subcategories from the same category, the chain requires the
model-authored requested type and taxonomy to retain both branches under
`member_of_requested_umbrella`. This exact capability check does not interpret
nonliteral alternatives. The pair remains one catalog execution: the plan uses
a pair-wide candidate window, then rank-preserving result selection keeps one
returned candidate per branch when available and trims to the configured result
count.

The runtime executes a normalized taxonomy-plus-hard-constraint scope at most
once per turn, regardless of semantic paraphrasing. Genuinely different hard-
filter scopes may execute within the configured cap. The agent's one query is
independent of taxonomy and is sent to the catalog as a singleton `text` list;
`shopper_guidance` stays in the chain tool-result boundary. Unsupported direct
must-haves stop the search rather than being silently weakened; an unsupported
modifier does not erase an advertised type, and subjective style stays semantic.

One invalid search may consume that distinct scope's single search-only repair.
Malformed or nonempty free-form `unadvertised_requirements` arguments on a
native schema-invalid call fail closed. Constraint review is model-owned only
for a schema-valid, genuinely open `agent_selected_type` role: explicit
objective must-haves remain and fail closed, while only inferred or subjective
requirements may be removed; deterministic code does not parse shopper prose.
A successful partial search may continue to another valid role with its own
repair opportunity, but no scope receives two repairs. The configured turn cap
remains three successful searches. Each successful search-tool result records
the model-authored semantic query as internal `SEARCH_DIRECTION_EVIDENCE` and
the pre-retrieval `shopper_guidance` authored under the active skill. For a
completed successful search-only turn, the active skill gets one tools-disabled
synthesis and the draft passes through the grounding editor. That guidance and
static skill `response_guidance` support deterministic fallback, which lists all
returned candidates, adds a neutral
continuation for partial successful evidence, and groups each pre-retrieval
guidance sentence and confirmed-filter set with products from its originating
search. Zero-result output retains
its exact advertised taxonomy and filters and supports no broader absence claim.

The chain's operator diagnostics carry bounded per-product search/detail
evidence, its truncation flag, and bounded `catalog_scope_outcomes` for
`no_direct_catalog_match` and `zero_results`. The Judge receives only those
three diagnostic fields alongside generated conversation history; semantic
queries, raw tool messages, reasoning, and other diagnostics are discarded.

The catalog HTTP contract retains its text-list shape for compatibility with
direct/internal clients. Multiple entries, when supplied directly, are embedded
concurrently and combined deterministically before product-ID deduplication,
shared filtering, thresholding, and final similarity ordering. The
serving agent does not use that shape for query expansion. The catalog does not
interpret “or,” invent queries, choose constraints, call a chat model, or run a
learned reranker.

The structure guarantees that supplied scope values are current and enforced;
language-to-scope selection remains an agent judgment.

## Product Details and Refresh

Search results return the source `record_id` as `product_id`; Milvus primary
keys and product names are never identities. `GET /products/{product_id}` reads
the active snapshot and returns fields marked `detail`, including exact care
instructions when provided. The agent requires a product ref from search,
performs this lookup, and preserves the returned fields through its grounding
editor.

Current generated IDs are safe only within the active snapshot because the
feed does not guarantee cross-catalog stability. Detail reads and cart adds
verify both request-evidence ID and display name against the active catalog;
stale or reused refs require a fresh search. Transient catalog failures leave
the cart unchanged and are not described as product removal.

The chain server authorizes refs only from current-request evidence: either a
search in that request or one unique exact resolution from durable product-card
events in the same conversation. That resolver survives chain-server restart
and makes no catalog call. It records catalog revision metadata when supplied
but does not yet enforce revision freshness, so catalog replacement can still
require a fresh search before details or cart adds. This does not change the
catalog service or its stateless request boundary.

An internal fingerprint covers the JSONL, sidecar, embedding model names,
image-search state, local image bytes when image search is enabled, and
semantic-template version. On restart, matching complete indexes are reused. A
mismatch rebuilds all enabled fixed collections. The fingerprint is internal
rebuild metadata, not a public catalog-version protocol.

To replace the catalog:

1. Replace the JSONL and, only when field meaning changes, its sidecar.
2. Restart the catalog retriever.
3. Wait for index synchronization and a healthy service, then inspect the live
   catalog contract at `http://localhost:8010/capabilities`.
4. Restart the chain server so it discards its process-lifetime cached
   contract.
5. Verify the chain-cached contract at
   `http://localhost:8009/capabilities` before serving traffic.

Hot reload, an ingestion API, an ontology service, LLM schema inference,
versioned collection aliases, inventory, and variants remain out of scope.
