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
| Interpret | Shopper language + advertised contract | One required `semantic_query` + required `taxonomy` envelope + required `required_constraints` | Shopper LLM |
| Enforce | Structured intent + the same capabilities | Taxonomy roles mapped to advertised fields plus validated hard filters, or a refusal | Deterministic chain code |
| Retrieve | Singleton agent text query/image + validated filters | Product refs; optional deterministic details lookup | Catalog service |

The agent depends on advertised capabilities, but capabilities are not trusted
as executable instructions. The chain validates the agent's structured intent,
and the catalog validates the resulting request again.

The agent tool schema requires `semantic_query`, `taxonomy`, and
`required_constraints`. The taxonomy envelope has stable `category` and
`subcategory` roles, but their enum values are generated from the cached
capability contract. `required_constraints` stays generic for non-taxonomy
must-haves. The chain keeps the full capability object for deterministic mapping
and validation, while the system prompt receives a compact projection of other
filter, semantic, and detail roles. Generated values are never baked into
application code.

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
exactly one semantic query, a required taxonomy envelope, and required
non-taxonomy constraints. Text search requires at least one advertised category
or subcategory; both arrays may be empty only for image-only search. Generic
taxonomy roles map through `taxonomy.category_field` and
`taxonomy.subcategory_field`;
subcategory-only selections infer all owning categories. When both roles are
present, every selected subcategory must belong to a selected category and
every selected category must own at least one selected subcategory; partial or
complete incompatibilities fail before retrieval.

The runtime executes a normalized taxonomy scope at most once per turn,
regardless of semantic paraphrasing. Different scopes may execute within the
configured cap. The agent's one query is sent to the catalog as a singleton
`text` list. Unsupported or invalid must-haves stop the search rather than being
silently weakened.

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
verify both cached ID and display name against the active catalog; stale or
reused refs require a fresh search. Transient catalog failures leave the cart
unchanged and are not described as product removal.

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
