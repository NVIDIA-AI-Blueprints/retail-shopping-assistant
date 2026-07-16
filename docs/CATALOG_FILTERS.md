# Catalog Schema and Filters

Catalog behavior is defined by two files:

```text
shared/data/enriched_products.jsonl
shared/data/enriched_products.schema.yaml
```

The JSONL contains products and values. The sidecar describes what each field
means. Do not hardcode category names, colors, tags, or other catalog values in
Python, prompts, UI code, or YAML configuration.

## Sidecar Roles

Core record mappings identify the product ID, display name, descriptions,
image, and price. Ordered taxonomy fields establish the hierarchy. Each other
field has a type and zero or more uses:

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
  primary_color:
    type: enum
    uses: [filter, semantic, detail]
  composition:
    type: text
    uses: [semantic, detail]
  care:
    type: text
    uses: [semantic, detail]
```

Types:

| Type | Meaning | Hard-filter behavior |
| --- | --- | --- |
| `enum` | One canonical value | Exact `in` membership |
| `enum_list` | Zero or more canonical values | Any requested value overlaps |
| `number` | Numeric value | `gte` / `lte` bounds |
| `text` | Free text | Semantic/detail only; `filter` is rejected during schema validation |

Uses:

| Use | Effect |
| --- | --- |
| `filter` | Advertised and enforced as a hard filter |
| `semantic` | Included in the product's text embedding document |
| `detail` | Returned by `GET /products/{product_id}` |

The sidecar declares field meaning only. It must never enumerate values or say
that a field applies to a particular category. Ingestion derives values,
ranges, coverage, and category scope from the active rows.

## Dynamic Capabilities

After ingestion, inspect:

```bash
curl -s http://localhost:8010/capabilities
curl -s http://localhost:8009/capabilities
```

Port `8010` returns the catalog service's live snapshot. Port `8009` returns
the chain server's process-lifetime cached copy. The chain fetches the first
successful full contract once, uses it for deterministic validation, and gives
the LLM only a compact projection without counts, coverage, or repeated scoped
values. The projection retains the actual taxonomy field names, every observed
enum/list value for this small-catalog design, and whether a hard filter is
also semantically searchable.

The catalog response contains:

- `product_count` and retrieval modes;
- `fields`, the authoritative field-role contract;
- `taxonomy.categories.<category>.subcategories.<subcategory>` with scoped
  filters and semantic fields; and
- `filters`, a flat compatibility projection used by existing clients.

Enum/list values and numeric ranges are always observed from the JSONL. The
nested scopes tell the agent which fields and values are present for a product
type. For example, current dress rows advertise `neckline`, while current boot
rows advertise `shaft_height`; neither relationship exists in application
code.

## Changing the Catalog

| Change | Required work |
| --- | --- |
| New product | Replace/update JSONL, then use the coordinated restart below |
| New category, subcategory, enum, or tag value | Update JSONL, then use the coordinated restart below |
| Optional field missing on some products | No schema change; update JSONL, then use the coordinated restart below |
| Entirely new field | Add its type/uses to the sidecar, then use the coordinated restart below |
| Field should no longer be filterable | Remove `filter` from its sidecar uses, then use the coordinated restart below |

Unknown source fields are preserved but exposed as `unclassified`. They cannot
silently become search text or hard filters.

Because newly structured attributes can make existing products discoverable in
new ways, changing rows or field roles also requires reviewing catalog-dependent
integration Goldens. Keep behavioral expectations stable, but reconcile frozen
inventory-absence claims with the active catalog snapshot.

## Direct Catalog-Service Query Example

This is the internal catalog API shape. The agent-facing tool instead uses
`semantic_query` plus `required_constraints`; the chain validates that intent
and converts supported must-haves into this `filters` payload.

Text requests keep semantic meaning in `text` and exact requirements in
`filters`:

```json
{
  "text": ["flowing formal dress"],
  "filters": {
    "subcategory": ["dresses"],
    "neckline": ["v_neck"],
    "price": {"lte": 200}
  },
  "k": 4
}
```

Different fields use AND. Multiple enum/list values within one field use OR.
Unsupported fields, values, taxonomy values, and operators return HTTP 422
instead of being ignored. Numeric bounds must be finite numbers; booleans,
`NaN`, infinities, and a range containing any invalid bound return HTTP 422
rather than weakening the requested filter. `min` and `gte` are aliases for
the lower bound, while `max` and `lte` are aliases for the upper bound. Supply
only one spelling for each bound; combining both aliases for the same bound is
ambiguous and returns HTTP 422. Do not combine a nested bound with its top-level
compatibility alias either: `price.min` plus `min_price`, or `price.max` plus
`max_price`, returns HTTP 422. Legacy `categories` can be combined with
non-taxonomy filters, but not with an explicit taxonomy filter in `filters`.

Care instructions remain semantic text because whole care paragraphs are not
safe enum values. For an exact care question, first search for the product and
then call `GET /products/{product_id}`. If a future feed needs strict care
filtering, add a structured field such as `care_method` and declare its role in
the sidecar; do not parse prose in catalog code.

## Rebuild Workflow

1. Validate that every JSONL line is an object with a unique product ID. IDs
   must be nonempty canonical strings that fit one URL path segment; ingestion
   rejects IDs whose whitespace would be normalized, slash-containing IDs, and
   dot-only path segments.
2. Update the sidecar only when field meaning changes.
3. Restart the catalog retriever.
4. Watch startup logs until all enabled indexes are synchronized.
5. Verify `http://localhost:8010/capabilities` and a targeted text/image query.
6. Restart the chain server so it drops its cached capability contract.
7. Verify `http://localhost:8009/capabilities` before serving traffic.

The service calculates an internal fingerprint from data, sidecar, embedding
models, image-search state, referenced local image bytes, and the semantic
template. Matching indexes are reused; changed inputs trigger a full rebuild.
Manual Milvus volume deletion is not required for normal catalog refreshes.
