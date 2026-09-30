# Catalog Architecture

The catalog describes itself, and everything downstream is derived from that
description. Two files are the source: `shared/data/enriched_products.jsonl`
holds the products and their values, and
`shared/data/enriched_products.schema.yaml` declares what each field *means*.

That split is the whole design. Because field meaning is declared rather than
coded, there are no fashion categories, color lists, or filter names anywhere in
the deterministic catalog code. Point the service at your own JSONL and the
search contract follows — a new category or value needs new rows and nothing
else; a genuinely new field needs one line in the sidecar.

No language model runs inside the catalog service. It does no query expansion,
no interpretation, and no learned reranking; its only inference is generating
embeddings. Retrieval is therefore reproducible, and the interesting judgment —
turning shopper language into a scope — stays in the agent where it can be
observed. The agent uses the advertised contract to express intent, and
deterministic code decides whether that intent can be enforced.

## At a glance

There are two separate timelines here, and keeping them apart is most of
understanding the design. The catalog builds and publishes itself **once, at
startup**. The agent consumes what was published **on every search**.

### At startup

The catalog service does all of this alone, before serving anything:

```text
JSONL + sidecar  ──▶  CatalogSnapshot  ──┬──▶  Milvus collections
                                         └──▶  GET /capabilities
```

| Step | In | Out |
| --- | --- | --- |
| Ingest | JSONL rows and the role sidecar | One validated `CatalogSnapshot` |
| Index | The snapshot's documents and images | Fingerprinted Milvus collections |
| Advertise | The same snapshot | `GET /capabilities` — fields, values, ranges, coverage, taxonomy scopes |

That single snapshot backs indexing, capabilities, filtering, and detail
lookup, so nothing later reopens and reinterprets the source files on its own.
Startup fails rather than serve a partially valid catalog.

### On each search

This crosses three owners, which is why it is worth naming who does what:

| Step | Owner | What happens |
| --- | --- | --- |
| Discover | Chain server | Fetches `/capabilities` on first use and caches it for the process lifetime |
| Interpret | Shopper LLM | Turns shopper language into a structured request, using only advertised fields and values |
| Enforce | Chain server | Validates that request against the cached contract, then maps it or refuses |
| Retrieve | Catalog service | Runs the embedding search with the validated hard filters and returns product refs |

Capabilities are advertised, but never trusted as executable instructions. The
chain server validates the model's intent against them, and the catalog service
validates the arriving request again. Both checks earn their place: the chain's
copy is cached and can be stale, while the catalog is the one actually holding
the snapshot.

## The sidecar

The sidecar maps the core roles and declares each field's type and uses:

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
  neckline:
    type: enum
    uses: [filter, semantic, detail]
  care:
    type: text
    uses: [semantic, detail]
```

Types are `enum`, `enum_list`, `number`, and `text`. Uses are `filter` (an
enforceable hard constraint), `semantic` (included in the embedded document),
and `detail` (returned by product lookup).

A few rules exist for reasons worth knowing:

- **Text fields cannot be hard filters.** Declaring one fails validation rather
  than inventing an undefined matching rule. In this catalog `composition` and
  `care` are semantic and detail evidence, which is why "must be cotton" is
  refused rather than approximated.
- **Taxonomy fields must be scalar `enum` with `filter` use,** because the
  agent's taxonomy envelope is enforced as a hard browse scope.
- **The sidecar holds no values,** no aliases, and no category-specific rules.
  Values are observed from the rows, globally and within each category scope.
- **`pk`, `text`, `vector`, and `catalog_fingerprint` are reserved** index
  names and are rejected as field mappings, so product IDs and metadata stay
  round-trippable.

## What the catalog advertises

`GET /capabilities` returns the product count, retrieval modes, field roles and
coverage, observed enum values and numeric ranges, and nested
category/subcategory nodes with their scoped filters.

This is what lets the system state "dresses have observed neckline values" or
"boots have shaft height" without a single `if category == ...` branch. A new
category picks up its own node from the same grouping pass.

The chain server renders a compact projection of that response into the prompt
rather than the whole object:

```text
Retrieval modes: text, image, hybrid
Hard filters (enum values are exact; numbers use min/max):
- neckline: enum; values boat, collared, crew, ..., v_neck; semantic yes
- price: number; range 39.9 to 269.99; semantic no
Taxonomy-specific field availability (category > subcategory):
- category=apparel
  - subcategory=dresses
    filters: closure, garment_length, neckline, pattern, price, primary_color
```

Every name, value, and range there is generated. None of it is application
configuration.

The first successful fetch is cached for the chain-server process lifetime; a
failed first fetch is not cached, so a later request retries. There is no
per-turn refresh, which is safe because the catalog revalidates every request
against its live snapshot and fails closed on anything stale.

## How intent becomes a search

> Show me beige skirts under $100, preferably cotton.

With `subcategory`, `primary_color`, and `price` advertised as filters and
`composition` semantic-only, the agent emits:

```json
{
  "semantic_query": "cotton skirt",
  "requested_product_type": "skirts",
  "taxonomy": {"category": ["apparel"], "subcategory": ["skirts"]},
  "required_constraints": {"primary_color": ["beige"], "price": {"max": 100}},
  "shopper_guidance": "These skirts keep the requested beige palette and budget while cotton remains a preference.",
  "scope_complete": true
}
```

"Preferably cotton" stays soft ranking. If the shopper says "must be cotton,"
the agent puts `composition` in `required_constraints` — and validation
**refuses the search**, because the catalog cannot enforce free-text
composition as a hard filter. This is the behavior to preserve when extending
the catalog: a must-have is never silently downgraded to a preference.

Each search carries at most one category. A shopper-named type that is not
separately advertised may be searched under one faithful parent category, with
results presented as closest alternatives. If neither a direct type nor one
faithful parent fits, the assistant asks a clarifying question instead of
substituting or claiming the catalog has nothing. Image or hybrid intent is
validated against the advertised modes and requires an attached image; it stops
rather than quietly degrading into a text search.

Unknown fields, values, and operators return explicit validation errors. Search
scope, repair attempts, and per-turn budgets are runtime concerns — see
[Assistant Architecture](ASSISTANT_ARCHITECTURE.md).

## Retrieval

Each product becomes one stable passage:

```text
name: {name}
taxonomy: {category} > {subcategory}
attributes:
- {searchable attribute, sorted}: {value}
summary: {enriched_description or description fallback}
```

Attribute order is sorted and missing values are omitted, so the same row
always produces the same text. Product ID, price, image, URL, and row
bookkeeping never enter the embedding — they are exact-match or display
concerns, and embedding them only adds noise. Image pixels go to a separate
collection. Public requests accept text and image data, never client-supplied
vectors.

Filtering is generic: AND across different fields, OR across values within one
field, any-overlap for `enum_list`, and bounds for numbers. At the bundled
catalog's size the default candidate window covers the whole active snapshot
before hard filters and `top_k` trimming.

## Identity and details

Search returns the source `record_id` as `product_id`. Milvus primary keys and
display names are never identities. `GET /products/{product_id}` reads the
active snapshot and returns the fields marked `detail`.

Generated IDs are only valid within the active snapshot, because the feed makes
no cross-catalog stability guarantee. Detail reads and cart adds verify both
the ID and the display name against the live catalog, so a stale ref forces a
fresh search rather than resolving to the wrong product. A transient catalog
failure leaves the cart untouched and is never reported as the product being
removed.

## Replacing the catalog

An internal fingerprint covers the JSONL, the sidecar's indexing-relevant
parts, the embedding model names, image-search state, and the semantic-template
version. Matching indexes are reused on restart; a mismatch rebuilds. Changing
the embedding model — including switching between hosted endpoints and locally
served models — changes the fingerprint and forces a rebuild, which is what
keeps vectors and model in sync. Prose-only sidecar edits deliberately do not,
since re-embedding the catalog because someone improved a description would
discourage describing anything.

1. Replace the JSONL, and the sidecar only if field meaning changed.
2. Restart the catalog retriever.
3. Wait for indexing and health, then inspect `http://localhost:8010/capabilities`.
4. Restart the chain server so it drops its cached contract.
5. Confirm the chain-side contract at `http://localhost:8009/capabilities`.

Hot reload, an ingestion API, LLM schema inference, versioned collection
aliases, inventory, and variants are out of scope.

## Where it lives

Ingest and semantic documents are in
[`catalog.py`](../catalog_retriever/src/catalog.py), generated capabilities in
[`capabilities.py`](../catalog_retriever/src/capabilities.py), the API and
request models in [`main.py`](../catalog_retriever/src/main.py), and retrieval,
filtering, and ranking in
[`retriever.py`](../catalog_retriever/src/retriever.py). On the chain-server
side, caching and prompt rendering are in
[`catalog_capabilities.py`](../chain_server/src/catalog_capabilities.py),
validation in [`catalog_request.py`](../chain_server/src/catalog_request.py),
and tool wiring in [`tools/catalog.py`](../chain_server/src/tools/catalog.py).
