# Catalog Architecture

The catalog describes itself, and everything downstream is derived from that
description. Two files are the source: `shared/data/enriched_products.jsonl`
holds the products and their values, and
`shared/data/enriched_products.schema.yaml` — the *sidecar* — declares what
each field *means*.

That split is the whole design. Because field meaning is declared rather than
coded, there are no fashion categories, color lists, or filter names anywhere in
the deterministic catalog code. Point the service at your own JSONL and the
search contract follows — a new category or value needs new rows and nothing
else; a genuinely new field needs one line in the sidecar. Never hardcode
catalog values in Python, prompts, UI code, or YAML.

No language model runs inside the catalog service. It does no query expansion,
no interpretation, and no learned reranking; its only inference is generating
embeddings. Retrieval is therefore reproducible, and the interesting judgment —
turning shopper language into a scope — stays in the agent where it can be
observed.

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

The sidecar maps the core record roles, names the ordered taxonomy, and gives
every other field a type and zero or more uses:

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
  primary_color:
    type: enum
    uses: [filter, semantic, detail]
  price:
    type: number
    uses: [filter, detail]
  care:
    type: text
    uses: [semantic, detail]
```

| Type | Meaning | Hard-filter behavior |
| --- | --- | --- |
| `enum` | One canonical value | Exact `in` membership |
| `enum_list` | Zero or more canonical values | Any requested value overlaps |
| `number` | Numeric value | `gte` / `lte` bounds |
| `text` | Free text | Semantic and detail only; `filter` is rejected at validation |

| Use | Effect |
| --- | --- |
| `filter` | Advertised and enforced as a hard constraint |
| `semantic` | Included in the product's text embedding document |
| `detail` | Returned by `GET /products/{product_id}` |

A few rules exist for reasons worth knowing:

- **Text fields cannot be hard filters.** Declaring one fails validation rather
  than inventing an undefined matching rule. In this catalog `composition` and
  `care` are semantic and detail evidence, which is why "must be cotton" is
  refused rather than approximated. If a future feed needs strict care
  filtering, add a structured field such as `care_method` and declare its role;
  do not parse prose in catalog code.
- **Taxonomy fields must be scalar `enum` with `filter` use,** because the
  agent's taxonomy envelope is enforced as a hard browse scope. That is a
  structural contract, not a fixed taxonomy — the field names and every value
  still come from the sidecar and the active rows.
- **The sidecar holds no values,** no aliases, and no statement that a field
  applies to a particular category. Values, ranges, coverage, and category
  scope are all observed from the rows.
- **`pk`, `text`, `vector`, and `catalog_fingerprint` are reserved** index
  names and are rejected as field mappings, so product IDs and metadata stay
  round-trippable.
- **Unknown source fields are preserved but exposed as `unclassified`.** They
  cannot silently become search text or hard filters.

## What the catalog advertises

`GET /capabilities` on port `8010` returns the catalog's live snapshot: the
product count and retrieval modes, `fields` as the authoritative field-role
contract with coverage, observed enum values and numeric ranges, nested
`taxonomy.categories.<category>.subcategories.<subcategory>` nodes with their
scoped filters, and `filters` as a flat projection for existing clients.

This is what lets the system state "dresses have observed neckline values" or
"boots have shaft height" without a single `if category == ...` branch. A new
category picks up its own node from the same grouping pass.

Port `8009` returns the chain server's cached copy. The chain renders a compact
projection of it into the prompt rather than the whole object, dropping counts
and coverage while keeping the real taxonomy field names, every observed value,
and whether a filter is also semantically searchable:

```text
Retrieval modes: text
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
`composition` semantic-only, the agent emits this scope (one entry in the
tool's `scopes` list):

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
substituting or claiming the catalog has nothing.

Search scope, repair attempts, and per-turn budgets are runtime concerns — see
[Assistant Architecture](ASSISTANT_ARCHITECTURE.md).

## Retrieval

Each product becomes one stable passage:

```text
name: {name}
taxonomy: {category} > {subcategory}
attributes:
- {semantic attribute, sorted}: {value}
summary: {enriched_description or description fallback}
```

Attribute order is sorted, missing values are omitted, enum underscores are
humanized, and list values are deduplicated, so the same row always produces
the same text. Product ID, price, image, URL, and row bookkeeping never enter
the embedding — they are exact-match or display concerns, and embedding them
only adds noise.

Clients send raw text and never send vectors. Candidates are
fused, deduplicated by product ID, hard-filtered, thresholded, and
similarity-sorted; Milvus COSINE scores are normalized from `[-1, 1]` to
`[0, 1]` before the configured threshold applies. The default candidate window
covers the whole active snapshot before hard filters and the final trim to `k`.

The index is `AUTOINDEX` with COSINE by default. An opt-in `GPU_CAGRA` index
uses IP on unit-length vectors, which gives the same scores, and caps the
candidate window at 1024; [Vector Search at Catalog Scale](VECTOR_SEARCH.md)
covers when to use it and how to size it.

Filtering is generic: AND across different fields, OR across values within one
field, any-overlap for `enum_list`, and bounds for numbers.

### Querying the service directly

The agent-facing tool takes a list of `scopes`, each with a `semantic_query`, a
capability-derived `taxonomy` envelope, and `required_constraints`; the chain maps that envelope
onto the real taxonomy field names and merges it with validated must-haves into
`filters`. The internal API below is that mapped shape. Semantic meaning goes
in `text`, exact requirements in `filters`. The serving agent sends a singleton
`text` list; direct clients keep the list shape for compatibility and their
entries are embedded concurrently.

```bash
curl -sS -X POST http://localhost:8010/query/text \
  -H 'Content-Type: application/json' \
  -d '{
    "text": ["flowing structured formal dress"],
    "filters": {"subcategory": ["dresses"], "price": {"lte": 200}},
    "k": 4
  }'
```

Unsupported fields, values, taxonomy values, and operators return HTTP 422
instead of being ignored — the whole point is that a constraint is never
quietly weakened. Numeric bounds must be finite, so booleans, `NaN`,
infinities, and a range containing any invalid bound are also 422. `min` and
`gte` are aliases for the lower bound, `max` and `lte` for the upper; supply
one spelling per bound, and do not pair a nested bound with its top-level
compatibility alias (`price.min` plus `min_price`). When both taxonomy roles
are given, each selected category must own at least one selected subcategory
and vice versa, or the search stops before retrieval.

## Identity and details

Search returns the source `record_id` as `product_id`. Milvus primary keys and
display names are never identities. `GET /products/{product_id}` reads the
active snapshot and returns the fields marked `detail` — which is how an exact
question about free-text evidence like care instructions is answered: search
first, then look the product up.

Generated IDs are only valid within the active snapshot, because the feed makes
no cross-catalog stability guarantee. Detail reads and cart adds verify both
the ID and the display name against the live catalog, so a stale ref forces a
fresh search rather than resolving to the wrong product. A transient catalog
failure leaves the cart untouched and is never reported as the product being
removed.

## Replace the catalog

An internal fingerprint covers the JSONL, the sidecar's indexing-relevant
parts, the embedding model name, and the semantic-document template version. Each
indexed row carries it, and an index counts as current only when both the
fingerprint and the row count match the active snapshot.

Changing the embedding model — including switching between hosted endpoints and
locally served models — changes the fingerprint, which is what keeps vectors
and model in sync. Prose-only sidecar edits deliberately do not, since
re-embedding the catalog because someone improved a description would
discourage describing anything. Any code change that alters
`build_search_document()` output must bump `SEARCH_DOCUMENT_TEMPLATE_VERSION`
in `catalog_retriever/src/catalog.py`; otherwise an index could match unchanged
data and models while holding text built from the previous template.

**Indexing and serving are separate processes, deliberately.** A rebuild starts
by dropping the collection, so two doing it at once is destructive and
undetectably so: the fingerprint is written row by row, so a collection
half-filled by one process while another drops it carries the right fingerprint
on every row it has. Only `python -m app.index_catalog` rebuilds. A serving
container never indexes and has no code path that could — when the fingerprint
does not match it answers `/health` with 200 and `/ready` with 503: alive, and
deliberately serving nothing.

1. Put the new JSONL and sidecar under `shared/data/`, and point the catalog
   config at both. Every JSONL line must be an object with a unique, nonempty
   product ID that fits one URL path segment; ingestion rejects IDs whose
   whitespace would be normalized, IDs containing slashes, and dot-only
   segments.

   ```yaml
   data_source: "/app/shared/data/my_products.jsonl"
   schema_source: "/app/shared/data/my_products.schema.yaml"
   ```

   For local process mode, override with `CATALOG_DATA_SOURCE` and
   `CATALOG_SCHEMA_SOURCE` instead.

2. Update the sidecar only if a field's meaning changed. New rows, categories,
   values, or optional fields left empty on some products need no sidecar edit;
   a genuinely new field needs its type and uses declared, and dropping
   `filter` from a non-taxonomy field's uses is how you stop advertising it.

3. Run the indexer. Compose does this for you and `catalog-retriever` waits for
   it, so the ordinary path is one command:

   ```bash
   docker compose up -d --build catalog-retriever
   ```

   To reindex without cycling the service, run it directly:

   ```bash
   docker compose exec catalog-retriever python -m app.index_catalog
   ```

   Either form is safe to repeat: it checks the fingerprint first and exits
   without doing anything when the index is already current, so it can sit
   unconditionally in a deployment pipeline. A failed required text or image
   embedding aborts it, so a partial snapshot is never served. Manual Milvus
   volume deletion is not part of a normal refresh.

4. Wait for readiness, then verify the active contract. Use `/ready`, not
   `/health`.

   ```bash
   curl -s http://localhost:8010/ready
   curl -s http://localhost:8010/capabilities
   ```

5. Restart the chain server so it drops its process-lifetime cached contract,
   and confirm the chain-side copy before serving traffic.

   ```bash
   docker compose restart chain-server
   curl -s http://localhost:8009/capabilities
   ```

Because newly structured attributes can make existing products discoverable in
new ways, changing rows or field roles also means reviewing catalog-dependent
integration Goldens — keep behavioral expectations stable, but reconcile frozen
inventory-absence claims with the new snapshot.

Manual collection or volume removal is reserved for database corruption. If you
do drop `shopping_advisor_text_db`, restarting will not bring it back: the service will load the snapshot, find no matching
index, and answer `/ready` with 503 indefinitely. Run the indexer to refill
it.

Hot reload, an ingestion API, LLM schema inference, versioned collection
aliases, inventory, and variants are out of scope.

## Where it lives

Ingest, semantic documents, and the fingerprint are in
[`catalog.py`](../catalog_retriever/src/catalog.py), generated capabilities in
[`capabilities.py`](../catalog_retriever/src/capabilities.py), the API and
request models in [`main.py`](../catalog_retriever/src/main.py), indexing in
[`index_catalog.py`](../catalog_retriever/src/index_catalog.py), and retrieval,
filtering, and ranking in
[`retriever.py`](../catalog_retriever/src/retriever.py). On the chain-server
side, caching and prompt rendering are in
[`catalog_capabilities.py`](../chain_server/src/catalog_capabilities.py),
validation in [`catalog_request.py`](../chain_server/src/catalog_request.py),
and tool wiring in [`tools/catalog.py`](../chain_server/src/tools/catalog.py).
