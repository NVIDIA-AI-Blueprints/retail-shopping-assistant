# Catalog Embedding Management

The catalog retriever owns text and image embedding creation. Clients send raw
text or raw image data; they never send embedding vectors. The service makes no
chat/completion call and performs no LLM interpretation, query expansion, or
learned reranking.

## Startup Synchronization

At startup the service loads and validates:

```text
shared/data/enriched_products.jsonl
shared/data/enriched_products.schema.yaml
```

It computes an internal fingerprint from:

- JSONL contents;
- sidecar contents;
- text embedding model;
- image embedding model and enabled state;
- referenced local image bytes when image search is enabled; and
- semantic-document template version.

Any code change that alters `build_search_document()` output must also bump
`SEARCH_DOCUMENT_TEMPLATE_VERSION` in `catalog_retriever/src/catalog.py`.
Otherwise an existing index could match unchanged data/model inputs even though
its embedded text uses the previous template.

Each indexed record carries that fingerprint. An index counts as current only
when its fingerprint and entity count match the active snapshot.

Two things follow, and they are done by different processes:

- **The serving container checks.** When the fingerprint does not match it
  answers `/ready` with 503 and serves nothing. It never rebuilds, and has no
  code path that could.
- **`python -m app.index_catalog` rebuilds**, dropping and refilling all enabled
  collections. A failed required text or image embedding aborts it, so a partial
  snapshot is never served.

The split exists because a rebuild starts by dropping the collection. One
process doing that is fine; two are destructive, and undetectably so -- the
fingerprint is written row by row, so a collection half-filled by one process
while another drops it carries the right fingerprint on every row it has, and
the check above passes on it.

Manual volume deletion is therefore unnecessary for ordinary catalog or model
changes. Running the indexer is.

## Text Embedding Source

One deterministic document is built per product:

```text
name: {name}
taxonomy: {ordered taxonomy path}
attributes:
- {sidecar field marked semantic}: {value}
summary: {primary description or configured fallback}
```

For the bundled catalog, the primary summary is `enriched_description` and
`description` is used only when enrichment is absent. Searchable attributes are
sorted, missing fields are omitted, enum underscores are humanized, and list
values are deduplicated and sorted.

IDs, prices, images, URLs, and source-row bookkeeping remain metadata and are
not embedded. `composition` and `care` are semantic/detail fields in the
current sidecar.

## Image Embeddings

When the `image_embedding` model role is enabled, the catalog service reads the
image field declared by the sidecar, loads each local/remote image, and creates
one image embedding per product. The text and image collections store the same
source product ID and filter metadata so all search modes enforce identical
hard filters. A missing local image fails startup, and changing its bytes
changes the internal fingerprint even when its JSONL path stays the same.

Text-only is the default. Set `CATALOG_IMAGE_EMBEDDING_ENABLED=true` to index
images as well; until then, image query and hybrid modes are absent from
`/capabilities`.

## Replace the Catalog

1. Put the new JSONL and role sidecar under `shared/data/`.
2. Point the catalog config at both files:

   ```yaml
   data_source: "/app/shared/data/my_products.jsonl"
   schema_source: "/app/shared/data/my_products.schema.yaml"
   ```

3. For local process mode, set both host paths if overriding config:

   ```bash
   export CATALOG_DATA_SOURCE="$PWD/shared/data/my_products.jsonl"
   export CATALOG_SCHEMA_SOURCE="$PWD/shared/data/my_products.schema.yaml"
   ```

4. Rebuild and bring the catalog service back up. **A serving container never
   indexes itself**, but Compose runs the `catalog-indexer` service for you and
   `catalog-retriever` waits for it to finish, so this is one command:

   ```bash
   docker compose up -d --build catalog-retriever
   ```

   To index without cycling the service, run it directly instead:

   ```bash
   docker compose exec catalog-retriever python -m app.index_catalog
   ```

   Indexing is a separate, deliberate step because rebuilding starts by
   dropping the collection. That is safe when exactly one process does it and
   destructive when two do: the second can drop what the first is still
   filling, and nothing downstream notices, because the fingerprint is written
   row by row, so a half-filled collection carries the right fingerprint on
   every row it has. A serving pod has no code path that indexes, so it cannot
   be the second one.

   The command is safe to repeat. It checks the fingerprint first and does
   nothing when the index is already current, so it can sit unconditionally in
   a deployment pipeline.

   Until it has run against a changed catalog, the service answers `/health`
   with 200 and `/ready` with 503: alive, deliberately serving no traffic.

5. Wait for catalog readiness and verify the active schema and values.
   Use `/ready`, not `/health`: health only says the process is alive, and a
   service with an unbuilt index is alive and unable to answer.

   ```bash
   curl -s http://localhost:8010/ready
   curl -s http://localhost:8010/health
   curl -s http://localhost:8010/capabilities
   ```

6. Restart the chain server so it drops its process-lifetime cached capability
   contract:

   ```bash
   docker compose restart chain-server
   ```

7. Verify the chain-cached contract before serving traffic:

   ```bash
   curl -s http://localhost:8009/capabilities
   ```

Changing only rows or observed values does not require application-code or
sidecar changes, but it still requires the catalog-health-then-chain restart
sequence above. Add a sidecar entry only when a new field's meaning must be
declared. See [Catalog Schema and Filters](CATALOG_FILTERS.md).

## Query Verification

At query time, the serving agent sends one entry in the `text` list. It receives
one text embedding and vector search, followed by deterministic candidate
fusion, product-ID deduplication, hard filtering, thresholding, and similarity
ordering. Milvus COSINE scores are normalized from `[-1, 1]` to `[0, 1]`
relevance scores before the configured similarity threshold is applied.
Direct/internal clients retain the list shape for compatibility, and supplied
entries are embedded concurrently. `/query/image` retains pooled image/text
similarity ordering.

Text:

```bash
curl -sS -X POST http://localhost:8010/query/text \
  -H 'Content-Type: application/json' \
  -d '{
    "text": ["formal geometric earrings"],
    "filters": {"price": {"lte": 200}},
    "k": 4
  }'
```

Image plus text:

```bash
curl -sS -X POST http://localhost:8010/query/image \
  -H 'Content-Type: application/json' \
  -d '{
    "text": ["similar style"],
    "image_base64": "data:image/jpeg;base64,...",
    "filters": {"price": {"lte": 200}},
    "k": 4
  }'
```

The default candidate window covers the complete active catalog before hard
filtering, thresholding, and final deterministic trimming to `k`.

## Recovery

Manual collection or volume removal is reserved for database corruption or
operational recovery. If needed, stop the catalog service, drop
`shopping_advisor_text_db` and `shopping_advisor_image_db`, then restart. The
service rebuilds them from the validated snapshot.
