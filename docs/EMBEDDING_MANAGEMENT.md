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

Each indexed record carries that fingerprint. The service reuses an index only
when its fingerprint and entity count match the active snapshot. Otherwise it
drops and rebuilds all enabled catalog collections. A failed required text or
image embedding aborts startup so a partial snapshot is not served.

Manual volume deletion is therefore unnecessary for ordinary catalog or model
changes.

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

Set `CATALOG_IMAGE_EMBEDDING_ENABLED=false` to run text-only. Image query and
hybrid modes then disappear from `/capabilities`.

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

4. Restart the catalog service and watch indexing:

   ```bash
   docker compose up -d --build catalog-retriever
   docker compose logs -f catalog-retriever
   ```

5. Wait for catalog health and verify the active schema and values:

   ```bash
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
ordering. Direct/internal
clients retain the list shape for compatibility, and supplied entries are
embedded concurrently. `/query/image` retains pooled image/text similarity
ordering.

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
