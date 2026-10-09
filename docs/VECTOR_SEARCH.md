# Vector Search at Catalog Scale

How the catalog's Milvus index is chosen, what changes when it runs on a GPU,
and what a large catalog needs that the defaults do not give it.

The defaults suit the shipped catalog: a CPU Milvus image, `AUTOINDEX` with
COSINE, and one streaming insert per collection. Nothing here has to change
until the catalog grows into hundreds of thousands of products or search
latency matters under load.

All Milvus access goes through `pymilvus`, which the catalog service already
uses.

## Contents

- [The vector store stack](#the-vector-store-stack)
- [Choosing an index](#choosing-an-index)
- [Configuration](#configuration)
- [Running Milvus on a GPU](#running-milvus-on-a-gpu)
- [Sizing](#sizing)
- [Search limits on a GPU index](#search-limits-on-a-gpu-index)
- [Indexing a large catalog](#indexing-a-large-catalog)
- [Not done yet](#not-done-yet)

## The vector store stack

| Service | Image | Role |
|---------|-------|------|
| `etcd` | `quay.io/coreos/etcd:v3.6.5` | Milvus metadata |
| `seaweedfs` | `chrislusf/seaweedfs:3.73` | S3-compatible object storage for Milvus segments and index files |
| `milvus` | `milvusdb/milvus:v2.6.5` (CPU) or `v2.6.5-gpu` | Vector database, standalone |

Milvus talks to SeaweedFS through its S3 gateway on port 9010. The `MINIO_*`
variables on the `milvus` service are Milvus's own names for its S3 endpoint
and credentials. SeaweedFS reads its S3
credentials from `seaweedfs-s3.json` at the repository root, and the `milvus`
service must use the same pair.

Object storage holds what Milvus persists: sealed segments, binlogs, and built
index files. It does not hold product images; the catalog service reads those
from local disk.

`MILVUS_VERSION` selects the Milvus image tag, so the GPU image is
`MILVUS_VERSION=v2.6.5-gpu`.

## Choosing an index

| Mode | Index | Image | Where it builds | Where it searches | Use when |
|------|-------|-------|-----------------|-------------------|----------|
| CPU only (default) | `AUTOINDEX`, COSINE | CPU or GPU | CPU | CPU | The catalog fits comfortably in memory and latency is acceptable |
| GPU build, CPU search | `GPU_CAGRA`, IP, `adapt_for_cpu=true` | GPU | GPU | CPU | Rebuilds are slow on CPU, but search traffic does not justify keeping the GPU busy |
| GPU build, GPU search | `GPU_CAGRA`, IP, `adapt_for_cpu=false` | GPU | GPU | GPU | Large catalog under concurrent search load |

`GPU_CAGRA` is built with these parameters:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `intermediate_graph_degree` | 128 | Neighbors per node before pruning |
| `graph_degree` | 100 | Neighbors per node in the final graph |
| `build_algo` | `NN_DESCENT` | Graph construction algorithm |
| `cache_dataset_on_device` | `true` | Keeps raw vectors on the GPU for refinement |
| `adapt_for_cpu` | opposite of `gpu_search` | Builds a graph the CPU can search |

When the CPU searches a GPU-built graph, every search must pass `ef`, and
Milvus rejects an `ef` smaller than the search limit. The catalog sends
`ef = max(configured ef, limit)`, with a configured default of 100. When the
GPU searches, the catalog sends `itopk_size`, the smallest power of two from 64
upwards that is at least the limit.

### Why the GPU index uses IP, not COSINE

Milvus GPU indexes accept `L2` and `IP` but not `COSINE`. With `GPU_CAGRA` the
catalog normalizes every vector to unit length, both when indexing and when
searching, and uses `IP`. The inner product of two unit vectors is their cosine,
so scores keep the same `[-1, 1]` range and the existing mapping to `[0, 1]`
and the relevance threshold apply unchanged. A zero-length embedding cannot be
normalized and is rejected.

## Configuration

`shared/configs/catalog_retriever/config.yaml`:

```yaml
vector_index:
  type: "AUTOINDEX"     # or "GPU_CAGRA"
  gpu_search: false     # GPU_CAGRA only: search on the GPU instead of the CPU
  ef: 100               # GPU_CAGRA CPU search only; raised to the limit when smaller
```

| Variable | Overrides | Values |
|----------|-----------|--------|
| `CATALOG_VECTOR_INDEX_TYPE` | `vector_index.type` | `AUTOINDEX`, `GPU_CAGRA` |
| `CATALOG_GPU_SEARCH` | `vector_index.gpu_search` | `true`, `false` |

Set them for both `catalog-indexer` and `catalog-retriever`; Compose passes
them to both. The index type and search placement are part of the catalog
fingerprint, so changing either one makes the next indexer run rebuild both
collections instead of serving an index built the other way.

## Running Milvus on a GPU

`GPU_CAGRA` fails on the CPU image. A GPU host needs the NVIDIA driver and the
NVIDIA Container Toolkit, and the `milvus` service needs the GPU image and a
device reservation. An override file along these lines does it. It has not been
run against this repository yet, so treat it as a starting point:

```yaml
# docker-compose.gpu-milvus.yaml
services:
  milvus:
    image: milvusdb/milvus:v2.6.5-gpu
    environment:
      KNOWHERE_GPU_MEM_POOL_SIZE: "2048;4096"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["0"]
              capabilities: ["gpu"]
```

```bash
export CATALOG_VECTOR_INDEX_TYPE=GPU_CAGRA
export CATALOG_GPU_SEARCH=true   # or false for GPU build, CPU search
docker compose -f docker-compose.yaml -f docker-compose.gpu-milvus.yaml up -d
```

`KNOWHERE_GPU_MEM_POOL_SIZE` is the initial and maximum size, in MB, of the
GPU memory pool Milvus uses for temporary search buffers. It does not include
the index itself; size the GPU for both.

If the same GPU also serves self-hosted models, give Milvus a different
`device_ids` entry, or account for both in the model server's memory fraction.

## Sizing

Raw vectors take `products × dimension × 4` bytes per collection. A
`GPU_CAGRA` index takes roughly 1.8 times that in GPU memory, because the graph
is stored alongside the cached vectors. With image embeddings enabled there are
two collections, and both are indexed the same way.

| Products | Dimension | Raw vectors | `GPU_CAGRA` (≈1.8×) |
|----------|-----------|-------------|---------------------|
| 100,000 | 1,024 | 0.4 GB | 0.7 GB |
| 1,000,000 | 1,024 | 4.1 GB | 7.4 GB |
| 1,000,000 | 2,048 | 8.2 GB | 14.7 GB |
| 10,000,000 | 1,024 | 41 GB | 74 GB |

Add the search memory pool, and headroom for a rebuild: a reindex drops the
collection and builds a new one, so the old and new indexes are not resident
together, but the build itself needs working memory above the final size.

## Search limits on a GPU index

`GPU_CAGRA` refuses a search whose limit exceeds 1024, and `itopk_size` must be
at least the limit. The catalog's candidate window defaults to the size of the
whole catalog, so that hard filters applied after the search never run out of
candidates. Under `GPU_CAGRA` the window is capped at 1024.

The consequence: with a large catalog and a narrow hard filter, the 1024
nearest products may contain fewer than `k` that pass the filter, and the
search returns fewer results than asked. `AUTOINDEX` has no such cap. Pushing
filters into the Milvus search expression removes the problem; see
[Not done yet](#not-done-yet).

## Indexing a large catalog

Today the indexer embeds the catalog in batches of 32, then writes each
collection with a single `insert` followed by `flush`. That works for the
shipped catalog. For a large one, a single insert can exceed the gRPC message
size limit, so the path forward is to split the records into fixed-size batches
of `insert` calls, then `flush` once and wait for the index build to finish
before reporting the index as current. All of that uses `pymilvus` only.

Milvus also offers bulk import: write Parquet files to object storage and call
`do_bulk_insert`. It is the fastest route for millions of rows, and SeaweedFS
can serve as its staging bucket. It is not adopted here: the Parquet writer is
the `pymilvus[bulk_writer]` extra, which pulls in `minio`, `pyarrow`, and
`azure-storage-blob`, and this project adds no libraries for it.

## Not done yet

- **Numeric filters in Milvus.** Enum filters are pushed into the Milvus
  search; numeric fields such as price are stored as text, so they are still
  applied in Python after the vector search. Pushing them down too is what
  makes the 1024 cap harmless.
- **Batched inserts and waiting for the index.** As described above, the
  indexer inserts each collection in one call and does not wait for the index
  build.
- **GPU verification.** The `GPU_CAGRA` path is covered by unit tests only. It
  has not run on a GPU host with the `-gpu` image.
