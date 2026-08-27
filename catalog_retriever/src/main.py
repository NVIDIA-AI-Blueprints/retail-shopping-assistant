# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from typing import List, Dict, Any
import time
import os
import yaml
import logging
import sys

from shared.model_config import resolve_model_config, validate_model_config

try:
    from app.catalog import build_product_detail, load_catalog
    from app.index_catalog import index_on_boot
    from app.retriever import CatalogFilterError, Retriever, RetrieverConfig
except ModuleNotFoundError:
    from .catalog import build_product_detail, load_catalog
    from .index_catalog import index_on_boot
    from .retriever import CatalogFilterError, Retriever, RetrieverConfig

# Set up logging 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# FastAPI app
app = FastAPI()

# Get directory contents and report them.
dir_contents = []
for entry in os.listdir("."):
    dir_contents.append(entry)
logging.info(f"CATALOG RETRIEVER | startup | Directory contents: {dir_contents}")

# Get service behavior from config.yaml. Model endpoints come from models.yaml.
def load_config(base_config_path: str):
    """Load service configuration from YAML file."""

    if not os.path.exists(base_config_path):
        logging.error(f"Base config file not found at {base_config_path}")
        raise FileNotFoundError(f"Base config file not found at {base_config_path}")

    with open(base_config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def env_flag(name: str, default: bool = True) -> bool:
    """Read a boolean environment flag."""

    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false.")

shared_config_root = os.environ.get("SHARED_CONFIG_ROOT", "/app/shared/configs")
data = load_config(os.path.join(shared_config_root, "catalog_retriever", "config.yaml"))
data.update(
    {
        key: value
        for key, value in {
            "db_port": os.environ.get("CATALOG_DB_PORT"),
            "data_source": os.environ.get("CATALOG_DATA_SOURCE"),
            "schema_source": os.environ.get("CATALOG_SCHEMA_SOURCE"),
        }.items()
        if value
    }
)
model_config = resolve_model_config(config_root=shared_config_root)
validate_model_config(model_config, roles=("text_embedding",))
text_embedding = model_config.require("text_embedding")
image_embedding = model_config.get("image_embedding")
image_enabled = bool(
    image_embedding
    and not image_embedding.disabled
    and env_flag("CATALOG_IMAGE_EMBEDDING_ENABLED", default=True)
)
data.update(
    {
        "text_embed_port": text_embedding.base_url,
        "text_model_name": text_embedding.model,
        "text_api_key_env": text_embedding.api_key_env,
        "image_enabled": image_enabled,
        "image_embed_port": image_embedding.base_url if image_enabled else None,
        "image_model_name": image_embedding.model if image_enabled else None,
        "image_api_key_env": image_embedding.api_key_env if image_enabled else None,
    }
)
snapshot = load_catalog(
    data["data_source"],
    data["schema_source"],
    catalog_id=str(data.get("catalog_id") or "default"),
    image_enabled=image_enabled,
    text_model_name=text_embedding.model,
    image_model_name=image_embedding.model if image_enabled and image_embedding else None,
    shared_root=os.environ.get("SHARED_ROOT", "/app/shared"),
)
capabilities = snapshot.capabilities


# Setup Retriever once when app starts
config = RetrieverConfig(  
    text_embed_port=data["text_embed_port"],
    image_embed_port=data["image_embed_port"],
    text_model_name=data["text_model_name"],
    image_model_name=data["image_model_name"],
    text_api_key_env=data["text_api_key_env"],
    image_api_key_env=data["image_api_key_env"],
    image_enabled=data["image_enabled"],
    db_port=data["db_port"],
    db_name=data["db_name"],
    sim_threshold=data["sim_threshold"],
    text_collection=data["text_collection"],
    image_collection=data["image_collection"],
    filter_capabilities=capabilities.filters,
    catalog_size=snapshot.product_count,
    product_id_field=snapshot.schema.record.product_id,
    name_field=snapshot.schema.record.name,
    description_field=snapshot.schema.record.description,
    fallback_description_field=snapshot.schema.record.fallback_description,
    image_field=snapshot.schema.record.image,
    price_field=snapshot.schema.record.price,
    taxonomy_fields=snapshot.schema.taxonomy.fields,
    detail_fields=snapshot.schema.detail_fields,
)

logging.info("CATALOG RETRIEVER | startup | config.yaml ingested.")
logging.info("CATALOG RETRIEVER | startup | Initializing Retriever object.")
retriever = Retriever(config=config)

#: Whether a serving pod builds the index itself.
#:
#: True is right for one replica, which is what docker compose runs, and keeps
#: local work a single command. It is wrong for more than one, because building
#: starts by dropping the collection: two pods doing it together can leave one
#: filling a collection the other has just dropped, and the result is a partial
#: index that nothing notices, since the fingerprint is written per row.
#:
#: Set it false and run `python -m app.index_catalog` once as a Job before the
#: pods roll. They will then wait, not build -- see the readiness check below.
_INDEX_ON_BOOT = index_on_boot()

if _INDEX_ON_BOOT:
    logging.info(
        "CATALOG RETRIEVER | startup | Checking and populating Milvus if needed."
    )
    retriever.sync_snapshot(snapshot, verbose=True)
    logging.info("CATALOG RETRIEVER | startup | Milvus database ready.")
else:
    # Deliberately not a failure. The Job may still be running, and a pod that
    # exits here would crash-loop through a perfectly normal rollout; refusing
    # readiness is how it waits without pretending to serve.
    logging.info(
        "CATALOG RETRIEVER | startup | Not indexing (CATALOG_INDEX_ON_BOOT is "
        "false). Readiness waits for the index to match."
    )
    # The snapshot still has to be described to the retriever even when nothing
    # is built, because sync_snapshot is what tells it which fields it is
    # serving. Skipping it entirely would leave the retriever unconfigured.
    retriever.describe_snapshot(snapshot)


#: `matches_catalog` flushes the collection, which is too expensive to do on
#: every probe, and the answer only ever changes once: the index a pod is
#: waiting for either arrives or the pod is replaced when the catalog changes.
#: So the positive answer is cached and never asked again.
_index_ready = False


def index_is_ready() -> bool:
    """Whether Milvus holds the index for the catalog this pod loaded."""

    global _index_ready
    if _index_ready:
        return True
    try:
        _index_ready = retriever.matches_snapshot(snapshot)
    except Exception as exc:  # pragma: no cover - Milvus may still be starting
        logging.warning("CATALOG RETRIEVER | readiness | index check failed: %s", exc)
        return False
    return _index_ready

# Request bodies
class TextQueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    filters: Dict[str, Any] = Field(default_factory=dict)
    k: int = Field(default=4, ge=1, le=50)
    candidate_k: int | None = Field(default=None, ge=1)

class ImageQueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: List[str] = Field(default_factory=list)
    image_base64: str = ""
    categories: List[str] = Field(default_factory=list)
    filters: Dict[str, Any] = Field(default_factory=dict)
    k: int = Field(default=4, ge=1, le=50)
    candidate_k: int | None = Field(default=None, ge=1)

# Handles queries only containing text.
@app.post("/query/text")
async def query_text(req: TextQueryRequest):
    logging.info(f"CATALOG RETRIEVER | query_text() | Received POST: {req}.")
    try:
        result = await retriever.retrieve(
            query=req.text,
            categories=req.categories,
            filters=req.filters,
            k=req.k,
            candidate_k=req.candidate_k,
            image_bool=False,
            verbose=True
        )
    except CatalogFilterError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "texts": result.texts,
        "ids": result.ids,
        "similarities": result.similarities,
        "names": result.names,
        "images": result.images,
        "products": result.products,
        "diagnostics": result.diagnostics,
        "no_result_reason": result.no_result_reason,
    }

# Handles queries containing text and b64 images.
@app.post("/query/image")
async def query_image(req: ImageQueryRequest):
    logging.info("CATALOG RETRIEVER | query_image() | Received POST.")
    try:
        result = await retriever.retrieve(
            query=req.text,
            image=req.image_base64,
            categories=req.categories,
            filters=req.filters,
            k=req.k,
            candidate_k=req.candidate_k,
            image_bool=True,
            verbose=True
        )
    except CatalogFilterError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "texts": result.texts,
        "ids": result.ids,
        "similarities": result.similarities,
        "names": result.names,
        "images": result.images,
        "products": result.products,
        "diagnostics": result.diagnostics,
        "no_result_reason": result.no_result_reason,
    }

@app.get("/ready")
async def readiness_check():
    """Readiness: has this pod got a catalog to answer from?

    Indexing runs at import, before uvicorn binds the port, so a pod cannot
    reach the point of answering this without having finished. That makes the
    check cheap on purpose -- it reads the snapshot already in memory rather
    than asking Milvus, because a probe that runs every ten seconds must not
    flush a collection to answer.

    It deliberately does not check the embedding service or any other
    dependency. Readiness that fails on a dependency removes every pod at once,
    which turns a partial outage into a total one.
    """

    if not snapshot.product_count:
        raise HTTPException(status_code=503, detail="catalog snapshot is empty")
    if not index_is_ready():
        # The usual reason is that the indexing Job has not finished. Saying so
        # keeps the pod out of rotation instead of serving empty searches, which
        # is what it would otherwise do -- a search against a missing index
        # returns no products rather than an error.
        raise HTTPException(
            status_code=503,
            detail="catalog index is not built for this snapshot yet",
        )
    return {
        "status": "ready",
        "catalog_id": capabilities.catalog_id,
        "products": snapshot.product_count,
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0"
    }


@app.get("/capabilities")
async def get_capabilities():
    """Return catalog-owned search and filter capabilities."""
    return capabilities.model_dump()


@app.get("/products/{product_id}")
async def get_product(product_id: str):
    """Return deterministic details for one product, read from the index.

    The index stores every field the catalog declares, so this does not need a
    copy of the catalog in the process to answer. Shaped by the same
    `build_product_detail` as the loader, so the response cannot drift from what
    a snapshot read would have produced -- verified equal for every product.
    """

    record = retriever.product_record(product_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Product not found in active catalog")
    return build_product_detail(record, snapshot.schema).model_dump(mode="json")
