# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import time
import os
import yaml
import logging
import sys

from shared.model_config import resolve_model_config, validate_model_config

try:
    from app.retriever import Retriever, RetrieverConfig
except ModuleNotFoundError:
    from .retriever import Retriever, RetrieverConfig

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
    image_collection=data["image_collection"]
)

logging.info("CATALOG RETRIEVER | startup | config.yaml ingested.")
logging.info("CATALOG RETRIEVER | startup | Initializing Retriever object.")
retriever = Retriever(config=config)
logging.info("CATALOG RETRIEVER | startup | Checking and populating Milvus database if needed.")
retriever.milvus_from_csv(csv_path=data["data_source"], verbose=True)
logging.info("CATALOG RETRIEVER | startup | Milvus database ready.")

# Request bodies
class TextQueryRequest(BaseModel):
    text: List[str] = []
    categories: List[str] = []
    filters: Dict[str, Any] = Field(default_factory=dict)
    k: int = 4

class ImageQueryRequest(BaseModel):
    text: List[str] = []
    image_base64: str = ""
    categories: List[str] = []
    filters: Dict[str, Any] = Field(default_factory=dict)
    k: int = 4

# Handles queries only containing text.
@app.post("/query/text")
async def query_text(req: TextQueryRequest):
    logging.info(f"CATALOG RETRIEVER | query_text() | Received POST: {req}.")
    texts, ids, sims, names, images = await retriever.retrieve(
        query=req.text,
        categories=req.categories,
        filters=req.filters,
        k=req.k,
        image_bool=False,
        verbose=True
    )
    return {
        "texts": texts,
        "ids": ids,
        "similarities": sims,
        "names": names,
        "images": images
    }

# Handles queries containing text and b64 images.
@app.post("/query/image")
async def query_image(req: ImageQueryRequest):
    logging.info(f"CATALOG RETRIEVER | query_image() | Received POST.")
    texts, ids, sims, names, images = await retriever.retrieve(
        query=req.text,
        image=req.image_base64,
        categories=req.categories,
        filters=req.filters,
        k=req.k,
        image_bool=True,
        verbose=True
    )
    return {
        "texts": texts,
        "ids": ids,
        "similarities": sims,
        "names": names,
        "images": images
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0"
    }
