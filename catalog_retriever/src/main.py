from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from app.retriever import Retriever, RetrieverConfig
import time
import os
import yaml
import logging
import sys

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

# Get our configuration from config.yaml with optional override support
def load_config_with_override(base_config_path: str):
    """Load configuration from YAML file with optional override support."""
    # Load base config
    if not os.path.exists(base_config_path):
        logging.error(f"Base config file not found at {base_config_path}")
        raise FileNotFoundError(f"Base config file not found at {base_config_path}")

    with open(base_config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Check for override config
    override_file = os.environ.get("CONFIG_OVERRIDE")
    if override_file:
        # Construct override path (same directory as base config)
        base_dir = os.path.dirname(base_config_path)
        override_path = os.path.join(base_dir, override_file)
        
        if os.path.exists(override_path):
            logging.info(f"Loading override config from {override_path}")
            with open(override_path, "r") as f:
                override_config = yaml.safe_load(f)
            
            # Merge override config into base config
            config.update(override_config)
            logging.info(f"Config override applied from {override_file}")
        else:
            logging.warning(f"Override config file not found at {override_path}")
    else:
        logging.info("No config override specified, using base config only")
    
    return config

data = load_config_with_override("app/config/config.yaml")


# Setup Retriever once when app starts
config = RetrieverConfig(  
    text_embed_port=data["text_embed_port"],
    image_embed_port=data["image_embed_port"],
    text_model_name=data["text_model_name"],
    image_model_name=data["image_model_name"],
    db_port=data["db_port"],
    db_name=data["db_name"],
    sim_threshold=data["sim_threshold"],
    text_collection=data["text_collection"],
    image_collection=data["image_collection"]
)

logging.info("CATALOG RETRIEVER | startup | config.yaml ingested.")
logging.info("CATALOG RETRIEVER | startup | Initializing Retriever object.")
retriever = Retriever(config=config)
logging.info("CATALOG RETRIEVER | startup | Filling Milvus database.")
retriever.milvus_from_csv(csv_path=data["data_source"], verbose=True)
logging.info("CATALOG RETRIEVER | startup | Milvus database filled.")

# Request bodies
class TextQueryRequest(BaseModel):
    text: str = ""
    categories: List[str] = []
    k: int = 4

class ImageQueryRequest(BaseModel):
    text: str = ""
    image_base64: str = ""
    categories: List[str] = []
    k: int = 4

# Handles queries only containing text.
@app.post("/query/text")
async def query_text(req: TextQueryRequest):
    logging.info(f"CATALOG RETRIEVER | query_text() | Received POST: {req}.")
    texts, ids, sims, names, images = await retriever.retrieve(
        query=req.text,
        categories=req.categories,
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

