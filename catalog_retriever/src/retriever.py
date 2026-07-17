# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
A Retriever class that uses two NVIDIA NIM models to retriever relevant products from a database.
The first model uses image embeddings to retrieve the most relevant products.
The second model uses text embeddings to retrieve relevant products.
Performs both of these in parallel and then re-ranks the results from bothmodels.
"""

from openai import OpenAI
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
from math import isfinite
from typing import List, Tuple, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility
import os
import sys
import re
import numpy as np
from numpy import mean
from .utils import image_url_to_base64, is_url, is_path, image_path_to_base64, resize_base64_image
import logging
import asyncio
from types import SimpleNamespace
from shared.commerce_contracts import CatalogFilterCapability
from .catalog import CatalogSnapshot

# Set up logging 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# Defines a type for configuring the Retriever.
class RetrieverConfig(BaseModel):
    text_embed_port: str
    image_embed_port: str | None = None
    text_model_name: str
    image_model_name: str | None = None
    text_api_key_env: str | None = "EMBED_API_KEY"
    image_api_key_env: str | None = None
    image_enabled: bool = True
    db_port: str
    db_name: str
    sim_threshold: float
    text_collection: str
    image_collection: str
    filter_capabilities: Dict[str, CatalogFilterCapability] = Field(default_factory=dict)
    catalog_size: int
    product_id_field: str
    name_field: str
    description_field: str
    fallback_description_field: str | None
    image_field: str
    price_field: str
    taxonomy_fields: List[str]


class CatalogFilterError(ValueError):
    """Raised when a caller requests a filter outside catalog capabilities."""


@dataclass
class RetrievalOutput:
    texts: List[str] = field(default_factory=list)
    ids: List[str] = field(default_factory=list)
    similarities: List[float] = field(default_factory=list)
    names: List[str] = field(default_factory=list)
    images: List[str] = field(default_factory=list)
    products: List[Dict[str, Any]] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    no_result_reason: str | None = None

    def __iter__(self):
        yield self.texts
        yield self.ids
        yield self.similarities
        yield self.names
        yield self.images

# Defines a type for storing and embedding text.
class TextEmbeddings(Embeddings):
    def __init__(self, retriever):
        self.retriever = retriever

    def embed_query(self, text: str) -> List[float]:
        """Generate text embedding for a single text"""
        logging.info(f"TextEmbeddings | embed_query() | called.\n\t| input: {text[:50]}")
        res = self.retriever.embed_chunk(text)
        normed = res / np.linalg.norm(res)
        return normed

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate text embeddings for multiple texts"""
        logging.info("TextEmbeddings | embed_documents() | called.")
        res = self.retriever.text_embeddings(texts)
        normed = [list(r/np.linalg.norm(r) for r in res)]
        return normed

# Defines a type for storing and embedding images.
class ImageEmbeddings(Embeddings):
    def __init__(self, retriever):
        self.retriever = retriever

    def embed_query(self, text: str) -> List[float]:
        """Generate image embedding for a single image"""
        logging.info(f"ImageEmbeddings | embed_query() | called.\n\t| input: {text[:50]}")
        embeddings = self.retriever.image_embeddings([text], verbose=True)
        if embeddings and embeddings[0] is not None:
            logging.info(f"ImageEmbeddings | embed_query() | embedding output:\n\t| {embeddings[0][:50]}")
            return embeddings[0]
        else:
            logging.error("ImageEmbeddings | embed_query() | Failed to generate embedding for image")
            raise ValueError("Failed to generate image embedding")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate image embeddings for multiple images"""
        logging.info("ImageEmbeddings | embed_query() | called.")
        return self.retriever.image_embeddings(texts)


class Milvus:
    """
    Minimal Milvus adapter for the catalog retriever.

    This keeps the small vector-store API surface used by the retriever
    while relying directly on pymilvus so the vulnerable wrapper dependency is
    not required at runtime.
    """

    TEXT_FIELD = "text"
    VECTOR_FIELD = "vector"
    PK_FIELD = "pk"
    MAX_TEXT_LENGTH = 65535

    def __init__(
        self,
        embedding_function: Embeddings,
        collection_name: str,
        connection_args: Dict[str, Any],
        auto_id: bool = True,
        index_params: Dict[str, Any] | None = None,
    ) -> None:
        if not auto_id:
            raise ValueError("Catalog retriever Milvus adapter requires auto_id=True")

        self.embedding_function = embedding_function
        self.collection_name = collection_name
        self.connection_args = connection_args
        self.index_params = index_params or {"metric_type": "COSINE"}
        self.search_params = {
            "metric_type": self.index_params.get("metric_type", "COSINE"),
            "params": self.index_params.get("params", {}),
        }
        self.alias = self._connection_alias(collection_name, connection_args)
        connections.connect(alias=self.alias, **connection_args)
        self.col = self._load_collection_if_exists()

    @staticmethod
    def _connection_alias(collection_name: str, connection_args: Dict[str, Any]) -> str:
        uri = str(connection_args.get("uri", "default"))
        raw_alias = f"catalog_{collection_name}_{uri}"
        return re.sub(r"[^A-Za-z0-9_]", "_", raw_alias)[:255]

    def _load_collection_if_exists(self) -> Collection | None:
        if not utility.has_collection(self.collection_name, using=self.alias):
            return None

        collection = Collection(self.collection_name, using=self.alias)
        try:
            collection.load()
        except Exception as exc:
            logging.info(
                "CATALOG RETRIEVER | Milvus._load_collection_if_exists() | "
                f"Collection load skipped: {exc}"
            )
        return collection

    def _ensure_collection(self, dimension: int) -> Collection:
        if self.col is not None:
            return self.col

        schema = CollectionSchema(
            fields=[
                FieldSchema(
                    name=self.PK_FIELD,
                    dtype=DataType.INT64,
                    is_primary=True,
                    auto_id=True,
                ),
                FieldSchema(
                    name=self.TEXT_FIELD,
                    dtype=DataType.VARCHAR,
                    max_length=self.MAX_TEXT_LENGTH,
                ),
                FieldSchema(
                    name=self.VECTOR_FIELD,
                    dtype=DataType.FLOAT_VECTOR,
                    dim=dimension,
                ),
            ],
            description=f"{self.collection_name} collection",
            enable_dynamic_field=True,
        )
        self.col = Collection(self.collection_name, schema=schema, using=self.alias)
        self.col.create_index(
            field_name=self.VECTOR_FIELD,
            index_params={
                "metric_type": self.search_params["metric_type"],
                "index_type": "AUTOINDEX",
                "params": {},
            },
        )
        self.col.load()
        return self.col

    @staticmethod
    def _metadata_value(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, float) and np.isnan(value):
            return None
        return value

    @classmethod
    def _vector(cls, embedding: List[float]) -> List[float]:
        return [float(value) for value in embedding]

    def add_embeddings(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        records = []
        for text, embedding, metadata in zip(texts, embeddings, metadatas):
            vector = self._vector(embedding)
            record = {
                self.TEXT_FIELD: text[: self.MAX_TEXT_LENGTH],
                self.VECTOR_FIELD: vector,
            }
            for key, value in metadata.items():
                if key in {self.PK_FIELD, self.TEXT_FIELD, self.VECTOR_FIELD}:
                    continue
                cleaned = self._metadata_value(value)
                if cleaned is not None:
                    record[key] = cleaned
            records.append(record)

        if not records:
            return

        collection = self._ensure_collection(len(records[0][self.VECTOR_FIELD]))
        collection.insert(records)
        collection.flush()

    def matches_catalog(self, fingerprint: str, expected_count: int) -> bool:
        """Return whether this collection exactly represents one snapshot."""

        if self.col is None:
            return False
        try:
            self.col.flush()
            if self.col.num_entities != expected_count:
                return False
            escaped = fingerprint.replace("\\", "\\\\").replace('"', '\\"')
            matches = self.col.query(
                expr=f'catalog_fingerprint == "{escaped}"',
                output_fields=["catalog_fingerprint"],
                limit=expected_count,
            )
            return len(matches) == expected_count
        except Exception as exc:
            logging.info(
                "CATALOG RETRIEVER | Milvus.matches_catalog() | "
                f"Collection state could not be verified: {exc}"
            )
            return False

    def reset(self) -> None:
        """Drop the fixed collection so it can be rebuilt from one snapshot."""

        if utility.has_collection(self.collection_name, using=self.alias):
            utility.drop_collection(self.collection_name, using=self.alias)
        self.col = None

    def similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
    ) -> List[Tuple[SimpleNamespace, float]]:
        if self.col is None:
            return []

        query_vector = self._vector(self.embedding_function.embed_query(query))
        self.col.load()
        search_result = self.col.search(
            data=[query_vector],
            anns_field=self.VECTOR_FIELD,
            param=self.search_params,
            limit=k,
            output_fields=["*"],
        )

        results = []
        for hit in search_result[0]:
            fields = dict(hit.fields or {})
            page_content = fields.pop(self.TEXT_FIELD, "")
            fields.pop(self.VECTOR_FIELD, None)
            fields[self.PK_FIELD] = hit.id
            document = SimpleNamespace(page_content=page_content, metadata=fields)
            # Match langchain-milvus's COSINE relevance-score contract.
            relevance_score = (float(hit.score) + 1.0) / 2.0
            results.append((document, relevance_score))
        return results

class Retriever:
    """
    This class defines the core functionality of the retrieval container.
    """
    def __init__(
        self,
        config: RetrieverConfig
    ):
        self.text_embed_port = config.text_embed_port
        self.image_embed_port = config.image_embed_port
        self.text_model_name = config.text_model_name
        self.image_model_name = config.image_model_name
        self.text_api_key_env = config.text_api_key_env
        self.image_api_key_env = config.image_api_key_env
        self.image_enabled = config.image_enabled
        self.db_port = config.db_port
        self.db_name = config.db_name
        self.sim_threshold = config.sim_threshold
        self.text_collection = config.text_collection
        self.image_collection = config.image_collection
        self.filter_capabilities = config.filter_capabilities
        self.catalog_size = config.catalog_size
        self.product_id_field = config.product_id_field
        self.name_field = config.name_field
        self.description_field = config.description_field
        self.fallback_description_field = config.fallback_description_field
        self.image_field = config.image_field
        self.price_field = config.price_field
        self.taxonomy_fields = config.taxonomy_fields

        text_key = os.environ.get(self.text_api_key_env, "") if self.text_api_key_env else ""
        image_key = os.environ.get(self.image_api_key_env, "") if self.image_api_key_env else ""
        if self.text_api_key_env and not text_key:
            raise RuntimeError(f"Missing required text embedding key: {self.text_api_key_env}")
        if self.image_enabled and self.image_api_key_env and not image_key:
            raise RuntimeError(f"Missing required image embedding key: {self.image_api_key_env}")
        if self.image_enabled and (not self.image_embed_port or not self.image_model_name):
            raise RuntimeError("Image embeddings are enabled but image endpoint/model is missing.")

        self.text_client = OpenAI(
            api_key=text_key or "not-needed",
            base_url=self.text_embed_port
        )
        self.image_client = None
        if self.image_enabled:
            self.image_client = OpenAI(
                api_key=image_key or "not-needed",
                base_url=self.image_embed_port
            )

        # Create embedding classes
        self.text_embeddings_obj = TextEmbeddings(self)
        self.image_embeddings_obj = ImageEmbeddings(self) if self.image_enabled else None

        logging.info("CATALOG RETRIEVER | Retriever.__init__() | Initializing Milvus connections.")


        # Initialize Milvus with embedding classes
        self.text_db = Milvus(
            embedding_function=self.text_embeddings_obj,
            collection_name=self.text_collection,
            connection_args={"uri": f"{self.db_port}"},
            auto_id=True,
            index_params={"metric_type": "COSINE"},
        )
        self.image_db = None
        if self.image_enabled and self.image_embeddings_obj is not None:
            self.image_db = Milvus(
                embedding_function=self.image_embeddings_obj,
                collection_name=self.image_collection,
                connection_args={"uri": f"{self.db_port}"},
                auto_id=True,
                index_params={"metric_type": "COSINE"},
            )

        logging.info("CATALOG RETRIEVER | Retriever.__init__() | Milvus collections initialized.")

    def _embedding_counts(self) -> Tuple[int, int]:
        """Return current text and image collection entity counts."""
        text_count = 0
        if self.text_db.col:
            self.text_db.col.flush()
            text_count = self.text_db.col.num_entities

        image_count = -1 if not self.image_enabled else 0
        if self.image_db and self.image_db.col:
            self.image_db.col.flush()
            image_count = self.image_db.col.num_entities

        return text_count, image_count

    def embeddings_exist(self) -> bool:
        """
        Check if embeddings already exist in both text and image collections.
        Returns True if both collections have data, False otherwise.
        """
        try:
            text_count, image_count = self._embedding_counts()
            
            logging.info(f"CATALOG RETRIEVER | embeddings_exist() | Text collection has {text_count} entities. Image collection has {image_count} entities.")
            # Check text and image collections
            image_ready = (not self.image_enabled) or image_count > 0
            if text_count > 0 and image_ready:
                logging.info("CATALOG RETRIEVER | embeddings_exist() | Required embeddings found.")
                return True
            else:
                logging.info("CATALOG RETRIEVER | embeddings_exist() | No embeddings found in either collection.")
                return False
            
        except Exception as e:
            logging.info(f"CATALOG RETRIEVER | embeddings_exist() | Error checking embeddings: {e}")
            return False

    def embed_chunk(
        self, 
        chunk: str, 
        query_type: str = "query"
        ) -> List[float]:
        """
        Embed a chunk of text.
        """
        response = self.text_client.embeddings.create(
            input=chunk,
            model=self.text_model_name,
            encoding_format="float",
            extra_body={"input_type": query_type, "truncate": "NONE"}
        )

        logging.info("CATALOG RETRIEVER | Retriever.embed_chunk() | Chunk embedded.")

        return response.data[0].embedding   

    def text_embeddings(
        self,
        texts: List[str],
        query_type: str = "query",
        verbose: bool = False
    ) -> List[List[float] | None]:
        """
        Generate text embeddings from a list of text strings, using chunking and batching.
        """
        if not texts:
            return []

        all_chunks, text_chunk_counts = self._create_text_chunks(texts, verbose)
        if not all_chunks:
            return [None] * len(texts)

        all_chunk_embeddings = self._embed_chunks_in_batches(all_chunks, query_type, verbose)
        
        final_embeddings = self._reconstruct_embeddings(texts, all_chunk_embeddings, text_chunk_counts)
        
        return final_embeddings

    def _create_text_chunks(self, texts: List[str], verbose: bool = False) -> Tuple[List[str], List[int]]:
        """
        Break all input texts into smaller chunks and return the chunks and their counts.
        """
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_chunks = []
        text_chunk_counts = []
        for text in texts:
            chunks = text_splitter.split_text(text)
            all_chunks.extend(chunks)
            text_chunk_counts.append(len(chunks))
        if verbose:
            logging.info(f"CATALOG RETRIEVER | Retriever.text_embeddings() | Created {len(all_chunks)} chunks from {len(texts)} texts.")
        return all_chunks, text_chunk_counts

    def _embed_chunks_in_batches(
        self,
        all_chunks: List[str],
        query_type: str,
        verbose: bool = False,
        batch_size: int = 32
    ) -> List[List[float] | None]:
        """
        Embed all created chunks in efficient batches.
        """
        all_chunk_embeddings = []
        num_batches = (len(all_chunks) + batch_size - 1) // batch_size
        for i in range(0, len(all_chunks), batch_size):
            batch_chunks = all_chunks[i:i + batch_size]
            if verbose:
                logging.info(f"CATALOG RETRIEVER | Retriever.text_embeddings() | Processing text chunk batch {i//batch_size + 1}/{num_batches} with {len(batch_chunks)} chunks.")
            try:
                response = self.text_client.embeddings.create(
                    input=batch_chunks,
                    model=self.text_model_name,
                    encoding_format="float",
                    extra_body={"input_type": query_type, "truncate": "NONE"}
                )
                all_chunk_embeddings.extend([d.embedding for d in response.data])
            except Exception as e:
                if verbose:
                    logging.error(f"CATALOG RETRIEVER | Retriever.text_embeddings() | Error embedding chunk batch: {e}")
                all_chunk_embeddings.extend([None for _ in batch_chunks])
        return all_chunk_embeddings

    def _reconstruct_embeddings(
        self,
        texts: List[str],
        all_chunk_embeddings: List[List[float] | None],
        text_chunk_counts: List[int]
    ) -> List[List[float] | None]:
        """
        Reconstruct a single embedding for each original text from chunk embeddings.
        """
        final_embeddings = []
        current_chunk_idx = 0
        for i, text in enumerate(texts):
            num_chunks = text_chunk_counts[i]
            if num_chunks == 0:
                final_embeddings.append(None)
                continue

            chunk_embeddings = all_chunk_embeddings[current_chunk_idx : current_chunk_idx + num_chunks]
            current_chunk_idx += num_chunks

            valid_chunk_embeddings = [emb for emb in chunk_embeddings if emb is not None]

            if valid_chunk_embeddings:
                average_embedding = list(mean(valid_chunk_embeddings, axis=0))
                final_embeddings.append(average_embedding)
            else:
                final_embeddings.append(None)
        
        return final_embeddings

    def image_embeddings(
        self,
        texts: List[str],
        verbose: bool = False
    ) -> List[List[float] | None]:
        """
        Generate image embeddings from a list of base64 image strings or image URLs using batching.
        Returns a list of embeddings, with None for failures, to maintain 1:1 mapping with input.
        """
        if not self.image_enabled or self.image_client is None or not self.image_model_name:
            logging.info("CATALOG RETRIEVER | Retriever.image_embeddings() | Image embeddings are disabled.")
            return [None for _ in texts]

        all_embeddings = []
        batch_size = 32
        num_batches = (len(texts) + batch_size - 1) // batch_size

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            if verbose:
                logging.info(f"CATALOG RETRIEVER | Retriever.image_embeddings() | Processing image batch {i//batch_size + 1}/{num_batches} with {len(batch_texts)} images.")
            
            input_data_list = []
            
            for text in batch_texts:
                try:
                    input_data = text
                    if is_url(text):
                        input_data = image_url_to_base64(text)
                    elif is_path(text):
                        input_data = image_path_to_base64(text)

                    MAX_VARCHAR_LENGTH = 65535
                    if len(input_data) > MAX_VARCHAR_LENGTH:
                        if verbose:
                            logging.info(f"CATALOG RETRIEVER | Image too large ({len(input_data)} bytes), resizing...")
                        # Try to resize the image
                        resized = resize_base64_image(input_data)
                        if resized and len(resized) <= MAX_VARCHAR_LENGTH:
                            input_data = resized
                            if verbose:
                                logging.info(f"CATALOG RETRIEVER | Image resized successfully to {len(input_data)} bytes")
                        else:
                            if verbose:
                                logging.warning("CATALOG RETRIEVER | Failed to resize image or still too large after resize")
                            input_data = None 
                except Exception as e:
                    if verbose:
                        logging.error(f"CATALOG RETRIEVER | Error processing image for batching: {e}")
                    input_data = None
                input_data_list.append(input_data)

            valid_inputs = [data for data in input_data_list if data is not None]
            
            try:
                if valid_inputs:
                    response = self.image_client.embeddings.create(
                        input=valid_inputs,
                        model=self.image_model_name,
                        encoding_format="float",
                    )
                    batch_embeddings = iter([d.embedding for d in response.data])
                else:
                    batch_embeddings = iter([])

            except Exception as e:
                if verbose:
                    error_msg = str(e)
                    if "webp" in error_msg.lower():
                        logging.error(f"CATALOG RETRIEVER | Unsupported image format detected (WebP). Only JPEG and PNG are supported: {e}")
                    elif "format" in error_msg.lower() or "expected" in error_msg.lower():
                        logging.error(f"CATALOG RETRIEVER | Image format error. Only JPEG and PNG are supported: {e}")
                    else:
                        logging.error(f"CATALOG RETRIEVER | Retriever.image_embeddings() | Error embedding image batch: {e}")
                batch_embeddings = iter([])

            # Reconstruct the batch with Nones for failed embeddings
            reconstructed_batch = []
            for data in input_data_list:
                if data is not None:
                    try:
                        embedding = next(batch_embeddings)
                        reconstructed_batch.append(embedding)
                    except StopIteration:
                        # If we run out of embeddings, add None for remaining items
                        reconstructed_batch.append(None)
                else:
                    reconstructed_batch.append(None)
            all_embeddings.extend(reconstructed_batch)

        return all_embeddings



    def sync_snapshot(self, snapshot: CatalogSnapshot, verbose: bool = False) -> None:
        """Reuse or rebuild the fixed collections for one validated snapshot."""

        record = snapshot.schema.record
        self.catalog_size = snapshot.product_count
        self.product_id_field = record.product_id
        self.name_field = record.name
        self.description_field = record.description
        self.fallback_description_field = record.fallback_description
        self.image_field = record.image
        self.price_field = record.price
        self.taxonomy_fields = snapshot.schema.taxonomy.fields
        self.filter_capabilities = snapshot.capabilities.filters
        text_ready = self.text_db.matches_catalog(
            snapshot.fingerprint, snapshot.product_count
        )
        image_ready = (not self.image_enabled) or bool(
            self.image_db
            and self.image_db.matches_catalog(
                snapshot.fingerprint, snapshot.product_count
            )
        )
        if text_ready and image_ready:
            logging.info(
                "CATALOG RETRIEVER | Retriever.sync_snapshot() | "
                "Indexes match the active catalog; reusing them."
            )
            return

        logging.info(
            "CATALOG RETRIEVER | Retriever.sync_snapshot() | "
            f"Rebuilding indexes for {snapshot.product_count} products."
        )
        self.text_db.reset()
        if self.image_db is not None:
            self.image_db.reset()

        metadatas = [
            {**product, "catalog_fingerprint": snapshot.fingerprint}
            for product in snapshot.products
        ]
        text_embeddings = self.text_embeddings(
            list(snapshot.search_documents),
            query_type="passage",
            verbose=verbose,
        )
        if len(text_embeddings) != snapshot.product_count or any(
            embedding is None for embedding in text_embeddings
        ):
            raise RuntimeError("Catalog text indexing failed; no partial snapshot will be served")
        self.text_db.add_embeddings(
            texts=list(snapshot.search_documents),
            embeddings=[embedding for embedding in text_embeddings if embedding is not None],
            metadatas=metadatas,
        )

        if self.image_enabled:
            if self.image_db is None:
                raise RuntimeError("Catalog image index is enabled but unavailable")
            image_references = [
                str(product[snapshot.schema.record.image])
                for product in snapshot.products
            ]
            image_embeddings = self.image_embeddings(image_references, verbose=verbose)
            if len(image_embeddings) != snapshot.product_count or any(
                embedding is None for embedding in image_embeddings
            ):
                raise RuntimeError(
                    "Catalog image indexing failed; no partial snapshot will be served"
                )
            self.image_db.add_embeddings(
                texts=image_references,
                embeddings=[
                    embedding for embedding in image_embeddings if embedding is not None
                ],
                metadatas=metadatas,
            )

        text_count, image_count = self._embedding_counts()
        if text_count != snapshot.product_count or (
            self.image_enabled and image_count != snapshot.product_count
        ):
            raise RuntimeError(
                "Catalog index count does not match the active catalog snapshot"
            )

    async def retrieve(
        self,
        query: List[str],
        categories: List[str],
        filters: Dict[str, Any] | None = None,
        image: str = "",
        k: int = 4,
        candidate_k: int | None = None,
        image_bool: bool = False,
        verbose: bool = True
    ) -> RetrievalOutput:
        """
        Asynchronously retrieve relevant items from both text and image databases.
        """
        candidate_limit = max(k, candidate_k or self.catalog_size or (k * 5))
        diagnostics: Dict[str, Any] = {
            "requested_top_k": k,
            "candidate_k": candidate_limit,
            "search_mode": "image" if image_bool else "text",
        }
        effective_filters = self._effective_filters(filters, categories)
        structured_filters = (
            self._canonical_filters(effective_filters) if effective_filters else {}
        )

        # Check if our query is blank. If it is, replace it with dummy text.
        local_queries = query
        if not query:
            local_queries = ["Can you find me something like this image?"]

        if image_bool:
            if not self.image_enabled or self.image_db is None:
                logging.info("CATALOG RETRIEVER | retrieve() | Image retrieval requested but image embeddings are disabled.")
                diagnostics["returned_count"] = 0
                return RetrievalOutput(
                    diagnostics=diagnostics,
                    no_result_reason="image_embeddings_disabled",
                )

            if verbose:
                logging.info("CATALOG RETRIEVER | retrieve() | Performing dual retrieval for image input.")

            # Use asyncio.gather for concurrency
            t2t_tasks = []
            for local_query in local_queries:
                if verbose:
                    logging.info(f"\t| retrieve() | Checking query: {local_query}.")
                t2t_tasks.append(
                    asyncio.to_thread(
                        self.text_db.similarity_search_with_relevance_scores,
                        local_query,
                        k=candidate_limit,
                    )
                )
            if verbose:
                logging.info("CATALOG RETRIEVER | retrieve() | Started text task.")
            base64_string = image.replace("data:application/octet-stream", "data:image/jpeg")
            if verbose:
                logging.info(f"CATALOG RETRIEVER | retrieve() | Starting image task...\n\t| {base64_string[:100]}")
            if verbose:
                logging.info("CATALOG RETRIEVER | retrieve() | Obtained embedding...")
            i2i_task = asyncio.to_thread(
                self.image_db.similarity_search_with_relevance_scores,
                base64_string,
                k=candidate_limit,
            )

            unformatted_results = await asyncio.gather(*t2t_tasks, i2i_task)
        else:
            if verbose:
                logging.info(f"CATALOG RETRIEVER | retrieve() | Text-only retrieval. Queries: {local_queries}")

            results  = []
            for local_query in local_queries:
                if verbose:
                    logging.info(f"\t| retrieve() | Launching text-only retrieval. Query type: {type(local_query)}, Query: {local_query}")
                results.append(
                    asyncio.to_thread(
                        self.text_db.similarity_search_with_relevance_scores,
                        local_query,
                        k=candidate_limit,
                    )
                )
            unformatted_results = await asyncio.gather(*results)

        diagnostics["source_result_count"] = sum(
            len(query_results) for query_results in unformatted_results
        )

        sorted_unformatted_results = []
        for query_results in unformatted_results:
            # Sort each list of (Document, score) tuples by the score in descending order
            sorted_query_results = sorted(query_results, key=lambda item: item[1], reverse=True)
            sorted_unformatted_results.append(sorted_query_results)

        if verbose:
            logging.info(f"""CATALOG RETRIEVER | retrieve() | Pre-interleaving data
                            \n\t| Similarities: {[res[1] for sublist in sorted_unformatted_results for res in sublist]}
                            \n\t| Names: {[res[0].metadata.get(self.name_field) for sublist in sorted_unformatted_results for res in sublist]}""")

        # For image search, combine all results and sort by similarity instead of interleaving
        if image_bool:
            # Combine all results from different sources
            all_unformatted_results = []
            for sublist in sorted_unformatted_results:
                all_unformatted_results.extend(sublist)
            # Sort combined results by similarity score
            interleaved_results = sorted(all_unformatted_results, key=lambda item: item[1], reverse=True)
        else:
            # For text-only search, use interleaving as before
            interleaved_results = []
            # Store them in a regular list
            active_iterators = [iter(lst) for lst in sorted_unformatted_results] 
            while active_iterators:
                current_it = active_iterators.pop(0)
                try:
                    item = next(current_it)
                    interleaved_results.append(item)
                    active_iterators.append(current_it)
                except StopIteration:
                    pass
                
        # Deduplicate by source product identity. Display names are not IDs and
        # two legitimate products may share one.
        seen_ids = set()
        final_results = [] 
        for res in interleaved_results:
            pk_value = res[0].metadata.get("pk") 
            product_id = res[0].metadata.get(self.product_id_field)
            id_ = str(product_id) if product_id is not None else (
                str(pk_value) if pk_value is not None else None
            )
            if id_ is not None and id_ not in seen_ids:
                seen_ids.add(id_)
                final_results.append(res)
        
        all_results = final_results
        diagnostics["deduped_count"] = len(all_results)

        if verbose:
            logging.info(f"""CATALOG RETRIEVER | retrieve() | All retrieved results length. {len(all_results)}
                            \n\t| Similarities: {[res[1] for res in all_results]}
                            \n\t| Names: {[res[0].metadata.get(self.name_field) for res in all_results]}""")

        if not all_results:
            diagnostics["returned_count"] = 0
            return RetrievalOutput(
                diagnostics=diagnostics,
                no_result_reason="no_candidates",
            )

        # Apply hard filters and the similarity threshold across the complete
        # candidate window before trimming to the final top-k response.
        candidate_results = all_results[:candidate_limit]
        diagnostics["candidate_window_count"] = len(candidate_results)
        filtered_results = self._apply_structured_filters(
            candidate_results,
            filters=structured_filters,
            verbose=verbose,
            canonical=True,
        )
        diagnostics["after_filter_count"] = len(filtered_results)
        if not filtered_results:
            diagnostics["returned_count"] = 0
            return RetrievalOutput(
                diagnostics=diagnostics,
                no_result_reason="filtered_out",
            )

        thresholded_results = [
            res for res in filtered_results if res[1] > self.sim_threshold
        ]
        diagnostics["after_threshold_count"] = len(thresholded_results)
        if not thresholded_results:
            diagnostics["returned_count"] = 0
            return RetrievalOutput(
                diagnostics=diagnostics,
                no_result_reason="below_similarity_threshold",
            )
        ranked_results = sorted(
            thresholded_results,
            key=lambda item: item[1],
            reverse=True,
        )
        ranked_results = ranked_results[:k]
        diagnostics["returned_count"] = len(ranked_results)

        if verbose:
            logging.info(
                "CATALOG RETRIEVER | retrieve() | "
                f"Ranked window after threshold+filters: {len(ranked_results)}"
            )

        final_texts = [
            res[0].page_content
            + f"\nPRICE: {res[0].metadata.get(self.price_field)}"
            for res in ranked_results
        ]
        final_ids = [
            str(
                res[0].metadata.get(self.product_id_field)
                or res[0].metadata.get("pk")
            )
            for res in ranked_results
        ]
        final_sims = [res[1] for res in ranked_results]
        final_names = [
            str(res[0].metadata.get(self.name_field) or "")
            for res in ranked_results
        ]
        final_images = [
            str(res[0].metadata.get(self.image_field) or "")
            for res in ranked_results
        ]
        final_products = [
            self._product_payload_from_result(res)
            for res in ranked_results
        ]

        if verbose:
            logging.info(f"CATALOG RETRIEVER | retrieve() | \n\tnames: {final_names} \n\tsimilarities: {final_sims}")

        return RetrievalOutput(
            texts=final_texts,
            ids=final_ids,
            similarities=final_sims,
            names=final_names,
            images=final_images,
            products=final_products,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        """Best-effort conversion to float for numeric filter/metadata values."""
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            try:
                number = float(value)
            except (OverflowError, ValueError):
                return None
            return number if isfinite(number) else None
        if isinstance(value, str):
            cleaned = value.strip().replace("$", "").replace(",", "")
            try:
                number = float(cleaned)
            except (OverflowError, ValueError):
                return None
            return number if isfinite(number) else None
        return None

    def _apply_structured_filters(
        self,
        results: List[Tuple[Any, float]],
        filters: Dict[str, Any] | None,
        verbose: bool = False,
        *,
        canonical: bool = False,
    ) -> List[Tuple[Any, float]]:
        """
        Apply structured metadata filters before assembling final response payloads.
        """
        if not filters:
            return results

        canonical_filters = filters if canonical else self._canonical_filters(filters)
        if not canonical_filters:
            return results

        filtered_results: List[Tuple[Any, float]] = []
        for result in results:
            doc = result[0]
            if not self._metadata_matches_filters(doc.metadata, canonical_filters):
                continue
            filtered_results.append(result)

        if verbose:
            logging.info(
                "CATALOG RETRIEVER | _apply_structured_filters() | "
                f"filters={filters} | input={len(results)} | output={len(filtered_results)}"
            )

        return filtered_results

    def _canonical_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        aliases = {
            alias
            for capability in self.filter_capabilities.values()
            for alias in capability.request_aliases.values()
        }
        unknown = sorted(set(filters) - set(self.filter_capabilities) - aliases)
        if unknown:
            raise CatalogFilterError(
                "Unsupported catalog filter(s): " + ", ".join(unknown)
            )

        canonical: Dict[str, Any] = {}
        for name, capability in self.filter_capabilities.items():
            if name in filters:
                canonical[name] = self._validated_filter_value(
                    name, filters[name], capability
                )
            if capability.type == "number":
                alias_filter = self._number_alias_filter(filters, name, capability)
                if alias_filter:
                    existing = canonical.get(name)
                    if isinstance(existing, dict):
                        duplicate_bounds = sorted(set(existing).intersection(alias_filter))
                        if duplicate_bounds:
                            raise CatalogFilterError(
                                f"Numeric filter '{name}' cannot combine canonical "
                                "and top-level aliases for bound(s): "
                                + ", ".join(duplicate_bounds)
                            )
                        canonical[name] = {**existing, **alias_filter}
                    else:
                        canonical[name] = alias_filter
                    canonical[name] = self._validated_filter_value(
                        name, canonical[name], capability
                    )
        return {
            name: value
            for name, value in canonical.items()
            if value not in (None, "", [], {})
        }

    def _validated_filter_value(
        self,
        name: str,
        value: Any,
        capability: CatalogFilterCapability,
    ) -> Any:
        if capability.type == "number":
            if not isinstance(value, dict):
                raise CatalogFilterError(
                    f"Numeric filter '{name}' requires min/max or gte/lte bounds"
                )
            unsupported = set(value) - {"min", "max", "gte", "lte"}
            if unsupported:
                raise CatalogFilterError(
                    f"Unsupported operator(s) for '{name}': "
                    + ", ".join(sorted(unsupported))
                )
            duplicate_aliases = [
                "/".join(aliases)
                for aliases in (("min", "gte"), ("max", "lte"))
                if all(alias in value for alias in aliases)
            ]
            if duplicate_aliases:
                raise CatalogFilterError(
                    f"Numeric filter '{name}' cannot combine bound aliases: "
                    + ", ".join(duplicate_aliases)
                )
            invalid_bounds = [
                operator
                for operator, raw_bound in value.items()
                if self._coerce_float(raw_bound) is None
            ]
            if invalid_bounds:
                raise CatalogFilterError(
                    f"Numeric filter '{name}' has invalid bound(s): "
                    + ", ".join(sorted(invalid_bounds))
                )
            bounds = self._normalize_number_filter(value)
            if not bounds:
                raise CatalogFilterError(f"Numeric filter '{name}' has no valid bounds")
            if (
                bounds.get("min") is not None
                and bounds.get("max") is not None
                and bounds["min"] > bounds["max"]
            ):
                raise CatalogFilterError(
                    f"Numeric filter '{name}' has a minimum above its maximum"
                )
            return bounds

        requested = self._normalize_filter_values(value)
        if not requested:
            raise CatalogFilterError(f"Filter '{name}' has no values")
        if capability.type in {"enum", "enum_list"}:
            allowed = {item.casefold(): item for item in capability.values}
            invalid = sorted(item for item in requested if item.casefold() not in allowed)
            if invalid:
                raise CatalogFilterError(
                    f"Unsupported value(s) for '{name}': " + ", ".join(invalid)
                )
            return [allowed[item.casefold()] for item in requested]
        return sorted(requested)

    @staticmethod
    def _number_alias_filter(
        filters: Dict[str, Any],
        name: str,
        capability: CatalogFilterCapability,
    ) -> Dict[str, Any]:
        aliases = {
            "min": capability.request_aliases.get("min") or f"min_{name}",
            "max": capability.request_aliases.get("max") or f"max_{name}",
        }
        number_filter: Dict[str, Any] = {}
        for bound, alias in aliases.items():
            if alias in filters:
                number_filter[bound] = filters[alias]
        return number_filter

    def _metadata_matches_filters(
        self,
        metadata: Dict[str, Any],
        filters: Dict[str, Any],
    ) -> bool:
        for name, value in filters.items():
            capability = self.filter_capabilities.get(name)
            if capability is None:
                continue
            if capability.type == "number":
                if not self._metadata_matches_number_filter(
                    metadata, value, name, capability
                ):
                    return False
                continue
            if not self._metadata_matches_value_filter(metadata, value, name, capability):
                return False
        return True

    def _metadata_matches_number_filter(
        self,
        metadata: Dict[str, Any],
        value: Any,
        name: str,
        capability: CatalogFilterCapability,
    ) -> bool:
        bounds = self._normalize_number_filter(value)
        if not bounds:
            return True
        if (
            bounds.get("min") is not None
            and bounds.get("max") is not None
            and bounds["min"] > bounds["max"]
        ):
            return False

        metadata_value = self._first_metadata_number(metadata, name, capability)
        if metadata_value is None:
            return False
        if bounds.get("min") is not None and metadata_value < bounds["min"]:
            return False
        if bounds.get("max") is not None and metadata_value > bounds["max"]:
            return False
        return True

    @classmethod
    def _normalize_number_filter(cls, value: Any) -> Dict[str, float]:
        if not isinstance(value, dict):
            return {}
        bounds: Dict[str, float] = {}
        lower = cls._coerce_float(value.get("min", value.get("gte")))
        upper = cls._coerce_float(value.get("max", value.get("lte")))
        if lower is not None:
            bounds["min"] = lower
        if upper is not None:
            bounds["max"] = upper
        return bounds

    @classmethod
    def _first_metadata_number(
        cls,
        metadata: Dict[str, Any],
        name: str,
        capability: CatalogFilterCapability,
    ) -> float | None:
        for source_field in capability.source_fields or [name]:
            value = cls._coerce_float(metadata.get(source_field))
            if value is not None:
                return value
        return None

    def _metadata_matches_value_filter(
        self,
        metadata: Dict[str, Any],
        value: Any,
        name: str,
        capability: CatalogFilterCapability,
    ) -> bool:
        requested_values = self._normalize_filter_values(value)
        if not requested_values:
            return True
        metadata_values: set[str] = set()
        for source_field in capability.source_fields or [name]:
            raw_value = metadata.get(source_field)
            values = raw_value if isinstance(raw_value, list) else [raw_value]
            metadata_values.update(
                normalized
                for item in values
                if (normalized := self._normalize_filter_text(item))
            )
        return bool(metadata_values.intersection(requested_values))

    @classmethod
    def _normalize_filter_values(cls, value: Any) -> set[str]:
        if value is None:
            return set()
        if isinstance(value, list):
            return {
                normalized
                for item in value
                if (normalized := cls._normalize_filter_text(item))
            }
        text = cls._normalize_filter_text(value)
        return {text} if text else set()

    @staticmethod
    def _normalize_filter_text(value: Any) -> str:
        if value is None:
            return ""
        return " ".join(str(value).split()).casefold()

    def _effective_filters(
        self,
        filters: Dict[str, Any] | None,
        categories: List[str],
    ) -> Dict[str, Any]:
        effective = dict(filters or {})
        if categories:
            explicit_taxonomy = sorted(
                field for field in self.taxonomy_fields if field in effective
            )
            if explicit_taxonomy:
                raise CatalogFilterError(
                    "Legacy 'categories' cannot be combined with explicit "
                    "taxonomy filter(s): " + ", ".join(explicit_taxonomy)
                )
            requested = {str(value).strip().casefold() for value in categories}
            matched = False
            for field in self.taxonomy_fields:
                capability = self.filter_capabilities.get(field)
                if capability is None:
                    continue
                available = {value.casefold() for value in capability.values}
                if requested and requested.issubset(available):
                    effective[field] = categories
                    matched = True
                    break
            if requested and not matched:
                raise CatalogFilterError(
                    "Unsupported taxonomy value(s): "
                    + ", ".join(sorted(str(value) for value in categories))
                )
        return effective

    def _product_payload_from_result(self, result: Tuple[Any, float]) -> Dict[str, Any]:
        doc, similarity = result
        metadata = doc.metadata
        price = Retriever._coerce_float(metadata.get(self.price_field))
        description = metadata.get(self.description_field)
        if not description and self.fallback_description_field:
            description = metadata.get(self.fallback_description_field)
        taxonomy = {
            field: metadata.get(field)
            for field in self.taxonomy_fields
            if metadata.get(field) not in (None, "")
        }
        category = list(taxonomy.values())[-1] if taxonomy else ""
        product: Dict[str, Any] = {
            "product_id": str(
                metadata.get(self.product_id_field) or metadata.get("pk") or ""
            ),
            "display_name": str(metadata.get(self.name_field) or ""),
            "description": str(description or doc.page_content or ""),
            "category": str(category),
            "image_url": str(metadata.get(self.image_field) or ""),
            "attributes": {
                "catalog_text": (
                    doc.page_content + f"\nPRICE: {metadata.get(self.price_field)}"
                ),
                "similarity": float(similarity),
                "taxonomy": taxonomy,
            },
        }
        if price is not None:
            product["price"] = {"amount": price, "currency": "USD"}
        return product
