"""
A Retriever class that uses two NVIDIA NIM models to retriever relevant products from a database.
The first model uses image embeddings to retrieve the most relevant products.
The second model uses text embeddings to retrieve relevant products.
Performs both of these in parallel and then re-ranks the results from bothmodels.
"""

from openai import OpenAI
from pydantic import BaseModel
from typing import List, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_milvus import Milvus
import os
import sys
import re
import pandas as pd
import numpy as np
from numpy import mean
from .utils import image_url_to_base64, is_url, is_path, image_path_to_base64
import logging
import asyncio

# Set up logging 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# Defines a type for configuring the Retriever.
class RetrieverConfig(BaseModel):
    text_embed_port: str
    image_embed_port: str
    text_model_name: str
    image_model_name: str
    db_port: str
    db_name: str
    sim_threshold: float
    text_collection: str
    image_collection: str

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
        logging.info(f"TextEmbeddings | embed_documents() | called.")
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
        _, embedding = self.retriever.image_embeddings([text])
        logging.info(f"ImageEmbeddings | embed_query() | embedding output:\n\t| {embedding[0][:50]}")
        return embedding[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate image embeddings for multiple images"""
        logging.info(f"ImageEmbeddings | embed_query() | called.")
        return self.retriever.image_embeddings(texts)

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
        self.db_port = config.db_port
        self.db_name = config.db_name
        self.sim_threshold = config.sim_threshold
        self.text_collection = config.text_collection
        self.image_collection = config.image_collection

        # Keys.
        embed_key = os.environ["EMBED_API_KEY"]

        self.text_client = OpenAI(
            api_key=embed_key,
            base_url=self.text_embed_port
        )
        self.image_client = OpenAI(
            api_key=embed_key,
            base_url=self.image_embed_port
        )

        # Create embedding classes
        self.text_embeddings_obj = TextEmbeddings(self)
        self.image_embeddings_obj = ImageEmbeddings(self)

        logging.info(f"CATALOG RETRIEVER | Retriever.__init__() | Initializing Milvus connections.")


        # Initialize Milvus with embedding classes
        self.text_db = Milvus(
            embedding_function=self.text_embeddings_obj,
            collection_name=self.text_collection,
            connection_args={"uri": f"{self.db_port}"},
            auto_id=True,
            index_params={"metric_type": "COSINE"},
        )
        self.image_db = Milvus(
            embedding_function=self.image_embeddings_obj,
            collection_name=self.image_collection,
            connection_args={"uri": f"{self.db_port}"},
            auto_id=True,
            index_params={"metric_type": "COSINE"},
        )

        logging.info(f"CATALOG RETRIEVER | Retriever.__init__() | Milvus collections initialized.")

    def embeddings_exist(self) -> bool:
        """
        Check if embeddings already exist in both text and image collections.
        Returns True if both collections have data, False otherwise.
        """
        try:
            self.text_db.col.flush()
            self.image_db.col.flush()
            text_count = self.text_db.col.num_entities
            image_count = self.image_db.col.num_entities
            
            logging.info(f"CATALOG RETRIEVER | embeddings_exist() | Text collection has {text_count} entities. Image collection has {image_count} entities.")
            # Check text and image collections
            if text_count > 0 and image_count > 0:
                logging.info("CATALOG RETRIEVER | embeddings_exist() | Embeddings found in both collections.")
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

        logging.info(f"CATALOG RETRIEVER | Retriever.embed_chunk() | Chunk embedded.")

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
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )

        # Break all input texts into smaller chunks.
        all_chunks = []
        # We track chunk counts per text to correctly reconstruct the final embeddings later.
        # This ensures the output list of embeddings aligns 1:1 with the input list of texts.
        text_chunk_counts = []
        for text in texts:
            chunks = text_splitter.split_text(text)
            all_chunks.extend(chunks)
            text_chunk_counts.append(len(chunks))

        if verbose:
            logging.info(f"CATALOG RETRIEVER | Retriever.text_embeddings() | Created {len(all_chunks)} chunks from {len(texts)} texts.")

        if not all_chunks:
            return [None] * len(texts)

        # Embed all created chunks in efficient batches
        all_chunk_embeddings = []
        batch_size = 32
        # Calculate the number of batches needed to embed all chunks for logging purposes.
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
        
        # Reconstruct a single embedding for each original text.
        final_embeddings = []
        current_chunk_idx = 0
        # Loop through the original texts to ensure the final list of embeddings is in the same order.
        for i, text in enumerate(texts):
            # Get the number of chunks that were created for this specific text.
            num_chunks = text_chunk_counts[i]
            if num_chunks == 0:
                final_embeddings.append(None)
                continue

            # Slice the `all_chunk_embeddings` list to get the embeddings for the current text.
            chunk_embeddings = all_chunk_embeddings[current_chunk_idx : current_chunk_idx + num_chunks]
            # Move the index forward for the next iteration.
            current_chunk_idx += num_chunks

            # Filter out any chunks that may have failed to embed.
            valid_chunk_embeddings = [emb for emb in chunk_embeddings if emb is not None]

            # If there are valid chunks, average their embeddings to create a single representative vector.
            if valid_chunk_embeddings:
                average_embedding = list(mean(valid_chunk_embeddings, axis=0))
                final_embeddings.append(average_embedding)
            else:
                # If all chunks for a text failed, mark the final embedding as None.
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
                            logging.warning(f"CATALOG RETRIEVER | Skipping image embedding, too large: {len(input_data)} bytes.")
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
                    logging.error(f"CATALOG RETRIEVER | Retriever.image_embeddings() | Error embedding image batch: {e}")
                batch_embeddings = iter([])

            # Reconstruct the batch with Nones for failed embeddings
            reconstructed_batch = [next(batch_embeddings) if data is not None else None for data in input_data_list]
            all_embeddings.extend(reconstructed_batch)

        return all_embeddings



    def milvus_from_csv(self, csv_path: str, verbose: bool = False) -> None:
        """
        Fills the milvus database with the data from a CSV file.
        Only populates if embeddings don't already exist.
        """ 

        # Check if embeddings already exist
        if self.embeddings_exist():
            logging.info("CATALOG RETRIEVER | Retriever.milvus_from_csv() | Embeddings already exist, skipping population.")
            return

        logging.info(f"CATALOG RETRIEVER | Retriever.milvus_from_csv() | No embeddings found, populating from: '{csv_path}'")

        # Get our pd dataframe
        try:
            df = pd.read_csv(csv_path)
            logging.info(f"CATALOG RETRIEVER | Retriever.milvus_from_csv() | CSV read in.")
        except Exception as e:
            logging.debug(f"CATALOG RETRIEVER | Retriever.milvus_from_csv() | Error: {e} -- Failed to read CSV: {csv_path}.")            
            dir_contents = []
            for entry in os.listdir("."):
                dir_contents.append(entry)
            logging.info(f"CATALOG RETRIEVER | Retriever.milvus_from_csv() | Directory contents at failure: {dir_contents}")

        # Create combined name and description strings
        metadatas = df.to_dict(orient="records")
        combined_texts = [f"{name} | {desc} | {category},{subcategory}" for name, desc, category, subcategory in zip(df["name"].tolist(), df["description"].tolist(), df["category"].tolist(), df["subcategory"].tolist())]
        
        # Embed the combined name and description fields
        text_embs = self.text_embeddings(combined_texts,query_type="passage",verbose=verbose)

        # Filter out failed embeddings and their corresponding metadata
        successful_texts_data = [
            (text, emb, meta) for text, emb, meta in zip(combined_texts, text_embs, metadatas) if emb is not None
        ]
        if successful_texts_data:
            successful_texts, successful_text_embs, successful_text_metadatas = zip(*successful_texts_data)
            self.text_db.add_embeddings(
                texts=list(successful_texts),
                embeddings=list(successful_text_embs),
                metadatas=list(successful_text_metadatas)
            )

        logging.info(f"CATALOG RETRIEVER | Retriever.milvus_from_csv() | Text embeddings obtained.")   

        # Embed the image field of each row
        image_embs = self.image_embeddings(df["image"].tolist(), verbose=verbose)

        # Filter out failed embeddings and their corresponding metadata
        successful_images_data = [
            (img_ref, emb, meta) for img_ref, emb, meta in zip(df["image"].tolist(), image_embs, metadatas) if emb is not None
        ]
        if successful_images_data:
            successful_images, successful_image_embs, successful_image_metadatas = zip(*successful_images_data)
            self.image_db.add_embeddings(
                texts=list(successful_images),
                embeddings=list(successful_image_embs),
                metadatas=list(successful_image_metadatas)
            )

        logging.info(f"CATALOG RETRIEVER | Retriever.milvus_from_csv() | Image embeddings obtained.") 

    async def retrieve(
        self,
        query: List[str],
        categories: List[str],
        image: str = "",
        k: int = 4,
        image_bool: bool = False,
        verbose: bool = True
    ) -> Tuple[List[str], List[str], List[float], List[str], List[str]]:
        """
        Asynchronously retrieve relevant items from both text and image databases.
        """

        # Check if our query is blank. If it is, replace it with dummy text.
        local_queries = query
        if not query:
            local_queries = ["Can you find me something like this image?"]

        if image_bool:
            if verbose:
                logging.info("CATALOG RETRIEVER | retrieve() | Performing dual retrieval for image input.")

            # Use asyncio.gather for concurrency
            t2t_tasks = []
            for local_query in local_queries:
                if verbose:
                    logging.info(f"\t| retrieve() | Checking query: {local_query}.")
                t2t_tasks.append(asyncio.to_thread(self.text_db.similarity_search_with_relevance_scores, local_query, k=k))
            if verbose:
                logging.info("CATALOG RETRIEVER | retrieve() | Started text task.")
            base64_string = image.replace("data:application/octet-stream", "data:image/jpeg")
            if verbose:
                logging.info(f"CATALOG RETRIEVER | retrieve() | Starting image task...\n\t| {base64_string[:100]}")
            if verbose:
                logging.info(f"CATALOG RETRIEVER | retrieve() | Obtained embedding...")
            i2i_task = asyncio.to_thread(self.image_db.similarity_search_with_relevance_scores, base64_string, k=k*len(query))

            unformatted_results = await asyncio.gather(*t2t_tasks, i2i_task)
        else:
            if verbose:
                logging.info(f"CATALOG RETRIEVER | retrieve() | Text-only retrieval. Queries: {local_queries}")

            results  = []
            for local_query in local_queries:
                if verbose:
                    logging.info(f"\t| retrieve() | Launching text-only retrieval. Query type: {type(local_query)}, Query: {local_query}")
                results.append(asyncio.to_thread(self.text_db.similarity_search_with_relevance_scores, local_query, k=k*len(query)))
            unformatted_results = await asyncio.gather(*results)

        sorted_unformatted_results = []
        for query_results in unformatted_results:
            # Sort each list of (Document, score) tuples by the score in descending order
            sorted_query_results = sorted(query_results, key=lambda item: item[1], reverse=True)
            sorted_unformatted_results.append(sorted_query_results)

        if verbose:
            logging.info(f"""CATALOG RETRIEVER | retrieve() | Pre-interleaving data
                            \n\t| Similarities: {[res[1] for sublist in sorted_unformatted_results for res in sublist]}
                            \n\t| Names: {[res[0].metadata['name'] for sublist in sorted_unformatted_results for res in sublist]}""")

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
                
        # Deduplicate.
        seen_ids = set()
        final_results = [] 
        for res in interleaved_results:
            pk_value = res[0].metadata.get("pk") 
            id_ = str(pk_value) if pk_value is not None else None 
            if id_ is not None and id_ not in seen_ids:
                seen_ids.add(id_)
                final_results.append(res)
        
        all_results = final_results

        if verbose:
            logging.info(f"""CATALOG RETRIEVER | retrieve() | All retrieved results length. {len(all_results)}
                            \n\t| Similarities: {[res[1] for res in all_results]}
                            \n\t| Names: {[res[0].metadata['name'] for res in all_results]}""")

        final_texts = [res[0].page_content+f"\nPRICE: {res[0].metadata['price']}" for res in all_results]
        final_ids = [str(res[0].metadata["pk"]) for res in all_results]
        final_sims = [res[1] for res in all_results]
        final_names = [res[0].metadata['name'] for res in all_results]
        final_images = [res[0].metadata['image'] for res in all_results]

        # Filter by threshold and top k
        final_texts = [text for text, sim in zip(final_texts[:k], final_sims[:k]) if sim > self.sim_threshold]
        final_ids = [id_ for id_, sim in zip(final_ids[:k], final_sims[:k]) if sim > self.sim_threshold]
        final_names = [id_ for id_, sim in zip(final_names[:k], final_sims[:k]) if sim > self.sim_threshold]
        final_images = [id_ for id_, sim in zip(final_images[:k], final_sims[:k]) if sim > self.sim_threshold]
        final_sims = [sim for sim in final_sims[:k] if sim > self.sim_threshold]

        #zipped_sorted = list(zip(final_sims, final_texts, final_ids, final_names, final_images))
        #zipped_sorted = sorted(zipped, key=lambda x: x[0], reverse=True)
        #final_sims, final_texts, final_ids, final_names, final_images = zip(*zipped_sorted)
        zip(final_sims, final_texts, final_ids, final_names, final_images)

        if verbose:
            logging.info(f"CATALOG RETRIEVER | retrieve() | \n\tfinal sims before truncating: {final_sims}")

        final_sims = list(final_sims)[:k]
        final_texts = list(final_texts)[:k]
        final_ids = list(final_ids)[:k]
        final_names = list(final_names)[:k]
        final_images = list(final_images)[:k]

        if verbose:
            logging.info(f"CATALOG RETRIEVER | retrieve() | \n\tnames: {final_names} \n\tsimilarities: {final_sims}")

        cat_list = [text.split("|")[-1].strip().split(",")[1].strip() for text in final_texts]

        if verbose:
            logging.info(f"CATALOG RETRIEVER | pre-category filtering:\n\tCategories: {cat_list}\n\tUser input: {categories}")

        if not categories:
            if verbose:
                logging.info("CATALOG RETRIEVER | No categories provided, returning empty.")
            return [], [], [], [], []

        # Filter by category
        filtered = [
            (text, id_, sim, name, img)
            for text, id_, sim, name, img, cat in zip(final_texts, 
                                           final_ids, 
                                           final_sims, 
                                           final_names, 
                                           final_images, 
                                           cat_list)
            if any(c in cat for c in categories)
        ]

        if not filtered:
            if verbose:
                logging.info("CATALOG RETRIEVER | No matches after category filtering.")
            return [], [], [], [], []

        texts_out, ids_out, sims_out, names_out, images_out = zip(*filtered)
        if verbose:
            logging.info(f"CATALOG RETRIEVER | length of output items: {len(names_out)}")
        return list(texts_out), list(ids_out), list(sims_out), list(names_out), list(images_out)
