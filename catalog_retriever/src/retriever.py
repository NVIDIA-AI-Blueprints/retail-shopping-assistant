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
from numpy import mean
from .utils import image_url_to_base64, is_url
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
        return self.retriever.embed_chunk(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate text embeddings for multiple texts"""
        logging.info(f"TextEmbeddings | embed_documents() | called.")
        return self.retriever.text_embeddings(texts)

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
            drop_old=True,
        )
        self.image_db = Milvus(
            embedding_function=self.image_embeddings_obj,
            collection_name=self.image_collection,
            connection_args={"uri": f"{self.db_port}"},
            auto_id=True,
            drop_old=True,
        )

        logging.info(f"CATALOG RETRIEVER | Retriever.__init__() | Milvus collections initialized.")

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
        ) -> Tuple[List[str],List[List[float]]]:
        """
        Generate text embeddings from a list of text strings.
        """
        out_texts = []
        embeddings = []
        for text in texts:
            try:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=200
                )
                chunks = text_splitter.split_text(text)
                chunk_embeddings = []
                for chunk in chunks:
                    embedding = self.embed_chunk(chunk, query_type)
                    chunk_embeddings.append(embedding)
                average_embedding = list(mean(chunk_embeddings, axis=0))
                out_texts.append(text)
                embeddings.append(average_embedding)
                if verbose:
                    logging.info(f"CATALOG RETRIEVER | Retriever.text_embeddings() | Added (first 50 characters): {text[:50]}")
            except Exception as e:
                if verbose:
                    logging.info(f"CATALOG RETRIEVER | Retriever.text_embeddings() |\n\t| Port: {self.text_embed_port} |\n\t| Error: {e}.")

        return out_texts, embeddings

    def image_embeddings(
        self, 
        texts: List[str],
        verbose: bool = False
    ) -> Tuple[List[str],List[List[float]]]:
        """
        Generate image embeddings from a list of base64 image strings or image URLs.
        If input is a URL, automatically download and convert to base64. 
        """
        if verbose:
            logging.info(f"CATALOG RETRIEVER | Retriever.image_embeddings() | Texts: {texts}")
        out_images = []
        embeddings = []
        for text in texts:
            try:
                # Auto-detect if input is a URL. If it is, embed it as such, otherwise continue assuming b64.
                if is_url(text):
                    if verbose:
                        logging.info(f"CATALOG RETRIEVER | Retriever.image_embeddings() | Detected URL, converting to base64 (First 50 chars): {text[:50]}")
                    input_data = image_url_to_base64(text)  
                else:
                    input_data = text 

                # Check that the input data is small enough. 
                MAX_VARCHAR_LENGTH = 65535
                if len(input_data) <= MAX_VARCHAR_LENGTH:
                    response = self.image_client.embeddings.create(
                        input=input_data,
                        model=self.image_model_name,
                        encoding_format="float",
                    )
                    logging.info(f"CATALOG RETRIEVER | Retriever.image_embeddings() | Obtained embedding.")
                    out_images.append(input_data)
                    logging.info(f"CATALOG RETRIEVER | Retriever.image_embeddings() | Accessing embedding.")
                    embeddings.append(response.data[0].embedding)
                    logging.info(f"CATALOG RETRIEVER | Retriever.image_embeddings() | Embedding accessed.\n\t| {response.data[0].embedding[:10]}")
                else:
                    logging.info(f"CATALOG RETRIEVER | Retriever.image_embeddings() | Encoding too large ({len(input_data)}). skipping embedding.")

                if verbose:
                    logging.info(f"CATALOG RETRIEVER | Retriever.image_embeddings() | Added (first 50 characters): {input_data[:50]}")

            except Exception as e:
                if verbose:
                    logging.info(f"CATALOG RETRIEVER | Retriever.image_embeddings() | Error: {e}")

        return out_images, embeddings



    def milvus_from_csv(self, csv_path: str, verbose: bool = False) -> None:
        """
        Fills the milvus database with the data from a CSV file.
        """ 

        logging.info(f"CATALOG RETRIEVER | Retriever.milvus_from_csv() | Reading in csv from: '{csv_path}'")

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
        new_texts,text_embs = self.text_embeddings(combined_texts,query_type="passage",verbose=verbose)
        self.text_db.add_embeddings(
            texts=new_texts,
            embeddings=text_embs,
            metadatas=metadatas
        )

        logging.info(f"CATALOG RETRIEVER | Retriever.milvus_from_csv() | Text embeddings obtained.")   

        # Embed the image field of each row
        new_images,image_embs = self.image_embeddings(df["image"].tolist(), verbose=verbose)
        self.image_db.add_embeddings(
            texts=new_images,
            embeddings=image_embs,
            metadatas=metadatas
        )

        logging.info(f"CATALOG RETRIEVER | Retriever.milvus_from_csv() | Image embeddings obtained.") 

    def print_database(self) -> None:
        """
        Print the contents of both text and image databases.
        """
        print("\nText Database Contents:")
        print("----------------------")
        docs = self.text_db.similarity_search("items", k=1000)
        for i, doc in enumerate(docs, 1):
            print(f"{i}. {doc.page_content[:100]}...")  

        print("\nImage Database Contents:")
        print("----------------------")
        try:
            # Get all documents from the image database without similarity search
            docs = self.image_db.get_all_documents()
            for i, doc in enumerate(docs, 1):
                print(f"{i}. {doc.page_content[:100]}...")
        except Exception as e:
            print(f"Error accessing image database: {str(e)}")
            print("Image database might be empty or not properly initialized.")

    async def retrieve(
        self,
        query: str,
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
        local_query = query
        if not query:
            local_query = "Can you find me something like this image?"

        if image_bool:
            if verbose:
                logging.info("CATALOG RETRIEVER | retrieve() | Performing dual retrieval for image input.")

            # Use asyncio.gather for concurrency
            t2t_task = asyncio.to_thread(self.text_db.similarity_search_with_relevance_scores, local_query, k=k)
            if verbose:
                logging.info("CATALOG RETRIEVER | retrieve() | Started text task.")
            base64_string = image.replace("data:application/octet-stream", "data:image/jpeg")
            if verbose:
                logging.info(f"CATALOG RETRIEVER | retrieve() | Starting image task...\n\t| {base64_string[:100]}")
            if verbose:
                logging.info(f"CATALOG RETRIEVER | retrieve() | Obtained embedding...")
            i2i_task = asyncio.to_thread(self.image_db.similarity_search_with_relevance_scores, base64_string, k=k)

            t2t_results, i2i_results = await asyncio.gather(t2t_task, i2i_task)
            temp_results = t2t_results + i2i_results

            if verbose:
                logging.info("CATALOG RETRIEVER | retrieve() | Gathered tasks.")

            # Deduplicate.
            seen_ids = set()
            all_results = []
            for res in temp_results:
                id_ = str(res[0].metadata["pk"])
                if id_ not in seen_ids:
                    seen_ids.add(id_)
                    all_results.append(res)
        else:
            if verbose:
                logging.info(f"CATALOG RETRIEVER | retrieve() | Text-only retrieval. Query: {local_query}")

            all_results = await asyncio.to_thread(self.text_db.similarity_search_with_relevance_scores, local_query, k=k)

        final_texts = [res[0].page_content for res in all_results]
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

        zipped = list(zip(final_sims, final_texts, final_ids, final_names, final_images))
        zipped_sorted = sorted(zipped, key=lambda x: x[0], reverse=True)
        final_sims, final_texts, final_ids, final_names, final_images = zip(*zipped_sorted)

        final_sims = list(final_sims)
        final_texts = list(final_texts)
        final_ids = list(final_ids)
        final_names = list(final_names)
        final_images = list(final_images)

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
        return list(texts_out), list(ids_out), list(sims_out), list(names_out), list(images_out)

if __name__ == "__main__":
    config = RetrieverConfig(  
        text_embed_port="https://integrate.api.nvidia.com/v1",
        image_embed_port="https://integrate.api.nvidia.com/v1",
        text_model_name="nvdev/nvidia/nv-embedqa-e5-v5",
        image_model_name="nvdev/nvidia/nvclip",
        db_port="http://localhost:19530", 
        db_name="shopping_advisor_db",
        sim_threshold=0.5,
        text_collection="shopping_advisor_text_db",
        image_collection="shopping_advisor_image_db"
    )
    retriever = Retriever(config=config)
    retriever.milvus_from_csv(csv_path="data/products.csv", verbose=True)
    retriever.print_database()
    texts,ids,sims = retriever.retrieve(query="skirt", categories=["apparel","dress"], k=4, verbose=True)
