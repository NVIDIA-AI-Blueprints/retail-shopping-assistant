"""
RetrieverAgent is an agent which retrieves relevant products based on user queries.
It uses a search tool to determine the category of the query and then queries the catalog retriever
service to find relevant products.
"""

from .agenttypes import State
from .functions import search_function
from openai import OpenAI
import os
import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import yaml
import sys
from typing import List, Dict, Any
import logging
import time
import ast

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    ) 

config_path = os.path.join("/app", "app", "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

CATALOG_RETRIEVER_URL = config["retriever_port"]
K_VALUE = config["top_k_retrieve"]
CATEGORIES = config["categories"]

class RetrieverAgent():
    def __init__(
        self,
        llm_name: str,
        llm_port: str,
    ) -> None:
        logging.info(f"RetrieverAgent.__init__() | Initializing with llm_name={llm_name}, llm_port={llm_port}")
        self.llm_name = llm_name
        self.llm_port = llm_port
        self.catalog_retriever_url = CATALOG_RETRIEVER_URL
        self.model = OpenAI(base_url=llm_port, api_key=os.environ["LLM_API_KEY"])
        logging.info(f"RetrieverAgent.__init__() | Initialization complete")

    def invoke(
        self,
        state: State,
        verbose: bool = True
    ) -> State:
        """
        Process the user query to determine categories and retrieve relevant products.
        """
        logging.info(f"RetrieverAgent.invoke() | Starting with query: {state.query}")

        # Set our k value for retrieval.
        k = K_VALUE

        # Get the user query and image from the state
        query = state.query
        image = state.image

        # Use the LLM to determine categories for the query
        start = time.monotonic()
        categories = self._get_categories(query)
        end = time.monotonic()
        state.timings["retriever_categories"] = end - start
        
        # Query the catalog retriever service
        start = time.monotonic()
        try:

            retry_strategy = Retry(
                total=3,                    
                status_forcelist=[422, 429, 500, 502, 503, 504],  
                allowed_methods=["POST"],   
                backoff_factor=1            
            )
            
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session = requests.Session()
            session.mount("https://", adapter)
            session.mount("http://", adapter)

            if image:
                logging.info(f"RetrieverAgent.invoke() | /query/image -- getting response.\n\t| query: {query}\n\t| categories: {categories}")
                response = session.post(
                    f"{self.catalog_retriever_url}/query/image",
                    json={
                        "text": query,
                        "image_base64": image,
                        "categories": categories,
                        "k": k
                    }
                )
            else:
                logging.info(f"RetrieverAgent.invoke() | /query/text -- getting response\n\t| query: {query}\n\t| categories: {categories}")
                response = session.post(
                    f"{self.catalog_retriever_url}/query/text",
                    json={
                        "text": query,
                        "categories": categories,
                        "k": k
                    }
                )

            response.raise_for_status()
            results = response.json()
            
            # Format the response with product details
            if results["texts"]:
                products = []
                retrieved_dict = {}
                for text, name, img, sim in zip(results["texts"], results["names"], results["images"], results["similarities"]):
                    products.append(f"{text} (similarity: {sim:.2f})")
                    retrieved_dict[name] = img
                state.response = f"These products are available in the catalog:\n" + "\n".join(products)
                state.retrieved = retrieved_dict
            else:
                state.response = "Unfortunately there are no products closely matching the user's query."
            
            logging.info(f"RetrieverAgent.invoke() | Retriever returned context: {state.response}")
            
            # Update context
            state.context = f"{state.context}\n{state.response}"
            
        except requests.exceptions.RequestException as e:
            if verbose:
                logging.error(f"RetrieverAgent.invoke() | Error querying catalog retriever service: {str(e)}")
            state.response = "I encountered an error while searching for products. Please try again."
        end = time.monotonic()
        state.timings["retriever_retrieval"] = end - start

        logging.info(f"RetrieverAgent.invoke() | Returning final state with response: {state.response}")

        return state

    def _get_categories(self, query: str) -> List[str]:
        """
        Use the LLM to determine relevant categories for the query using the search function.
        """
        logging.info(f"RetrieverAgent._get_categories() | Starting with query: {query}")
        category_list = CATEGORIES

        if query:
            logging.info(f"RetrieverAgent._get_categories() | Checking for categories.")
            category_list_str = ", ".join(category_list)    
            messages = [
                #{"role": "system", "content": f"Categories: {category_list_str}"},
                {"role": "user", "content": f"User Query: '{query}'\nCategories: '{category_list_str}'"}
            ]

            response = self.model.chat.completions.create(
                model=self.llm_name,
                messages=messages,
                tools=[search_function],
                tool_choice="auto"
            )

            logging.info(f"RetrieverAgent._get_categories() | Response: {response}")

            # Extract categories from the function call
            if response.choices[0].message.tool_calls:
                category_dict = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
                category_list = category_dict.get("relevant_categories", [])
                if type(category_list) == str: 
                    categories = ast.literal_eval(category_list)
                else:
                    categories = category_list
                return categories
                logging.info(f"RetrieverAgent._get_categories() | Returning empty list")
            return []
        else:
            logging.info(f"RetrieverAgent._get_categories() | No valid query.")
            return category_list
