"""
RetrieverAgent is an agent which retrieves relevant products based on user queries.
It uses a search tool to determine the category of the query and then queries the catalog retriever
service to find relevant products.
"""

from .agenttypes import State
from .functions import search_function, category_function
from openai import OpenAI
import os
import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import sys
from typing import Tuple, List, Dict
import asyncio
import logging
import time
import ast


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    ) 

# Configuration will be loaded by the main application

class RetrieverAgent():
    def __init__(
        self,
        config,
    ) -> None:
        logging.info(f"RetrieverAgent.__init__() | Initializing with llm_name={config.llm_name}, llm_port={config.llm_port}")
        self.llm_name = config.llm_name
        self.llm_port = config.llm_port
        
        # Store configuration
        self.catalog_retriever_url = config.retriever_port
        self.k_value = config.top_k_retrieve
        self.categories = config.categories
        
        self.model = OpenAI(base_url=config.llm_port, api_key=os.environ["LLM_API_KEY"])
        logging.info(f"RetrieverAgent.__init__() | Initialization complete")

    async def invoke(
        self,
        state: State,
        verbose: bool = True
    ) -> State:
        """
        Process the user query to determine categories and retrieve relevant products.
        """
        logging.info(f"RetrieverAgent.invoke() | Starting with query: {state.query}")

        # Set our k value for retrieval.
        k = self.k_value

        # Get the user query and image from the state
        query = f"The user has asked: '{state.query}'. With the following context: '{state.context}'.\n" 
        image = state.image

        # Use the LLM to determine categories for the query
        start = time.monotonic()
        entities, categories = await self._get_categories(query)
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
                logging.info(f"RetrieverAgent.invoke() | /query/image -- getting response.\n\t| entities: {entities}\n\t| categories: {categories}")
                response = session.post(
                    f"{self.catalog_retriever_url}/query/image",
                    json={
                        "text": entities,
                        "image_base64": image,
                        "categories": categories,
                        "k": k
                    }
                )
            else:
                logging.info(f"RetrieverAgent.invoke() | /query/text -- getting response\n\t| query: {entities}\n\t| categories: {categories}")
                response = session.post(
                    f"{self.catalog_retriever_url}/query/text",
                    json={
                        "text": entities,
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
            
            logging.info(f"RetrieverAgent.invoke() | Retriever returned context.")
            
            # Update context
            state.context = f"{state.context}\n{state.response}"
            
        except requests.exceptions.RequestException as e:
            if verbose:
                logging.error(f"RetrieverAgent.invoke() | Error querying catalog retriever service: {str(e)}")
            state.response = "I encountered an error while searching for products. Please try again."
        end = time.monotonic()
        state.timings["retriever_retrieval"] = end - start

        logging.info(f"RetrieverAgent.invoke() | Returning final state with response.")

        return state

    async def _get_categories(self, query: str) -> Tuple[List[str],List[str]]:
        """
        Use the LLM to determine relevant categories for the query using the search function.
        """
        logging.info(f"RetrieverAgent | _get_categories() | Starting with query (first 50 characters): {query[:50]}")
        category_list = self.categories
        entity_list = []

        if query:
            logging.info(f"RetrieverAgent | _get_categories() | Checking for categories.")
            category_list_str = ", ".join(category_list)    
            category_messages = [
                {"role": "user", "content": f"""
                                            \nAVAILABLE CATEGORIES\n '{category_list_str}'
                                            \nPROCESS THIS USER QUERY WITH CONTEXT:\n '{query}'"""}
            ]
            entity_messages = [
                {"role": "user", "content": f"""\nUSER QUERY WITH CONTEXT:\n '{query}'"""}
            ]

            entity_response = asyncio.to_thread(self.model.chat.completions.create, 
                                                model=self.llm_name,
                                                messages=entity_messages,
                                                tools=[search_function],
                                                tool_choice="auto"
                                                )
            category_response = asyncio.to_thread(self.model.chat.completions.create, 
                                                model=self.llm_name,
                                                messages=category_messages,
                                                tools=[category_function],
                                                tool_choice="auto"
                                                )
            entity_gather, category_gather = await asyncio.gather(entity_response,category_response) 

            logging.info(f"RetrieverAgent | _get_categories()\n\t| Entity Response: {entity_gather}\n\t| Category Response: {category_gather}")

            entities = [query]
            categories = category_list
            if entity_gather.choices[0].message.tool_calls:
                response_dict = json.loads(entity_gather.choices[0].message.tool_calls[0].function.arguments)
                entity_list = response_dict.get("search_entities", [])
                if type(entity_list) == str: 
                    entities = ast.literal_eval(entity_list)
                else:
                    entities = entity_list
                if category_gather.choices[0].message.tool_calls:
                    response_dict = json.loads(category_gather.choices[0].message.tool_calls[0].function.arguments)
                    category_list = [
                        response_dict.get("category_one", ""),
                        response_dict.get("category_two", ""),
                        response_dict.get("category_three", ""),
                        ]
                    if type(category_list) == str: 
                        categories = ast.literal_eval(category_list)
                    else:
                        categories = category_list

            logging.info(f"RetrieverAgent | _get_categories() | entities: {entities}\n\t| categories: {categories}")
            return entities, categories
        else:
            logging.info(f"RetrieverAgent | _get_categories() | No valid query.")
            return entity_list, category_list
