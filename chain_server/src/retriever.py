# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
        entities, categories = await self._get_categories(query, state)
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
                    products.append(text)
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

    async def _get_categories(self, query: str, state: State) -> Tuple[List[str],List[str]]:
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
            # Split the query into user question and context for clarity
            user_question = state.query
            conversation_context = state.context
            
            entity_messages = [
                {"role": "system", "content": """You are a search entity extractor. Your task is to identify the specific product the user is asking about based on the conversation history.

    CRITICAL RULES:
    1.  **Analyze Intent:** Determine if the user's "Current question" is a follow-up about a previously discussed product or a request for a new product.
    2.  **Follow-up Clues:** Questions about attributes (e.g., "other colors", "different sizes") or using pronouns (e.g., "it", "that", "those") strongly suggest a follow-up.
    3.  **For Follow-ups, Use Context:** If the question is a follow-up, you MUST extract the full, specific product name from the "Previous conversation context".
    4.  **For New Searches, Use Query:** If the user is asking for a new type of item, you MUST extract the search term directly from the "Current question".
    5.  **Strict Separation:** Never merge or combine terms from the context with terms from the current query.

    **Decision Logic:**

    -   **IF** the `Current question` refers to an existing item (e.g., "does it come in blue?")
        **AND** the `Previous conversation context` contains a specific `[Product Name]`,
        **THEN** you must extract that `[Product Name]`.

    -   **IF** the `Current question` introduces a new item (e.g., "show me some chairs"),
        **THEN** you must extract `chairs`.

    Your goal is to use the context to understand *references*, not to interfere with *new searches*.
    """},
                                {"role": "user", "content": f"""Current question: {user_question}

                Previous conversation context: {conversation_context}

                Apply the decision logic. What is the user searching for?"""}
            ]

            entity_response = asyncio.to_thread(self.model.chat.completions.create, 
                                                model=self.llm_name,
                                                messages=entity_messages,
                                                tools=[search_function],
                                                tool_choice="auto",
                                                temperature=0.0
                                                )
            category_response = asyncio.to_thread(self.model.chat.completions.create, 
                                                model=self.llm_name,
                                                messages=category_messages,
                                                tools=[category_function],
                                                tool_choice="auto",
                                                temperature=0.0
                                                )
            entity_gather, category_gather = await asyncio.gather(entity_response,category_response) 

            logging.info(f"RetrieverAgent | _get_categories()\n\t| Entity Response: {entity_gather}\n\t| Category Response: {category_gather}")
            
            # Add debug logging to see what query was sent
            logging.info(f"RetrieverAgent | _get_categories() | Query sent to entity extractor: {query[:200]}...")

            entities = [query]
            categories = category_list
            if entity_gather.choices[0].message.tool_calls:
                response_dict = json.loads(entity_gather.choices[0].message.tool_calls[0].function.arguments)
                entity_list = response_dict.get("search_entities", [])
                if type(entity_list) == str: 
                    logging.info(f"RetrieverAgent | _get_categories()\n\t| Entity list {entity_list}")
                    cleaned = entity_list.strip("[]")
                    entities = [item.strip().strip("'\"") for item in cleaned.split(',')]
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
                        logging.info(f"RetrieverAgent | _get_categories()\n\t| Category list {category_list}")
                        cleaned = category_list.strip("[]")
                        categories = [item.strip().strip("'\"") for item in cleaned.split(',')]
                    else:
                        categories = category_list

            logging.info(f"RetrieverAgent | _get_categories() | entities: {entities}\n\t| categories: {categories}")
            return entities, categories
        else:
            logging.info(f"RetrieverAgent | _get_categories() | No valid query.")
            return entity_list, category_list
