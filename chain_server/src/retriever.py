# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
RetrieverAgent is an agent which retrieves relevant products based on user queries.
It extracts structured retrieval inputs (entities, categories, filters) and then queries
the catalog retriever service to find relevant products.
"""

from .agenttypes import State
from .functions import retrieval_extraction_function
from openai import OpenAI
import os
import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import sys
from typing import Tuple, List, Dict, Any
import asyncio
import logging
import time


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
        image = state.image

        # Use the LLM to determine entities/categories/filters for retrieval
        start = time.monotonic()
        entities, categories, filters = await self._extract_retrieval_inputs(state)
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
                logging.info(
                    "RetrieverAgent.invoke() | /query/image -- getting response.\n"
                    f"\t| entities: {entities}\n"
                    f"\t| categories: {categories}\n"
                    f"\t| filters: {filters}"
                )
                response = session.post(
                    f"{self.catalog_retriever_url}/query/image",
                    json={
                        "text": entities,
                        "image_base64": image,
                        "categories": categories,
                        "filters": filters,
                        "k": k
                    }
                )
            else:
                logging.info(
                    "RetrieverAgent.invoke() | /query/text -- getting response\n"
                    f"\t| query: {entities}\n"
                    f"\t| categories: {categories}\n"
                    f"\t| filters: {filters}"
                )
                response = session.post(
                    f"{self.catalog_retriever_url}/query/text",
                    json={
                        "text": entities,
                        "categories": categories,
                        "filters": filters,
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

    async def _extract_retrieval_inputs(self, state: State) -> Tuple[List[str], List[str], Dict[str, float]]:
        """
        Extract retrieval entities, categories, and structured filters from the user request.
        """
        query_text = state.query or ""
        logging.info(f"RetrieverAgent | _extract_retrieval_inputs() | Starting with query (first 50 characters): {query_text[:50]}")
        category_list = self.categories
        entity_list = []
        filters: Dict[str, float] = {}
        entities: List[str] = [query_text] if query_text else []
        categories = category_list

        if query_text:
            logging.info("RetrieverAgent | _extract_retrieval_inputs() | Extracting retrieval inputs.")
            category_list_str = ", ".join(category_list)
            # Split the query into user question and context for clarity
            user_question = query_text
            conversation_context = state.context
            
            extraction_messages = [
                {"role": "system", "content": """You are a retrieval input extractor. Your task is to identify the specific product the user is asking about based on the conversation history.

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

    -   **IF** the `Current question` introduces a new item (e.g., "show me some hats"),
        **THEN** you must extract `hats`.

    -   For categories, only choose from the provided available categories.
        You may reuse the same category if only one is relevant.

    -   For filters, return only explicit constraints.
        If price bounds are present, return numeric values without currency symbols.

    Your goal is to use the context to understand *references*, not to interfere with *new searches*.
    """},
                {"role": "user", "content": f"""Current question: {user_question}

Previous conversation context: {conversation_context}
Available categories: {category_list_str}

Apply the decision logic and extract retrieval inputs."""}
            ]

            extraction_response = await asyncio.to_thread(
                self.model.chat.completions.create,
                model=self.llm_name,
                messages=extraction_messages,
                tools=[retrieval_extraction_function],
                tool_choice="auto",
                temperature=0.0
            )

            logging.info(
                "RetrieverAgent | _extract_retrieval_inputs()\n"
                f"\t| Combined Extraction Response: {extraction_response}"
            )
            
            # Add debug logging to see what query was sent
            logging.info(f"RetrieverAgent | _extract_retrieval_inputs() | Query sent to retrieval extractor: {user_question[:200]}...")

            if extraction_response.choices[0].message.tool_calls:
                response_dict = json.loads(extraction_response.choices[0].message.tool_calls[0].function.arguments)
                entity_list = response_dict.get("search_entities", [])
                if isinstance(entity_list, str):
                    logging.info(f"RetrieverAgent | _extract_retrieval_inputs()\n\t| Entity list {entity_list}")
                    cleaned = entity_list.strip("[]")
                    entities = [item.strip().strip("'\"") for item in cleaned.split(',')]
                else:
                    entities = entity_list
                category_list = [
                    response_dict.get("category_one", ""),
                    response_dict.get("category_two", ""),
                    response_dict.get("category_three", ""),
                    ]
                if isinstance(category_list, str):
                    logging.info(f"RetrieverAgent | _extract_retrieval_inputs()\n\t| Category list {category_list}")
                    cleaned = category_list.strip("[]")
                    categories = [item.strip().strip("'\"") for item in cleaned.split(',')]
                else:
                    categories = category_list

                filters = self._normalize_filters(response_dict)

            logging.info(
                "RetrieverAgent | _extract_retrieval_inputs() | "
                f"entities: {entities}\n\t| categories: {categories}\n\t| filters: {filters}"
            )
            return entities, categories, filters
        else:
            logging.info("RetrieverAgent | _extract_retrieval_inputs() | No valid query.")
            return entity_list, categories, filters

    @staticmethod
    def _normalize_numeric_filter(value: Any) -> float | None:
        """Convert potentially string-based numeric filters into floats."""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            cleaned = value.strip().replace("$", "").replace(",", "")
            try:
                return float(cleaned)
            except ValueError:
                return None
        return None

    def _normalize_filters(self, raw_filters: Dict[str, Any]) -> Dict[str, float]:
        """Normalize extracted filters to a minimal numeric contract."""
        normalized: Dict[str, float] = {}
        min_price = self._normalize_numeric_filter(raw_filters.get("min_price"))
        max_price = self._normalize_numeric_filter(raw_filters.get("max_price"))

        if min_price is not None:
            normalized["min_price"] = min_price
        if max_price is not None:
            normalized["max_price"] = max_price

        return normalized
