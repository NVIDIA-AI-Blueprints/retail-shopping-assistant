# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
RetrieverAgent is an agent which retrieves relevant products based on user queries.
It extracts structured retrieval inputs (entities, categories, filters) and then queries
the catalog retriever service to find relevant products.
"""

from .agenttypes import State
from .functions import retrieval_extraction_function, parse_tool_call_fallback
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
            has_image = bool(state.image)

            base_system_prompt = """You are a retrieval input extractor. Identify what the user wants to BROWSE or FIND in the catalog and translate it into structured search inputs.

GUIDING PRINCIPLE:
`search_entities` describes WHAT THE USER WANTS THE CATALOG TO RETURN. It is never the item the user is merely referencing for comparison, styling, or context.

DECISION LOGIC (apply in order, stop at the first match):

1. ATTRIBUTE FOLLOW-UP about a specific named product.
   Triggers: the question asks about properties of an item already under discussion ("does it come in blue?", "how do I wash it?", "what sizes does it come in?", "how much does it cost?") AND does NOT introduce any new product type or category.
   -> Echo the full `[Product Name]` from the previous conversation context. These names are illustrative only; always use the exact name that appears in the context.
   -> Examples (abstract names spanning different categories):
      Context discusses "Midnight Velvet Blazer"; user asks "does it come in blue?"   -> search_entities: ["Midnight Velvet Blazer"]
      Context discusses "Alpine Waterproof Hiking Boot"; user asks "what sizes does it come in?" -> search_entities: ["Alpine Waterproof Hiking Boot"]
      Context discusses "Pearl Drop Stud Earrings"; user asks "how much are they?" -> search_entities: ["Pearl Drop Stud Earrings"]

2. NEW PRODUCT TYPE (with or without a reference to another item).
   Triggers: the question names a product type/category the user wants to see, EVEN IF it references a previously discussed item (via "it", "this", "that dress", "my cart", etc.) for styling or compatibility context.
   Phrases like "...that go with...", "...to match...", "...to pair with...", "...for this outfit..." are ALWAYS new-product-type queries.
   -> Extract the NEW product type(s) from the current question, not the referenced item.
   -> Examples:
      "show me some hats"                       -> search_entities: ["hats"]
      "what shoes go well with it?"             -> search_entities: ["shoes"]
      "any earrings that match this dress?"     -> search_entities: ["earrings"]
      "do you have a bag for this outfit?"      -> search_entities: ["bag"]

3. OPEN-ENDED BROWSE (no specific referent).
   Triggers: the question includes a product type, category, occasion, style, outfit goal, or image referent that can anchor catalog retrieval.
   -> Extract the product target(s) verbatim from the current question.
   -> Examples:
      "I need a summer top"                  -> search_entities: ["summer top"]
      "I need something for a party"         -> search_entities: ["party outfit"]
   -> Do NOT use generic browse words such as "anything", "everything", "something", "items", "products", or "stuff" as search_entities when they are the only catalog target. For constraint-only requests like "show me anything under $100" or "what do you have on sale", return an empty search_entities list and preserve any explicit filters.

CATEGORIES:
- Choose up to three from the provided Available categories list ONLY.
- Do NOT invent generic labels like "apparel" or "clothing".
- You may reuse the same category when only one is relevant.

FILTERS:
- Return `min_price` / `max_price` ONLY when the user explicitly states a budget ("under $50", "between 20 and 100 dollars").
- NEVER default to 0. If no price is mentioned, OMIT the field entirely.
- Return numeric values without currency symbols.

STRICT SEPARATION:
- Never merge terms from the context with terms from the current question.
- Never combine a referenced product's name with a new product type in the same entity.
"""

            # When an image is attached, the image is the semantic referent for
            # any deictic phrase. We must not force the LLM to invent a product
            # name for the image; an empty `search_entities` is correct and the
            # downstream image-search path uses the image embedding as the
            # retrieval signal. Filters and explicit new-product-types still
            # apply — the examples below make that explicit because a prior
            # iteration of this rule caused the model to drop the price filter
            # when it saw "return EMPTY search_entities".
            image_rule = """
IMAGE ATTACHED TO THIS TURN:
- The user has uploaded an image. Treat the image as the product being referenced.
- Phrases like "this", "this product", "this item", "these", "it", "one like this", or no product noun at all refer to the IMAGE, not to any previously discussed named item.
- Do NOT invent a product name from the image, and do NOT echo a name from the prior context just because of a pronoun.
- DECISION LOGIC step 1 (ATTRIBUTE FOLLOW-UP echoing a name from context) DOES NOT APPLY when an image is attached.

Extraction rules WITH an image:
A. Filter-only refinement of the image ("do you have this under $100", "anything like this in blue", "is this on sale"):
   -> search_entities: []  (the image carries the semantic intent)
   -> STILL extract `min_price` / `max_price` from any explicit budget words ("under", "below", "less than", "no more than", "between X and Y"). Dropping the budget is a bug; always emit it when it appears.
   -> Pick one matching category from the allowed list only if the text makes it obvious; otherwise repeat the first category slot.
B. New product type alongside the image ("a bag that goes with this", "shoes like these"):
   -> Apply DECISION LOGIC step 2: extract the NEW product type into search_entities.
   -> STILL emit price filters if the user gave a budget.

Worked examples (image always attached):
  User: "do you have this product under $100"
    -> search_entities: []
    -> max_price: 100
  User: "anything like this between 50 and 80 dollars"
    -> search_entities: []
    -> min_price: 50
    -> max_price: 80
  User: "a bag that goes with this under $200"
    -> search_entities: ["bag"]
    -> max_price: 200
  User: "show me the red one"
    -> search_entities: []
    (no price mentioned, no price filter)
"""

            system_prompt = base_system_prompt + image_rule if has_image else base_system_prompt

            user_content = (
                f"Current question: {user_question}\n\n"
                f"Previous conversation context: {conversation_context}\n"
                f"Available categories: {category_list_str}\n"
                f"Image attached this turn: {'yes' if has_image else 'no'}\n\n"
                "Apply the decision logic and extract retrieval inputs."
            )

            extraction_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]

            extraction_response = await asyncio.to_thread(
                self.model.chat.completions.create,
                model=self.llm_name,
                messages=extraction_messages,
                tools=[retrieval_extraction_function],
                tool_choice="auto",
                temperature=0.0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}}
            )

            logging.info(
                "RetrieverAgent | _extract_retrieval_inputs()\n"
                f"\t| Combined Extraction Response: {extraction_response}"
            )
            
            # Add debug logging to see what query was sent
            logging.info(f"RetrieverAgent | _extract_retrieval_inputs() | Query sent to retrieval extractor: {user_question[:200]}...")

            message = extraction_response.choices[0].message
            response_dict = None
            if message.tool_calls:
                response_dict = json.loads(message.tool_calls[0].function.arguments)
            elif message.content:
                logging.warning("RetrieverAgent | _extract_retrieval_inputs() | No structured tool_calls returned, attempting fallback parse.")
                tool_name, tool_args = parse_tool_call_fallback(message.content)
                if tool_name == "extract_retrieval_inputs" and tool_args:
                    response_dict = tool_args

            if response_dict:
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

                # Drop invented categories; fall back to the full allowlist
                # so we never query with an empty category filter.
                sanitized_categories = self._sanitize_categories(categories)
                categories = sanitized_categories if sanitized_categories else self.categories

                filters = self._normalize_filters(response_dict)

            # Drop blank/whitespace-only entities regardless of how they got
            # there. The text-embedding endpoint 400s on empty input, so this
            # keeps us safe even if the extractor returns `[""]` via a quirky
            # tool-call serialization.
            entities = [
                e for e in entities
                if isinstance(e, str) and e.strip()
            ]

            # When an image is attached, the LLM is instructed to return empty
            # entities for filter-only refinements ("under $100"). The catalog
            # retriever's dual text+image path still needs at least one text
            # entry to keep the text branch alive and to size the image
            # search's k-multiplier. Fall back to the raw query text only for
            # that wiring; the image itself remains the primary semantic
            # signal on the image DB side.
            if has_image and not entities:
                entities = [user_question]

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
        """Normalize extracted filters to a minimal numeric contract.

        LLMs frequently fill optional price parameters with ``0`` when the
        user did not mention a budget. Treat any non-positive price as
        unspecified so ``max_price=0`` can't silently zero out results.
        """
        normalized: Dict[str, float] = {}
        min_price = self._normalize_numeric_filter(raw_filters.get("min_price"))
        max_price = self._normalize_numeric_filter(raw_filters.get("max_price"))

        if min_price is not None and min_price > 0:
            normalized["min_price"] = min_price
        elif min_price is not None:
            logging.info(
                f"RetrieverAgent | _normalize_filters() | dropping non-positive min_price={min_price}"
            )
        if max_price is not None and max_price > 0:
            normalized["max_price"] = max_price
        elif max_price is not None:
            logging.info(
                f"RetrieverAgent | _normalize_filters() | dropping non-positive max_price={max_price}"
            )

        if "min_price" in normalized and "max_price" in normalized:
            if normalized["min_price"] > normalized["max_price"]:
                logging.warning(
                    "RetrieverAgent | _normalize_filters() | min_price exceeds max_price; "
                    f"dropping both ({normalized['min_price']} > {normalized['max_price']})"
                )
                normalized.pop("min_price")
                normalized.pop("max_price")

        return normalized

    def _sanitize_categories(self, raw_categories: List[str]) -> List[str]:
        """Keep only categories present in the configured allowlist.

        The extractor occasionally invents generic labels ("apparel",
        "clothing") that aren't in the catalog taxonomy; filtering to the
        allowlist keeps retrieval behaviour predictable.
        """
        allowed = {c for c in self.categories}
        seen: set = set()
        sanitized: List[str] = []
        for cat in raw_categories:
            if not isinstance(cat, str):
                continue
            normalized = cat.strip()
            if not normalized:
                continue
            if normalized not in allowed:
                logging.info(
                    f"RetrieverAgent | _sanitize_categories() | dropping unknown category '{normalized}'"
                )
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            sanitized.append(normalized)
        return sanitized
