# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from .agenttypes import Cart, State
from .functions import add_to_cart_function, remove_from_cart_function, view_cart_function
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
import os
import json
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import sys
import time


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )

# Configuration will be loaded by the main application

class CartAgent():
    """
    CartAgent is an agent which manages a user's cart.
    It can perform a few actions:
    - add_to_cart : Adds a product to the cart.
    - remove_from_cart : Removes a product from the cart.
    - view_cart : Reports what is in the cart.
    """
    def __init__(self,
        config,
    ) -> None:
        logging.info(f"CartAgent.__init__() | Initializing with llm_name={config.llm_name}, llm_port={config.llm_port}")
        self.llm_name = config.llm_name
        self.llm_port = config.llm_port
        
        # Store configuration
        self.memory_retriever_url = config.memory_port
        self.model = OpenAI(base_url=config.llm_port, api_key=os.environ["LLM_API_KEY"])
        self.catalog_retriever_port = config.retriever_port
        self.categories = config.categories
        self.retry_strategy = Retry(
                total=3,                    
                status_forcelist=[422, 429, 500, 502, 503, 504],  
                allowed_methods=["POST"],   
                backoff_factor=1            
            )
        logging.info(f"CartAgent.__init__() | Initialization complete")
        
    def _get_cart(self, user_id: int) -> Cart:
        response = requests.get(f"{self.memory_retriever_url}/user/{user_id}/cart")
        logging.info(f"CartAgent._get_cart() | Response text: {response.text}.")
        if response.status_code == 200:
            cart_data = json.loads(response.text)["cart"]
            return Cart(contents=cart_data)
        return Cart(contents=[])

    def _add_to_cart(self, user_id: int, item_name: str, quantity: int) -> str:
        # First we have to perfom a retrieval to ensure that the item being looked for is in the catalog.
        adapter = HTTPAdapter(max_retries=self.retry_strategy)
        session = requests.Session()
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        logging.info(f"CartAgent.add_to_cart() | /query/text -- getting response\n\t| query: {item_name}\n\t")
        ret_response = session.post(
            f"{self.catalog_retriever_port}/query/text",
            json={
                "text": [item_name],
                "categories": self.categories,
                "k": 1
            }
        )
        ret_response.raise_for_status()
        res_json = ret_response.json()
        sim = 0
        if res_json["similarities"]:
            sim = res_json["similarities"][0]
            if sim > 0.8:
                catalog_item_name = res_json["names"][0]
                logging.info(f"CartAgent.add_to_cart() | input name: {item_name}, retrieved item: {catalog_item_name}, sim: {sim}")
                response = requests.post(
                    f"{self.memory_retriever_url}/user/{user_id}/cart/add",
                    json={"item": catalog_item_name, "amount": quantity}
                )
                if response.status_code == 200:
                    return response.json()["message"]
                return f"Failed to add {quantity} {catalog_item_name} to cart."
            else:
                logging.info(f"CartAgent.remove_from_cart() | Nothing sufficiently similar to {item_name} in the cart.")
                return f"No such item ({item_name}) could be found in the catalog."
        else:
            return f"No such item ({item_name}) could be found in the catalog."

    def _remove_from_cart(self, user_id: int, item_name: str, quantity: int) -> str:
        # First we have to perfom a retrieval to ensure that the item being looked for is in the catalog.
        adapter = HTTPAdapter(max_retries=self.retry_strategy)
        session = requests.Session()
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        logging.info(f"CartAgent.remove_from_cart() | /query/text -- getting response\n\t| query: {item_name}\n\t")
        ret_response = session.post(
            f"{self.catalog_retriever_port}/query/text",
            json={
                "text": [item_name],
                "categories": self.categories,
                "k": 1
            }
        )
        ret_response.raise_for_status()
        res_json = ret_response.json()
        sim = 0
        if res_json["similarities"]:
            sim = res_json["similarities"][0]
            if sim > 0.8:
                catalog_item_name = res_json["names"][0]
                logging.info(f"CartAgent.remove_from_cart() | input name: {item_name}, retrieved item: {catalog_item_name}, sim: {sim}")
                response = requests.post(
                    f"{self.memory_retriever_url}/user/{user_id}/cart/remove",
                    json={"item": catalog_item_name, "amount": quantity}
                )
                if response.status_code == 200:
                    return response.json()["message"]
                return f"Failed to remove {quantity} {catalog_item_name} from cart."
            else:
                logging.info(f"CartAgent.remove_from_cart() | Nothing sufficiently similar to {item_name} in the cart.")
                return f"No such item ({item_name}) could be found in the catalog."                
        else:
            return f"No such item ({item_name}) could be found in the catalog."

    def _update_context(self, user_id: int, context: str) -> None:
        response = requests.post(
            f"{self.memory_retriever_url}/user/{user_id}/context/add",
            json={"new_context": context}
        )
        if response.status_code != 200:
            logging.error(f"Failed to update context: {response.text}")

    def invoke(
        self,
        state: State,
        verbose : bool = True
    ) -> State:
        """
        Determines which function to perform and does that function using NVIDIA NIM.
        """
        start = time.monotonic()
        logging.info(f"CartAgent.invoke() | Starting with query: {state.query}")
        tools = [add_to_cart_function, remove_from_cart_function, view_cart_function]
        
        # Create proper ChatCompletionMessageParam objects
        messages: list[ChatCompletionMessageParam] = [
            {
                "role": "system", 
                "content": "You are a retail agent that assists shoppers with their cart.\nOnly use the tools provided to help them."
            },
            {
                "role": "user", 
                "content": f"USER QUERY: {state.query}\nCONTEXT: {state.context}"
            }
        ]

        # Create the request parameters
        response = self.model.chat.completions.create(
            model=self.llm_name,
            messages=messages,
            temperature=0.0,
            max_tokens=8192,
            tools=tools,
            tool_choice="auto",
            stream=False
        )

        # Parse our function call.
        called_tool = response.choices[0].message.tool_calls[0]
        tool_name = called_tool.function.name
        tool_args = json.loads(called_tool.function.arguments)  

        logging.info(f"CartAgent.invoke() | Tool name: {tool_name}")

        output_state = state 
        if verbose:
            logging.info(f"CartAgent.invoke() | tool_name: {tool_name}\n\t| tool_args: {tool_args}")

        # Perform our associated action.
        if tool_name == "add_to_cart":
            logging.info(f"CartAgent.invoke() | Adding to cart")
            item_name = tool_args["item_name"]
            quantity = tool_args["quantity"]
            output_state.response = self._add_to_cart(state.user_id, item_name, quantity)
            output_state.cart = self._get_cart(state.user_id)
            
        elif tool_name == "remove_from_cart":
            logging.info(f"CartAgent.invoke() | Removing from cart")
            item_name = tool_args["item_name"]
            quantity = tool_args["quantity"]    
            output_state.response = self._remove_from_cart(state.user_id, item_name, quantity)
            output_state.cart = self._get_cart(state.user_id)
            
        elif tool_name == "view_cart":
            cart = self._get_cart(state.user_id)
            logging.info(f"CartAgent.invoke() | Viewing cart.\n\t| Cart: {cart}")
            if len(cart.contents) == 0:
                output_state.response = "Your cart is empty."
            else:
                contents = cart.contents
                items = [f"The user has ({contents[ind]['amount']} {contents[ind]['item']}) in their cart" for ind in range(len(contents))]
                items_str = ". ".join(items)
                logging.info(f"CartAgent.invoke() | item list retrieved: {items_str}")
                output_state.response = f"{items_str}"
            output_state.cart = cart

        # Update our context and return our state.
        if verbose:
            logging.info(f"CartAgent.invoke() | output_state: {output_state}")
        
        #self._update_context(state.user_id, f"USER QUERY:{output_state.query}\nRESPONSE:{output_state.response}")
        end = time.monotonic()
        output_state.context = output_state.context + f"\nAgent Response: {output_state.response}"
        output_state.timings["cart"] = end - start
        logging.info(f"CartAgent.invoke() | Returning final state with response: {output_state.response}")

        return output_state


