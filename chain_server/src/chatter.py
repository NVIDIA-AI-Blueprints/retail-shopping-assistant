# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import AsyncGenerator
from openai import AsyncOpenAI
from langgraph.config import get_stream_writer
from .agenttypes import State
import json
import os
import logging
import sys
import time


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    ) 

# Configuration will be loaded by the main application


class ChatterAgent:
    def __init__(self, config):
        """
        Initialize the ChatterAgent with LLM configuration.
        
        Args:
            config: Configuration instance
        """
        logging.info(f"ChatterAgent.__init__() | Initializing with llm_name={config.llm_name}, llm_port={config.llm_port}")
        self.llm_name = config.llm_name
        self.llm_port = config.llm_port
        self.config = config
        
        self.model = AsyncOpenAI(
            base_url=config.llm_port, 
            api_key=os.environ["LLM_API_KEY"]
        )
        logging.info(f"ChatterAgent.__init__() | Initialization complete")

    @staticmethod
    def _format_cart(state: State) -> str:
        """Render the authoritative cart as a bullet list.

        The only source the chatter may cite for cart claims. ``state.cart``
        is refreshed by the memory service each turn and by the cart agent
        on every mutation.
        """
        if not state.cart or not state.cart.contents:
            return "(empty)"
        lines = []
        for entry in state.cart.contents:
            amount = entry.get("amount", 1)
            name = entry.get("item", "")
            price = entry.get("price")
            if price is not None:
                try:
                    lines.append(f"- {amount} x {name} @ ${float(price):.2f}")
                except (TypeError, ValueError):
                    lines.append(f"- {amount} x {name}")
            else:
                lines.append(f"- {amount} x {name}")
        return "\n".join(lines)

    @staticmethod
    def _format_available_catalog(state: State) -> str:
        """Render products retrieved this turn as an explicit allowlist.

        Only fresh retrievals populate ``state.retrieved``; past results
        live in the running context as prose and are excluded so the
        chatter does not re-claim out-of-scope items.
        """
        if not state.retrieved:
            return "(no fresh catalog results this turn)"
        return "\n".join(f"- {name}" for name in state.retrieved.keys())

    @staticmethod
    def _describe_preceding_agent(state: State) -> str:
        """Report which specialist ran this turn.

        The chatter runs regardless of routing; passing the upstream
        agent lets it gate cart-mutation language on the current turn
        instead of extrapolating from past turns in context.
        """
        agent = (state.next_agent or "").strip().lower()
        if agent == "cart":
            return "cart"
        if agent == "retriever":
            return "retriever"
        return "none"

    async def invoke(
        self, 
        state: State,
        verbose: bool = True
        ) -> AsyncGenerator[State, None]:
        """
        Process the user query and generate a response with streaming.
        """
        logging.info(f"ChatterAgent.invoke() | Starting with query: {state.query}")
        output_state = state
        logging.info(f"ChatterAgent.invoke() | State retrieved.")

        query_text = state.query or (
            "The user has submitted an image and is looking for items that appear similar."
        )
        preceding_agent = self._describe_preceding_agent(state)
        agent_result = (state.response or "").strip() or "(none)"
        cart_block = self._format_cart(state)
        catalog_block = self._format_available_catalog(state)
        recent_context = (state.context or "").strip() or "(none)"

        user_message = (
            f"USER QUERY: {query_text}\n\n"
            f"PRECEDING AGENT (ran this turn before you): {preceding_agent}\n"
            f"PRECEDING AGENT RESULT (verbatim, authoritative for this turn): {agent_result}\n\n"
            f"CURRENT CART (authoritative):\n{cart_block}\n\n"
            f"AVAILABLE CATALOG (fresh retrieval for this turn; the only NEW products you may introduce):\n"
            f"{catalog_block}\n\n"
            f"RECENT DISCUSSION (reference only; paraphrased past turns, NOT authoritative for cart state):\n"
            f"{recent_context}"
        )

        messages = [
            {"role": "system", "content": self.config.chatter_prompt},
            {"role": "user", "content": user_message},
        ]

        logging.info(
            f"ChatterAgent.invoke() | preceding_agent={preceding_agent} "
            f"cart_items={len(state.cart.contents) if state.cart else 0} "
            f"retrieved_count={len(state.retrieved)} context_len={len(state.context or '')}"
        )

        start = time.monotonic()

        logging.info(f"ChatterAgent.invoke() | Context length is less than memory length")
        full_response = ""
        ftr = False

        writer = get_stream_writer()

        # Send our 'retrieved' dictionary.
        writer(f"{json.dumps({'type' : 'images' , 'payload' : state.retrieved, 'timestamp' : time.time()})}")

        stream = await self.model.chat.completions.create(
            model=self.llm_name,
            messages=messages,
            stream=True,
            temperature=0.0,
            max_tokens=self.config.memory_length,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                output_state.response = full_response

                if not ftr:
                    ftr = True
                    ftt = time.monotonic() - start
                    logging.info(f"ChatterAgent.invoke() | First token time: {ftt}")
                    output_state.timings["first_token"] = ftt

                writer(f"{json.dumps({'type' : 'content', 'payload' : content, 'timestamp' : time.time()})}")

        output_state.response = full_response
        output_state.context = f"{state.context}\n{full_response}"
            
        logging.info(f"ChatterAgent.invoke() | Returning final state with response: {output_state.response[0:50]}")

        end = time.monotonic()
        output_state.timings["chatter"] = end - start
        return(output_state)
