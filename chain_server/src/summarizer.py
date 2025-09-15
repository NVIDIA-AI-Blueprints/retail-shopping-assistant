# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from openai import OpenAI
from .agenttypes import State
from .functions import summary_function
import requests
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

class SummaryAgent:
    def __init__(self, config):
        """
        Initialize the SummaryAgent with LLM configuration.
        
        Args:
            config: Configuration instance
        """
        logging.info(f"SummaryAgent.__init__() | Initializing with llm_name={config.llm_name}, llm_port={config.llm_port}")
        self.llm_name = config.llm_name
        self.llm_port = config.llm_port
        
        # Store configuration
        self.memory_length = config.memory_length
        self.memory_port = config.memory_port
        
        self.model = OpenAI(base_url=config.llm_port, api_key=os.environ["LLM_API_KEY"])
        logging.info(f"SummaryAgent.__init__() | Initialization complete")

    def invoke(
        self, 
        state: State,
        verbose: bool = True
        ) -> State:
        """
        Process the user query and generate a response.
        """
        logging.info(f"SummaryAgent.invoke() | Starting with query: {state.query}\n\t Context: {state.context}")
        output_state = state

        messages = [
            {"role": "system", "content": """/no_think You are a conversation summarizer for a shopping assistant. 

                CRITICAL RULES:
                1. You MUST preserve ALL product information, including:
                - Complete product names with all descriptors
                - ALL care instructions (washing, drying, ironing, dry cleaning instructions)
                - Materials and fabric composition
                - Prices
                - Colors, sizes, and any other specifications

                2. For ANY product the user has shown interest in or asked about:
                - Keep the ENTIRE product description including all details
                - Preserve any attributes mentioned (care instructions, materials, features)

                3. You may condense only:
                - General conversation flow and greetings
                - Redundant phrases that don't contain product information
                - User's general preferences (but keep specific requirements)

                4. NEVER remove or shorten product specifications, even to save space.

                The goal is to maintain all factual product information while reducing conversational overhead."""},
                            {"role": "user", "content": f"CONTEXT TO SUMMARIZE:\n{state.context}"}
        ]

        start = time.monotonic()
        if len(state.context) > self.memory_length:
            logging.info(f"SummaryAgent.invoke() | Context length is greater than memory length")
            response = self.model.chat.completions.create(
                model=self.llm_name,
                messages=messages,
                tools=[summary_function],
                tool_choice="auto",
                stream=False,
                temperature=0.0,
                max_tokens=self.memory_length
            )

            if not response.choices[0].message.tool_calls:
                logging.error(f"SummaryAgent.invoke() | No tool calls generated for summarization")
                output_state.context = output_state.context  # Keep existing context if summarization fails
            else:
                tool_json = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
                output_state.context = tool_json["summary"]
            logging.info(f"SummaryAgent.invoke() | Returning final state with response: {output_state.context}")
        else:
            logging.info(f"SummaryAgent.invoke() | Context length is less than memory length -- writing to memory.")
        
        requests.post(f"{self.memory_port}/user/{output_state.user_id}/context/replace", json={"new_context": output_state.context})

        end = time.monotonic()
        
        logging.info(f"SummaryAgent.invoke() | Completed summarization in {end - start} seconds.")
        return output_state
