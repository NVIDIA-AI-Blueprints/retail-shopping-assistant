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
            {"role": "system", "content": "It is your job to summarize the context and cart of the user."},
            {"role": "user", "content": f"CONTEXT:{state.context}"}
        ]

        start = time.monotonic()
        if len(state.context) > self.memory_length:
            logging.info(f"SummaryAgent.invoke() | Context length is greater than memory length")
            response = self.model.chat.completions.create(
                model=self.llm_name,
                messages=messages,
                tools=[summary_function],
                tool_choice="required",
                stream=False,
                temperature=0.0,
                max_tokens=self.memory_length
            )

            tool_json = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
            output_state.context = tool_json["summary"]
            logging.info(f"SummaryAgent.invoke() | Returning final state with response: {output_state.context}")
        else:
            logging.info(f"SummaryAgent.invoke() | Context length is less than memory length -- writing to memory.")
        
        requests.post(f"{self.memory_port}/user/{output_state.user_id}/context/replace", json={"new_context": output_state.context})

        end = time.monotonic()
        
        logging.info(f"SummaryAgent.invoke() | Completed summarization in {end - start} seconds.")
