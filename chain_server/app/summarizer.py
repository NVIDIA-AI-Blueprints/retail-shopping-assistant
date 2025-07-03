from typing import Dict, Any
from openai import OpenAI
from .agenttypes import State
from .functions import summary_function
import requests
import json
import os
import logging
import sys
import yaml
import time

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    ) 

# Load configuration
config_path = os.path.join("/app", "app", "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

MEMORY_LENGTH = config["memory_length"]

class SummaryAgent:
    def __init__(self, llm_name: str, llm_port: str):
        """
        Initialize the SummaryAgent with LLM configuration.
        
        Args:
            llm_name: Name of the LLM model to use
            llm_port: URL of the LLM service
        """
        logging.info(f"SummaryAgent.__init__() | Initializing with llm_name={llm_name}, llm_port={llm_port}")
        self.llm_name = llm_name
        self.llm_port = llm_port
        self.model = OpenAI(base_url=llm_port, api_key=os.environ["LLM_API_KEY"])
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
        if len(state.context) > MEMORY_LENGTH:
            logging.info(f"SummaryAgent.invoke() | Context length is greater than memory length")
            response = self.model.chat.completions.create(
                model=self.llm_name,
                messages=messages,
                tools=[summary_function],
                tool_choice="required",
                stream=False,
                temperature=0.0,
                max_tokens=MEMORY_LENGTH
            )

            tool_json = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
            output_state.context = tool_json["summary"]
            logging.info(f"SummaryAgent.invoke() | Returning final state with response: {output_state.context}")
        else:
            logging.info(f"SummaryAgent.invoke() | Context length is less than memory length -- writing to memory.")
        
        requests.post(f"{config['memory_port']}/user/{output_state.user_id}/context/replace", json={"new_context": output_state.context})

        end = time.monotonic()
        
        logging.info(f"SummaryAgent.invoke() | Completed summarization in {end - start} seconds.")
