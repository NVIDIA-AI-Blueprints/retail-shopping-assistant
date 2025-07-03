from typing import Dict, Any, AsyncGenerator
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.config import get_stream_writer
from .agenttypes import State
import requests
import json
import os
import logging
import sys
import yaml
import time
import copy

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
CHAT_PROMPT = config["chatter_prompt"]


class ChatterAgent:
    def __init__(self, llm_name: str, llm_port: str):
        """
        Initialize the ChatterAgent with LLM configuration.
        
        Args:
            llm_name: Name of the LLM model to use
            llm_port: URL of the LLM service
        """
        logging.info(f"ChatterAgent.__init__() | Initializing with llm_name={llm_name}, llm_port={llm_port}")
        self.llm_name = llm_name
        self.llm_port = llm_port
        self.model = ChatNVIDIA(
            url=llm_port, 
            model=llm_name, 
            api_key=os.environ["LLM_API_KEY"],
            temperature=0.0,
            max_tokens=MEMORY_LENGTH
        )
        logging.info(f"ChatterAgent.__init__() | Initialization complete")

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

        if state.query:
            messages = [
                {"role": "system", "content": CHAT_PROMPT},
                {"role": "user", "content": f"QUERY: {state.query}\nPREVIOUS CONTEXT: {state.context}"}
            ]
        else:
            messages = [
                {"role": "system", "content": CHAT_PROMPT},
                {"role": "user", "content": f"QUERY: 'You have been sent an image, and the retrieved items are the most similar items.'\nPREVIOUS CONTEXT: {state.context}"}
            ]

        start = time.monotonic()

        logging.info(f"ChatterAgent.invoke() | Context length is less than memory length")
        full_response = ""
        
        ftr = False

        writer = get_stream_writer()

        # Send our 'retrieved' dictionary.
        writer(f"{json.dumps({'type' : 'images' , 'payload' : state.retrieved, 'timestamp' : time.time()})}")

        async for chunk in self.model.astream(messages):

            if chunk.content:
                content = chunk.content
                full_response += content
                output_state.response = full_response

                if not ftr:
                    ftr = True
                    ftt = time.monotonic() - start
                    logging.info(f"ChatterAgent.invoke() | First token time: {ftt}")
                    output_state.timings["first_token"] = ftt

                writer(f"{json.dumps({'type' : 'content', 'payload' : chunk.content, 'timestamp' : time.time()})}")

        output_state.response = full_response
        output_state.context = f"{state.context}\n{full_response}"
            
        logging.info(f"ChatterAgent.invoke() | Returning final state with response: {output_state.response}")

        end = time.monotonic()
        output_state.timings["chatter"] = end - start
        return(output_state)
