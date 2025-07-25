from typing import AsyncGenerator
from langchain_nvidia_ai_endpoints import ChatNVIDIA
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
        
        self.model = ChatNVIDIA(
            url=config.llm_port, 
            model=config.llm_name, 
            api_key=os.environ["LLM_API_KEY"],
            temperature=0.0,
            max_tokens=config.memory_length
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
                {"role": "system", "content": self.config.chatter_prompt},
                {"role": "user", "content": f"QUERY: {state.query}\nPREVIOUS CONTEXT: {state.context}"}
            ]
        else:
            messages = [
                {"role": "system", "content": self.config.chatter_prompt},
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
            
        logging.info(f"ChatterAgent.invoke() | Returning final state with response: {output_state.response[0:50]}")

        end = time.monotonic()
        output_state.timings["chatter"] = end - start
        return(output_state)
