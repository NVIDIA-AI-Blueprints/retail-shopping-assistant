"""
LangGraph orchestration for the Shopping Assistant.

This module defines the topology and flow of the shopping assistant using LangGraph,
connecting various specialized agents to handle different types of user queries.
"""
from typing import Any
import time
import logging
import requests
import json
import sys

from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer
from langchain_core.runnables import RunnablePassthrough

from .agenttypes import State, Cart, Rail


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


# Global configuration variable
_config = None


class GraphNodes:
    """Container for graph node functions."""
    
    @staticmethod
    async def get_memory(state: State) -> State:
        """Retrieve user memory and cart from the memory service."""
        start = time.monotonic()
        logger.info(f"GraphNodes.get_memory() | Retrieving memory for user {state.user_id}")
        
        try:
            # Retrieve memory from the memory database
            memory_response = requests.get(
                f"{_config.memory_port}/user/{state.user_id}/context",
                timeout=10
            )
            memory_response.raise_for_status()
            memory = memory_response.json()
            
            # Retrieve cart from the memory database
            cart_response = requests.get(
                f"{_config.memory_port}/user/{state.user_id}/cart",
                timeout=10
            )
            cart_response.raise_for_status()
            cart = cart_response.json()

            logger.info(f"GraphNodes.get_memory() | Memory retrieved: {memory}, Cart: {cart}")
            
            # Update state with retrieved data
            state.context = memory["context"]
            state.cart.contents = cart["cart"]
            
            end = time.monotonic()
            state.timings["memory"] = end - start
            
            return state
            
        except requests.RequestException as e:
            logger.error(f"GraphNodes.get_memory() | Failed to retrieve memory: {e}")
            # Return state with empty context/cart on failure
            state.context = ""
            state.cart.contents = []
            state.timings["memory"] = time.monotonic() - start
            return state
    
    @staticmethod
    async def check_input_safety(state: State) -> Rail:
        """Check if the user input is safe using guardrails."""
        if not state.guardrails:
            return {"is_safe": True}
        
        start = time.monotonic()
        
        try:
            response = requests.post(
                f"{_config.rails_port}/rail/input/check",
                json={"user_id": state.user_id, "query": state.query},
                timeout=10
            )
            response.raise_for_status()
            
            is_safe = response.json()["content"] == state.query
            end = time.monotonic()
            
            return {
                "is_safe": is_safe,
                "rail_timings": {"rails_input_check": end - start}
            }
            
        except requests.RequestException as e:
            logger.error(f"Failed to check input safety: {e}")
            # Default to safe on failure
            return {
                "is_safe": True,
                "rail_timings": {"rails_input_check": time.monotonic() - start}
            }
    
    @staticmethod
    async def check_output_safety(state: State) -> Rail:
        """Check if the generated response is safe using guardrails."""
        if not state.guardrails:
            return {"is_safe": True}
        
        start = time.monotonic()
        
        try:
            response = requests.post(
                f"{_config.rails_port}/rail/output/check",
                json={"user_id": state.user_id, "query": state.response},
                timeout=10
            )
            response.raise_for_status()
            
            is_safe = response.json()["content"] == state.response
            end = time.monotonic()
            
            return {
                "is_safe": is_safe,
                "rail_timings": {"rails_output_check": end - start}
            }
            
        except requests.RequestException as e:
            logger.error(f"Failed to check output safety: {e}")
            # Default to safe on failure
            return {
                "is_safe": True,
                "rail_timings": {"rails_output_check": time.monotonic() - start}
            }
    
    @staticmethod
    async def check_rail_node(rail: Rail) -> State:
        """Process rail check results and update state timings."""
        logger.info(f"GraphNodes.check_rail_node() |Rail check result: {rail}")
        return {"timings": rail.rail_timings}
    
    @staticmethod
    async def unsafe_output(rail: Rail) -> State:
        """Handle unsafe content by returning a safe message."""
        unsafe_message = _config.unsafe_message
        writer = get_stream_writer()
        writer(f"{json.dumps({'type': 'content', 'payload': unsafe_message, 'timestamp': time.time()})}")
        return {"response": unsafe_message}


class GraphRouting:
    """Routing logic for the graph."""
    
    @staticmethod
    def decide_if_input_safe(rail: Rail) -> str:
        """Route based on input safety check."""
        return "chatter_node" if rail.is_safe else "unsafe_output"
    
    @staticmethod
    def decide_if_output_safe(rail: Rail) -> str:
        """Route based on output safety check."""
        return "summarize_node" if rail.is_safe else "unsafe_output"


def create_graph(
    cart_agent: Any,
    retriever_agent: Any,
    planner_agent: Any,
    chatter_agent: Any,
    summary_agent: Any,
    config
) -> StateGraph:
    """
    Create the LangGraph for the shopping assistant.
    
    The graph orchestrates the flow between different specialized agents:
    - Memory retrieval
    - Input safety checks
    - Query routing via planner
    - Specialized agent processing (cart, retriever)
    - Output generation via chatter
    - Output safety checks
    - Response summarization
    
    Args:
        cart_agent: Agent for shopping cart operations
        retriever_agent: Agent for product search and retrieval
        planner_agent: Agent for query routing
        chatter_agent: Agent for natural language responses
        summary_agent: Agent for response summarization
    
    Returns:
        Compiled LangGraph instance
    """
    logger.info("Creating shopping assistant graph")
    
    # Set the global config for use throughout the graph
    global _config
    _config = config
    
    # Create the graph
    graph = StateGraph(State)
    
    # Add nodes with descriptive names
    graph.add_node("memory_node", GraphNodes.get_memory)
    graph.add_node("rails_input_node", GraphNodes.check_input_safety)
    graph.add_node("planner_node", planner_agent.invoke)
    graph.add_node("cart_node", cart_agent.invoke)
    graph.add_node("retriever_node", retriever_agent.invoke)
    graph.add_node("check_rail_node", GraphNodes.check_rail_node)
    graph.add_node("check_out_node", GraphNodes.check_rail_node)
    graph.add_node("passthrough_node", RunnablePassthrough())
    graph.add_node("chatter_node", chatter_agent.invoke)
    graph.add_node("rails_output_node", GraphNodes.check_output_safety)
    graph.add_node("summarize_node", summary_agent.invoke)
    graph.add_node("unsafe_output", GraphNodes.unsafe_output)

    # Set the entry point
    graph.add_edge(START, "memory_node")
    
    # Start planner node and rails checks in parallel
    graph.add_edge("memory_node", "planner_node")
    graph.add_edge("memory_node", "rails_input_node")

    # Add conditional routing based on planner decision
    graph.add_conditional_edges(
        "planner_node",
        planner_agent.decide_function,
        {
            "cart": "cart_node",
            "retriever": "retriever_node",
            "chatter": "passthrough_node",
        }
    )

    # Add edges from specialized agents to safety checks
    graph.add_edge(["cart_node", "rails_input_node"], "check_rail_node")
    graph.add_edge(["retriever_node", "rails_input_node"], "check_rail_node")
    graph.add_edge(["passthrough_node", "rails_input_node"], "check_rail_node")

    # Add conditional routing for input safety
    graph.add_conditional_edges("check_rail_node", GraphRouting.decide_if_input_safe)

    # Add edges for output processing
    graph.add_edge("chatter_node", "rails_output_node")
    graph.add_edge("rails_output_node", "check_out_node")

    # Add conditional routing for output safety
    graph.add_conditional_edges("check_out_node", GraphRouting.decide_if_output_safe)

    # End graph
    graph.add_edge("summarize_node", END)
    graph.add_edge("unsafe_output", END)
    
    # Compile and return the graph
    compiled_graph = graph.compile()
    logger.info("create_graph() | Graph created successfully.")
    
    return compiled_graph
