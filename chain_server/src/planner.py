"""
Query routing agent for the Shopping Assistant.

This module contains the PlannerAgent that determines which specialized agent
should handle a user's query based on the query content and context.
"""
import os
import logging
import sys
import time
from typing import Dict, List
from openai import OpenAI

from .agenttypes import State, Cart


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


# Configuration will be loaded by the main application


class PlannerAgent:
    """
    Agent responsible for routing user queries to appropriate specialized agents.
    
    The planner analyzes the user's query and determines whether it should be
    handled by the cart agent, retriever agent, visualizer agent, or chatter agent.
    """
    
    def __init__(
        self,
        config,
    ) -> None:
        """
        Initialize the PlannerAgent.
        
        Args:
            config: Configuration instance
        """
        logger.info(f"PlannerAgent.__init__() | llm_name={config.llm_name}, llm_port={config.llm_port}")
        
        self.llm_name = config.llm_name
        self.llm_port = config.llm_port
        self.agent_choices = config.agent_choices
        self.system_prompt = config.routing_prompt
        
        # Initialize the LLM client
        try:
            self.model = OpenAI(
                base_url=self.llm_port,
                api_key=os.environ.get("LLM_API_KEY")
            )
            logger.info("PlannerAgent.__init__() | initialization complete")
        except Exception as e:
            logger.error(f"Failed to initialize PlannerAgent: {e}")
            raise

    def _create_routing_messages(self, query: str) -> List[Dict[str, str]]:
        """
        Create the messages for the routing decision.
        
        Args:
            query: The user's query to route
            
        Returns:
            List of messages for the LLM
        """
        return [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": f"Customer Query: {query}"
            }
        ]

    def _call_llm_for_routing(self, query: str) -> str:
        """
        Call the LLM to determine the appropriate agent for the query.
        
        Args:
            query: The user's query
            
        Returns:
            The name of the agent to route to
        """
        try:
            messages = self._create_routing_messages(query)
            
            response = self.model.chat.completions.create(
                model=self.llm_name,
                messages=messages,
                temperature=0.0,
                max_tokens=100
            )
            
            response_content = response.choices[0].message.content.strip().lower()
            logger.debug(f"LLM routing response: {response_content}")
            
            return response_content
            
        except Exception as e:
            logger.error(f"Error calling LLM for routing: {e}")
            return "chatter"  # Default to chatter on error

    def _normalize_agent_name(self, agent_name: str) -> str:
        """
        Normalize agent names to match graph node names.
        
        Args:
            agent_name: Raw agent name from LLM
            
        Returns:
            Normalized agent name
        """
        # Map common variations to standard names
        mappings = {
            "search": "retriever",
            "cart_node": "cart",
            "product_finder": "retriever",
            "general": "chatter",
            "assistant": "chatter"
        }
        
        normalized = mappings.get(agent_name, agent_name)
        
        # Ensure the normalized name is in our valid choices
        if normalized not in self.agent_choices:
            logger.warning(f"Invalid agent choice '{normalized}', defaulting to 'chatter'")
            return "chatter"
        
        return normalized

    def invoke(
        self,
        state: State,
        verbose: bool = True
    ) -> State:
        """
        Process the user query to determine which agent should handle it.
        
        Args:
            state: Current state containing the user query
            verbose: Whether to log detailed information
            
        Returns:
            Updated state with the next_agent field set
        """
        start_time = time.monotonic()
        logger.info(f"PlannerAgent.invoke() | Processing routing for query: {state.query}")
        
        output_state = state
        
        # Handle image-only queries
        if state.has_image() and state.is_empty_query():
            logger.info("PlannetAgent.invoke() | Image-only query detected, routing to retriever")
            response_content = "retriever"
        else:
            # Use LLM to determine routing
            query_string = f"USER QUERY: {state.query}\n CONTEXT: {state.context}" 
            response_content = self._call_llm_for_routing(query_string)
        
        # Normalize the agent name
        normalized_agent = self._normalize_agent_name(response_content)
        
        # Update the state
        output_state.next_agent = normalized_agent
        end_time = time.monotonic()
        output_state.add_timing("planner", end_time - start_time)

        logger.info(f"PlannerAgent.invoke() | Routed query to agent: {normalized_agent}")
        return output_state

    def decide_function(self, state: State) -> str:
        """
        Return the next agent to route to based on the state.
        
        This method is used by the LangGraph for conditional routing.
        
        Args:
            state: Current state with next_agent field
            
        Returns:
            Name of the next agent to route to
        """
        next_agent = getattr(state, "next_agent", "")
        if not next_agent:
            logger.warning("No next_agent in state, defaulting to chatter")
            return "chatter"
        
        logger.debug(f"Routing to agent: {next_agent}")
        return next_agent