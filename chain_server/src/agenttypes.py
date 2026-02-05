# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Data models for the Shopping Assistant.

This module defines the core data structures used throughout the shopping assistant,
including the main State object that flows through the LangGraph and supporting models.
"""
from operator import ior
from pydantic import BaseModel, Field
from typing import Annotated, Dict, List, Any


class Cart(BaseModel):
    """
    Shopping cart model for storing user's selected items.
    
    Attributes:
        contents: List of cart items with their quantities and metadata
    """
    contents: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of items in the cart with their quantities and metadata"
    )
    
    def is_empty(self) -> bool:
        """Check if the cart is empty."""
        return len(self.contents) == 0
    
    def get_item_count(self) -> int:
        """Get the total number of items in the cart."""
        return sum(item.get('amount', 0) for item in self.contents)
    
    def get_items(self) -> List[str]:
        """Get a list of unique item names in the cart."""
        return list(set(item.get('item', '') for item in self.contents))


class State(BaseModel):
    """
    Main state object that flows through the LangGraph.
    
    This object contains all the information needed by the various agents
    to process user queries and generate responses.
    
    Attributes:
        user_id: Unique identifier for the user
        query: The user's input query
        context: Previous conversation context
        cart: User's shopping cart
        response: Generated response from agents
        image: Base64 encoded image data (if provided)
        retrieved: Dictionary of retrieved product information
        next_agent: Next agent to route to (set by planner)
        guardrails: Whether to enable content safety checks
        timings: Performance timing information
    """
    user_id: int = Field(..., description="Unique user identifier")
    query: str = Field(..., description="User's input query")
    context: str = Field(default="", description="Previous conversation context")
    cart: Cart = Field(default_factory=Cart, description="User's shopping cart")
    response: str = Field(default="", description="Generated response from agents")
    image: str = Field(default="", description="Base64 encoded image data")
    retrieved: Dict[str, str] = Field(
        default_factory=dict,
        description="Dictionary of retrieved product information"
    )
    next_agent: str = Field(default="", description="Next agent to route to")
    guardrails: bool = Field(default=True, description="Enable content safety checks")
    timings: Annotated[Dict[str, float], ior] = Field(
        default_factory=dict,
        description="Performance timing information for each step"
    )
    
    def add_timing(self, step: str, duration: float) -> None:
        """Add timing information for a processing step."""
        self.timings[step] = duration
    
    def get_total_time(self) -> float:
        """Get the total processing time."""
        return sum(self.timings.values())
    
    def has_image(self) -> bool:
        """Check if the state contains an image."""
        return bool(self.image.strip())
    
    def is_empty_query(self) -> bool:
        """Check if the query is empty."""
        return not bool(self.query.strip())


class Rail(BaseModel):
    """
    Guardrails check result model.
    
    This model represents the result of content safety checks
    performed by the guardrails service.
    
    Attributes:
        is_safe: Whether the content passed safety checks
        rail_timings: Timing information for the safety check
    """
    is_safe: bool = Field(default=True, description="Whether content passed safety checks")
    rail_timings: Dict[str, float] = Field(
        default_factory=dict,
        description="Timing information for safety checks"
    )
    
    def add_timing(self, check_type: str, duration: float) -> None:
        """Add timing information for a specific safety check."""
        self.rail_timings[check_type] = duration
    
    def get_total_rail_time(self) -> float:
        """Get the total time spent on safety checks."""
        return sum(self.rail_timings.values())


# Type aliases for better code readability
AgentResponse = Dict[str, Any]
ProductInfo = Dict[str, Any]
TimingInfo = Dict[str, float]