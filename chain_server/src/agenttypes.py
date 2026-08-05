# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Data models for the Shopping Assistant.

This module defines the core data structures used throughout the shopping assistant,
including the main State object that flows through the LangGraph and supporting models.
"""
from operator import ior
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Annotated, Dict, List, Any


SHOPPER_PROFILE_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$"


class ShopperContext(BaseModel):
    """Server-resolved soft guidance for one representative shopper."""

    model_config = ConfigDict(extra="forbid", strict=True)

    shopper_type: str = Field(
        ...,
        min_length=1,
        max_length=80,
        pattern=r"^[a-z][a-z0-9_]*$",
    )
    behavior: str = Field(..., min_length=1, max_length=512)
    zipcode: str = Field(..., pattern=r"^[0-9]{5}$")

    @field_validator("behavior")
    @classmethod
    def _validate_behavior(cls, value: str) -> str:
        if value != value.strip() or "\n" in value or "\r" in value:
            raise ValueError("shopper behavior must be one trimmed line")
        return value


class DialogueTurn(BaseModel):
    """One prior exchange carried as typed intent context.

    Dialogue may establish what the shopper means — "yes", "the first one" — but
    never product, policy, inventory, or cart facts. Text here is exactly what
    was rendered into the prompt, so the typed lane and the rendered lane can
    never disagree.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence: int = Field(..., ge=1)
    shopper_text: str = Field(...)
    assistant_text: str = Field(...)


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
    shopper_profile_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=SHOPPER_PROFILE_ID_PATTERN,
        description="Selected immutable representative-shopper key",
    )
    #: Audience most recently declared for who is being shopped for, carried
    #: from earlier turns. Dialogue establishes intent, never fact, so a wearer
    #: named a turn ago has no standing until it arrives as a value.
    wearer_audience: List[str] = Field(default_factory=list)
    shopper_context: ShopperContext | None = Field(
        default=None,
        description="Server-resolved current-turn shopper guidance",
    )
    dialogue: List[DialogueTurn] = Field(
        default_factory=list,
        description="Typed prior turns; authoritative for shopper intent context"
    )
    historical_product_sets: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Typed historical product reference sets; identity authority only, "
            "never current product-fact authority"
        )
    )
    dialogue_context: str = Field(
        default="",
        description="Rendered dialogue only, excluding the product index"
    )
    context: str = Field(
        default="",
        description="Rendered prompt text only; never parsed back into state"
    )
    cart: Cart = Field(default_factory=Cart, description="User's shopping cart")
    response: str = Field(default="", description="Generated response from agents")
    image: str = Field(default="", description="Base64 encoded image data")
    media: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Normalized media attachments for the current turn"
    )
    media_analysis: str = Field(
        default="",
        description="Structured VLM analysis for the current turn's media"
    )
    retrieved: Dict[str, str] = Field(
        default_factory=dict,
        description="Dictionary of retrieved product information"
    )
    product_results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Structured product summaries returned during the current turn"
    )
    token_usage: Dict[str, int] = Field(
        default_factory=dict,
        description="Normalized model token usage for the current turn"
    )
    model_usage: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-role model usage summary for the current turn"
    )
    agent_diagnostics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Ordered Deep Agents tool and termination diagnostics"
    )
    previous_selected_skill_names: List[str] = Field(
        default_factory=list,
        description="Prior turn skill-selection hint loaded from durable memory"
    )
    selected_skill_names: List[str] = Field(
        default_factory=list,
        description="Shopper skills selected during the current turn"
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
