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

from shared.weather_receipts import (
    WeatherForecastReceipt,
    WeatherReceiptPromotion,
)
from shared.weather_scope import (
    MAX_CURRENT_WEATHER_SCOPE_SOURCE_TURNS,
    CurrentWeatherScope,
    CurrentWeatherScopeResolution,
    CurrentWeatherScopeSourceTurn,
    CurrentWeatherScopeTransition,
)
from .weather_scope_resolver import WeatherScopeResolverDecision


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
    shopper_context: ShopperContext | None = Field(
        default=None,
        description="Server-resolved current-turn shopper guidance",
    )
    conversation_summary: str = Field(
        default="",
        description="Durable semantic continuity summary; never exact evidence",
    )
    context: str = Field(
        default="",
        description="Exact bounded raw conversation turns",
    )
    recent_conversation_turns: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Request-local structured bounded turns used to resolve typed "
            "weather-scope source-sequence semantics"
        ),
    )
    current_weather_scope_source_turns: List[
        CurrentWeatherScopeSourceTurn
    ] = Field(
        default_factory=list,
        max_length=MAX_CURRENT_WEATHER_SCOPE_SOURCE_TURNS,
        description=(
            "Durable turns referenced by the current weather scope, with "
            "assistant weather prose sanitized and the lane isolated from "
            "general conversation context"
        ),
    )
    historical_product_context: str = Field(
        default="",
        description="Authoritative bounded historical product-reference projection",
    )
    conversation_projection_version: int = Field(
        default=0,
        ge=0,
        description="Version used for optional atomic projection updates",
    )
    conversation_memory_contract_version: int = Field(
        default=4,
        ge=1,
        description="Negotiated durable-memory response contract for this turn",
    )
    active_weather_receipts: List[WeatherForecastReceipt] = Field(
        default_factory=list,
        description="Fresh typed weather receipts available for semantic binding",
    )
    selected_weather_receipt_id: str | None = Field(
        default=None,
        description="Current-turn receipt explicitly bound during skill activation",
    )
    weather_receipt_promotion: WeatherReceiptPromotion | None = Field(
        default=None,
        description="Validated current-turn forecast prepared for durable promotion",
    )
    current_weather_scope: CurrentWeatherScope = Field(
        default_factory=CurrentWeatherScope,
        description="Memory-owned current location/date authority for weather",
    )
    current_weather_scope_transition: CurrentWeatherScopeTransition | None = Field(
        default=None,
        description="Legacy v3 weather-scope transition; never authored by v4 runtime",
    )
    current_weather_scope_resolution: CurrentWeatherScopeResolution | None = Field(
        default=None,
        description="Validated atomic v4 weather-scope resolution for finalization",
    )
    weather_scope_resolver_decision: WeatherScopeResolverDecision | None = Field(
        default=None,
        description=(
            "Request-local business-tool-disabled semantic decision for an existing "
            "weather scope; never persisted as conversation authority"
        ),
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
