# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Centralized configuration management for the chain server."""

import math
import os
from pathlib import Path
import yaml
import logging
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, model_validator, validator

from .weather import WeatherConfig
from shared.model_config import resolve_model_config, validate_model_config

logger = logging.getLogger(__name__)

_MIN_CONVERSATION_SUMMARY_SOURCE_HEADROOM = 512


def load_config_data(base_config_path: str) -> Dict[str, Any]:
    """Load a service config YAML file."""

    if not os.path.exists(base_config_path):
        logger.error(f"Base config file not found at {base_config_path}")
        raise FileNotFoundError(f"Base config file not found at {base_config_path}")

    with open(base_config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


class MediaInputConfig(BaseModel):
    """Configuration for user-attached media accepted by the chain server."""

    enabled: bool = True
    allow_mixed_media: bool = True
    max_images_per_turn: int = 1
    max_videos_per_turn: int = 1
    image_mime_types: List[str] = Field(default_factory=lambda: ["image/jpeg", "image/png"])
    video_mime_types: List[str] = Field(default_factory=lambda: ["video/mp4"])
    max_image_bytes: int = 10 * 1024 * 1024
    max_video_bytes: int = 50 * 1024 * 1024
    max_video_duration_seconds: int = 120

    @validator("max_images_per_turn", "max_videos_per_turn")
    def validate_media_counts(cls, v):
        if v < 0:
            raise ValueError("media limits cannot be negative")
        return v

    @validator("max_image_bytes", "max_video_bytes", "max_video_duration_seconds")
    def validate_positive_media_limits(cls, v):
        if v <= 0:
            raise ValueError("media size and duration limits must be positive")
        return v

    @validator("image_mime_types", "video_mime_types")
    def validate_mime_types(cls, v):
        if not v:
            raise ValueError("media MIME type lists cannot be empty")
        return v

    class Config:
        extra = "forbid"
        validate_assignment = True


class ConversationSummaryConfig(BaseModel):
    """Bounded rolling-summary compaction policy."""

    enabled: bool = True
    trigger_raw_turns: int = 6
    retain_raw_turns: int = 2
    max_output_chars: int = 4096
    timeout_seconds: float = 5.0

    @validator("trigger_raw_turns", "max_output_chars")
    def validate_positive_limits(cls, value):
        if value <= 0:
            raise ValueError("conversation summary limits must be positive")
        return value

    @validator("retain_raw_turns")
    def validate_retained_turns(cls, value):
        if value < 2:
            raise ValueError(
                "conversation summary must retain at least two raw turns"
            )
        return value

    @validator("timeout_seconds")
    def validate_timeout(cls, value):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                "conversation summary timeout must be finite and positive"
            )
        return value

    @model_validator(mode="after")
    def validate_trigger_and_output(self):
        if self.retain_raw_turns >= self.trigger_raw_turns:
            raise ValueError(
                "conversation summary retain_raw_turns must be smaller than "
                "trigger_raw_turns"
            )
        if self.max_output_chars > 16_384:
            raise ValueError(
                "conversation summary max_output_chars cannot exceed 16384"
            )
        return self

    class Config:
        extra = "forbid"
        validate_assignment = True


class ChainServerConfig(BaseModel):
    """Configuration class for the chain server application."""
    
    # LLM Configuration
    llm_port: str = Field(..., description="LLM service endpoint URL")
    llm_name: str = Field(..., description="LLM model name")
    llm_api_key_env: Optional[str] = Field(default="LLM_API_KEY", description="LLM API key environment variable")
    llm_api_key_required: bool = Field(default=True, description="Whether the LLM key must be present")
    vlm_port: Optional[str] = Field(default=None, description="Optional VLM service endpoint URL")
    vlm_name: Optional[str] = Field(default=None, description="Optional VLM model name")
    vlm_api_key_env: Optional[str] = Field(default=None, description="VLM API key environment variable")
    vlm_api_key_required: bool = Field(default=False, description="Whether the VLM key must be present")
    vlm_enabled: bool = Field(default=False, description="Whether VLM media perception is enabled")
    
    # Service Endpoints
    retriever_port: str = Field(..., description="Catalog retriever service endpoint")
    memory_port: str = Field(..., description="Memory retriever service endpoint")
    rails_port: str = Field(..., description="Guardrails service endpoint")
    
    # Prompts
    routing_prompt: str = Field(..., description="System prompt for routing queries to appropriate agents")
    chatter_prompt: str = Field(..., description="System prompt for general conversation")
    
    # Legacy Product Configuration
    categories: List[str] = Field(
        default_factory=list,
        description=(
            "Legacy category list for non-entrypoint graph agents. The active "
            "Deep Agents runtime gets catalog filters from catalog capabilities."
        ),
    )
    agent_choices: List[str] = Field(..., description="Available agent types")
    
    # Performance Configuration
    memory_length: int = Field(..., description="Maximum memory length for context")
    conversation_summary: ConversationSummaryConfig = Field(
        default_factory=ConversationSummaryConfig,
        description="Durable rolling-summary compaction policy",
    )
    top_k_retrieve: int = Field(..., description="Number of top results to retrieve")
    deepagents_recursion_limit: int = Field(
        default=24,
        description="Maximum Deep Agents graph steps allowed for one assistant turn",
    )
    deepagents_execution_timeout_seconds: float = Field(
        default=45.0,
        description="Maximum Deep Agents execution time allowed for one assistant turn",
    )
    max_catalog_searches_per_turn: int = Field(
        default=3,
        description="Maximum distinct catalog taxonomy scopes allowed per turn",
    )
    max_product_detail_reads_per_turn: int = Field(
        default=2,
        description="Maximum product-detail tool calls allowed for one assistant turn",
    )
    grounding_rewrite_enabled: bool = Field(
        default=True,
        description=(
            "Whether to run a final grounded response editor over Deep Agents "
            "drafts that have tool evidence."
        ),
    )
    grounding_rewrite_max_evidence_chars: int = Field(
        default=12000,
        description="Maximum tool evidence characters passed to the grounding editor",
    )
    expose_agent_diagnostics: bool = Field(
        default=False,
        description=(
            "Whether shopper query responses expose detailed agent diagnostics. "
            "Enable only on a trusted operator or evaluation surface."
        ),
    )
    catalog_search_timeout_seconds: Optional[float] = Field(
        default=None,
        description=(
            "Optional HTTP timeout for catalog search requests. None preserves "
            "the previous no-timeout behavior for slower remote embedding calls."
        ),
    )
    weather: WeatherConfig = Field(default_factory=WeatherConfig)
    multimodal: bool = Field(..., description="Whether multimodal features are enabled")
    media_input: MediaInputConfig = Field(default_factory=MediaInputConfig)
    
    # Safety Configuration
    guardrails_enabled: bool = Field(
        default=True,
        description="Default guardrails setting for requests that omit it",
    )
    unsafe_message: str = Field(..., description="Message to display for unsafe content")
    
    @validator('llm_port', 'retriever_port', 'memory_port', 'rails_port')
    def validate_urls(cls, v):
        """Validate that URLs are properly formatted."""
        if not v.startswith(('http://', 'https://')):
            raise ValueError(f"URL must start with http:// or https://: {v}")
        return v

    @validator("vlm_port")
    def validate_optional_vlm_url(cls, v):
        if v is not None and not v.startswith(("http://", "https://")):
            raise ValueError(f"URL must start with http:// or https://: {v}")
        return v
    
    @validator('memory_length')
    def validate_memory_length(cls, v):
        """Validate memory length is positive."""
        if v <= 0:
            raise ValueError("memory_length must be positive")
        return v
    
    @validator('top_k_retrieve')
    def validate_top_k(cls, v):
        """Validate top_k_retrieve is positive."""
        if v <= 0:
            raise ValueError("top_k_retrieve must be positive")
        return v

    @validator('deepagents_recursion_limit')
    def validate_deepagents_recursion_limit(cls, v):
        """Validate Deep Agents recursion limit is positive."""
        if v <= 0:
            raise ValueError("deepagents_recursion_limit must be positive")
        return v

    @validator('deepagents_execution_timeout_seconds')
    def validate_deepagents_execution_timeout_seconds(cls, v):
        """Validate Deep Agents execution timeout is positive."""
        if not math.isfinite(v) or v <= 0:
            raise ValueError(
                "deepagents_execution_timeout_seconds must be finite and positive"
            )
        return v

    @validator('max_catalog_searches_per_turn')
    def validate_max_catalog_searches_per_turn(cls, v):
        """Validate the per-turn catalog taxonomy-scope budget is positive."""
        if v <= 0:
            raise ValueError("max_catalog_searches_per_turn must be positive")
        return v

    @validator('max_product_detail_reads_per_turn')
    def validate_max_product_detail_reads_per_turn(cls, v):
        """Validate per-turn product-detail read cap is positive."""
        if v <= 0:
            raise ValueError("max_product_detail_reads_per_turn must be positive")
        return v

    @validator('grounding_rewrite_max_evidence_chars')
    def validate_grounding_rewrite_max_evidence_chars(cls, v):
        """Validate grounding editor evidence window size."""
        if v <= 0:
            raise ValueError("grounding_rewrite_max_evidence_chars must be positive")
        return v

    @validator('catalog_search_timeout_seconds')
    def validate_catalog_search_timeout(cls, v):
        """Validate optional catalog search timeout."""
        if v is not None and v <= 0:
            raise ValueError("catalog_search_timeout_seconds must be positive")
        return v
    
    @validator('agent_choices')
    def validate_lists_not_empty(cls, v):
        """Validate that lists are not empty."""
        if not v:
            raise ValueError("List cannot be empty")
        return v

    @model_validator(mode="after")
    def validate_summary_context_budget(self):
        summary_budget = self.conversation_summary.max_output_chars
        input_budget = max(1000, self.memory_length)
        if (
            summary_budget + _MIN_CONVERSATION_SUMMARY_SOURCE_HEADROOM
            > input_budget
        ):
            raise ValueError(
                "conversation summary output must leave at least "
                f"{_MIN_CONVERSATION_SUMMARY_SOURCE_HEADROOM} characters "
                "for compaction source input"
            )
        return self
    
    class Config:
        """Pydantic configuration."""
        extra = "forbid"  # Prevent additional fields
        validate_assignment = True  # Validate when attributes are set


def load_config(config_path: Optional[str] = None) -> ChainServerConfig:
    """
    Load service configuration and inject the configured app LLM endpoint.
    
    Args:
        config_path: Optional path to config file. If None, uses default path.
        
    Returns:
        ChainServerConfig: The loaded configuration
        
    Raises:
        FileNotFoundError: If config file is not found
        ValueError: If config validation fails
    """
    if config_path is None:
        config_root = Path(os.environ.get("SHARED_CONFIG_ROOT", "/app/shared/configs"))
        config_path = str(config_root / "chain_server" / "config.yaml")
    
    config_data = load_config_data(config_path)
    env_overrides = {
        "retriever_port": os.environ.get("CATALOG_RETRIEVER_URL"),
        "memory_port": os.environ.get("MEMORY_RETRIEVER_URL"),
        "rails_port": os.environ.get("RAILS_URL"),
        "catalog_search_timeout_seconds": os.environ.get("CATALOG_SEARCH_TIMEOUT_SECONDS"),
        "deepagents_recursion_limit": os.environ.get("DEEPAGENTS_RECURSION_LIMIT"),
        "deepagents_execution_timeout_seconds": os.environ.get(
            "DEEPAGENTS_EXECUTION_TIMEOUT_SECONDS"
        ),
        "max_catalog_searches_per_turn": os.environ.get("MAX_CATALOG_SEARCHES_PER_TURN"),
        "max_product_detail_reads_per_turn": os.environ.get(
            "MAX_PRODUCT_DETAIL_READS_PER_TURN"
        ),
        "grounding_rewrite_enabled": _env_bool("GROUNDING_REWRITE_ENABLED"),
        "grounding_rewrite_max_evidence_chars": os.environ.get(
            "GROUNDING_REWRITE_MAX_EVIDENCE_CHARS"
        ),
        "expose_agent_diagnostics": _env_bool("EXPOSE_AGENT_DIAGNOSTICS"),
        "guardrails_enabled": _env_bool("GUARDRAILS_ENABLED"),
    }
    config_data.update(
        {
            key: value
            for key, value in env_overrides.items()
            if value is not None and value != ""
        }
    )
    weather_enabled = _env_bool("WEATHER_ENABLED")
    if weather_enabled is not None:
        weather_data = dict(config_data.get("weather") or {})
        weather_data["enabled"] = weather_enabled
        config_data["weather"] = weather_data

    config_root = Path(os.environ.get("SHARED_CONFIG_ROOT", str(Path(config_path).parents[1])))
    model_config = resolve_model_config(config_root=config_root)
    validate_model_config(model_config, roles=("app_llm",))
    app_llm = model_config.require("app_llm")
    vlm = model_config.get("vlm")
    if vlm is not None and not vlm.disabled:
        validate_model_config(model_config, roles=("vlm",))
    config_data.update(
        {
            "llm_port": app_llm.base_url,
            "llm_name": app_llm.model,
            "llm_api_key_env": app_llm.api_key_env,
            "llm_api_key_required": app_llm.api_key_required,
            "vlm_enabled": bool(vlm is not None and not vlm.disabled),
            "vlm_port": vlm.base_url if vlm is not None and not vlm.disabled else None,
            "vlm_name": vlm.model if vlm is not None and not vlm.disabled else None,
            "vlm_api_key_env": (
                vlm.api_key_env if vlm is not None and not vlm.disabled else None
            ),
            "vlm_api_key_required": (
                vlm.api_key_required if vlm is not None and not vlm.disabled else False
            ),
        }
    )
    
    # Create Pydantic config instance
    try:
        return ChainServerConfig(**config_data)
    except Exception as e:
        raise ValueError(f"Configuration validation failed: {e}")


def _env_bool(name: str) -> bool | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be one of true/false, yes/no, on/off, or 1/0; got {value!r}"
    )
