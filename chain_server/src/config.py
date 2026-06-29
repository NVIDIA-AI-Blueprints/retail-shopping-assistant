# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Centralized configuration management for the chain server."""

import os
from pathlib import Path
import yaml
import logging
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator

from shared.model_config import resolve_model_config, validate_model_config

logger = logging.getLogger(__name__)


def load_config_data(base_config_path: str) -> Dict[str, Any]:
    """Load a service config YAML file."""

    if not os.path.exists(base_config_path):
        logger.error(f"Base config file not found at {base_config_path}")
        raise FileNotFoundError(f"Base config file not found at {base_config_path}")

    with open(base_config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


class ChainServerConfig(BaseModel):
    """Configuration class for the chain server application."""
    
    # LLM Configuration
    llm_port: str = Field(..., description="LLM service endpoint URL")
    llm_name: str = Field(..., description="LLM model name")
    llm_api_key_env: Optional[str] = Field(default="LLM_API_KEY", description="LLM API key environment variable")
    llm_api_key_required: bool = Field(default=True, description="Whether the LLM key must be present")
    
    # Service Endpoints
    retriever_port: str = Field(..., description="Catalog retriever service endpoint")
    memory_port: str = Field(..., description="Memory retriever service endpoint")
    rails_port: str = Field(..., description="Guardrails service endpoint")
    
    # Prompts
    routing_prompt: str = Field(..., description="System prompt for routing queries to appropriate agents")
    chatter_prompt: str = Field(..., description="System prompt for general conversation")
    
    # Product Configuration
    categories: List[str] = Field(..., description="List of product categories")
    agent_choices: List[str] = Field(..., description="Available agent types")
    
    # Performance Configuration
    memory_length: int = Field(..., description="Maximum memory length for context")
    top_k_retrieve: int = Field(..., description="Number of top results to retrieve")
    multimodal: bool = Field(..., description="Whether multimodal features are enabled")
    
    # Safety Configuration
    unsafe_message: str = Field(..., description="Message to display for unsafe content")
    
    @validator('llm_port', 'retriever_port', 'memory_port', 'rails_port')
    def validate_urls(cls, v):
        """Validate that URLs are properly formatted."""
        if not v.startswith(('http://', 'https://')):
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
    
    @validator('categories', 'agent_choices')
    def validate_lists_not_empty(cls, v):
        """Validate that lists are not empty."""
        if not v:
            raise ValueError("List cannot be empty")
        return v
    
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
    }
    config_data.update({key: value for key, value in env_overrides.items() if value})

    config_root = Path(os.environ.get("SHARED_CONFIG_ROOT", str(Path(config_path).parents[1])))
    model_config = resolve_model_config(config_root=config_root)
    validate_model_config(model_config, roles=("app_llm",))
    app_llm = model_config.require("app_llm")
    config_data.update(
        {
            "llm_port": app_llm.base_url,
            "llm_name": app_llm.model,
            "llm_api_key_env": app_llm.api_key_env,
            "llm_api_key_required": app_llm.api_key_required,
        }
    )
    
    # Create Pydantic config instance
    try:
        return ChainServerConfig(**config_data)
    except Exception as e:
        raise ValueError(f"Configuration validation failed: {e}")
