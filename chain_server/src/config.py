"""
Centralized configuration management for the chain server.

This module provides a Pydantic-based configuration class that loads
configuration from YAML files with optional override support.
"""

import os
import yaml
import logging
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


def load_config_with_override(base_config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file with optional override support.
    
    Args:
        base_config_path: Path to the base configuration file
        
    Returns:
        Dictionary containing the merged configuration
        
    Environment Variables:
        CONFIG_OVERRIDE: If set, specifies the override config file name
                        (e.g., "config-local.yaml" or "config-build.yaml")
    """
    # Load base config
    if not os.path.exists(base_config_path):
        logger.error(f"Base config file not found at {base_config_path}")
        raise FileNotFoundError(f"Base config file not found at {base_config_path}")

    with open(base_config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Check for override config
    override_file = os.environ.get("CONFIG_OVERRIDE")
    if override_file:
        # Construct override path (same directory as base config)
        base_dir = os.path.dirname(base_config_path)
        override_path = os.path.join(base_dir, override_file)
        
        if os.path.exists(override_path):
            logger.info(f"Loading override config from {override_path}")
            with open(override_path, "r") as f:
                override_config = yaml.safe_load(f)
            
            # Merge override config into base config
            config.update(override_config)
            logger.info(f"Config override applied from {override_file}")
        else:
            logger.warning(f"Override config file not found at {override_path}")
    else:
        logger.info("No config override specified, using base config only")
    
    return config


class ChainServerConfig(BaseModel):
    """Configuration class for the chain server application."""
    
    # LLM Configuration
    llm_port: str = Field(..., description="LLM service endpoint URL")
    llm_name: str = Field(..., description="LLM model name")
    
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
    Load configuration from YAML file with optional override support.
    
    Args:
        config_path: Optional path to config file. If None, uses default path.
        
    Returns:
        ChainServerConfig: The loaded configuration
        
    Raises:
        FileNotFoundError: If config file is not found
        ValueError: If config validation fails
    """
    if config_path is None:
        config_path = "/app/shared/configs/chain_server/config.yaml"
    
    # Load raw config data with override support
    config_data = load_config_with_override(config_path)
    
    # Create Pydantic config instance
    try:
        return ChainServerConfig(**config_data)
    except Exception as e:
        raise ValueError(f"Configuration validation failed: {e}") 