"""
Configuration validation for the Shopping Assistant.

This module provides utilities to validate the application configuration
and ensure all required services are properly configured.
"""
import os
import yaml
import logging
import requests
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ConfigValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]


class ConfigValidator:
    """Validates the shopping assistant configuration."""
    
    def __init__(self, config_path: str = "/app/app/config.yaml"):
        """
        Initialize the config validator.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config = None
    
    def load_config(self) -> bool:
        """
        Load the configuration file.
        
        Returns:
            True if config loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(self.config_path):
                logger.error(f"Config file not found: {self.config_path}")
                return False
            
            with open(self.config_path, "r") as f:
                self.config = yaml.safe_load(f)
            
            logger.info("Configuration loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return False
    
    def validate_required_fields(self) -> List[str]:
        """
        Validate that all required configuration fields are present.
        
        Returns:
            List of validation errors
        """
        errors = []
        required_fields = [
            "llm_name",
            "llm_port",
            "retriever_port",
            "memory_port",
            "rails_port",
            "routing_prompt",
            "chatter_prompt",
            "agent_choices",
            "memory_length",
            "top_k_retrieve",
            "multimodal",
            "unsafe_message"
        ]
        
        for field in required_fields:
            if field not in self.config:
                errors.append(f"Missing required field: {field}")
            elif self.config[field] is None:
                errors.append(f"Required field is None: {field}")
        
        return errors
    
    def validate_agent_choices(self) -> List[str]:
        """
        Validate that agent choices are properly configured.
        
        Returns:
            List of validation errors
        """
        errors = []
        
        if "agent_choices" not in self.config:
            errors.append("Missing agent_choices configuration")
            return errors
        
        choices = self.config["agent_choices"]
        if not isinstance(choices, list):
            errors.append("agent_choices must be a list")
            return errors
        
        expected_choices = ["cart", "retriever", "visualizer", "chatter"]
        for choice in expected_choices:
            if choice not in choices:
                errors.append(f"Missing required agent choice: {choice}")
        
        return errors
    
    def validate_prompts(self) -> List[str]:
        """
        Validate that prompts are properly configured.
        
        Returns:
            List of validation errors
        """
        errors = []
        
        required_prompts = ["routing_prompt", "chatter_prompt"]
        for prompt in required_prompts:
            if prompt not in self.config:
                errors.append(f"Missing required prompt: {prompt}")
            elif not self.config[prompt].strip():
                errors.append(f"Empty prompt: {prompt}")
        
        return errors
    
    def validate_ports(self) -> List[str]:
        """
        Validate that service ports are properly configured.
        
        Returns:
            List of validation errors
        """
        errors = []
        
        port_fields = ["retriever_port", "memory_port", "rails_port"]
        for field in port_fields:
            if field not in self.config:
                errors.append(f"Missing port configuration: {field}")
            else:
                port_url = self.config[field]
                if not isinstance(port_url, str) or not port_url.startswith("http"):
                    errors.append(f"Invalid port URL format: {field} = {port_url}")
        
        return errors
    
    def validate_environment(self) -> List[str]:
        """
        Validate that required environment variables are set.
        
        Returns:
            List of validation errors
        """
        errors = []
        
        required_env_vars = ["LLM_API_KEY"]
        for var in required_env_vars:
            if not os.environ.get(var):
                errors.append(f"Missing environment variable: {var}")
        
        return errors
    
    def check_service_connectivity(self) -> List[str]:
        """
        Check connectivity to required services.
        
        Returns:
            List of connectivity errors
        """
        errors = []
        
        services = {
            "retriever": self.config.get("retriever_port"),
            "memory": self.config.get("memory_port"),
            "guardrails": self.config.get("rails_port")
        }
        
        for service_name, service_url in services.items():
            if not service_url:
                errors.append(f"No URL configured for {service_name} service")
                continue
            
            try:
                response = requests.get(f"{service_url}/health", timeout=5)
                if response.status_code != 200:
                    errors.append(f"{service_name} service health check failed: {response.status_code}")
            except requests.RequestException as e:
                errors.append(f"Cannot connect to {service_name} service: {e}")
        
        return errors
    
    def validate(self) -> ConfigValidationResult:
        """
        Perform comprehensive configuration validation.
        
        Returns:
            Configuration validation result
        """
        errors = []
        warnings = []
        
        # Load configuration
        if not self.load_config():
            return ConfigValidationResult(False, ["Failed to load configuration"], [])
        
        # Validate required fields
        errors.extend(self.validate_required_fields())
        
        # Validate agent choices
        errors.extend(self.validate_agent_choices())
        
        # Validate prompts
        errors.extend(self.validate_prompts())
        
        # Validate ports
        errors.extend(self.validate_ports())
        
        # Validate environment
        errors.extend(self.validate_environment())
        
        # Check service connectivity (warnings only)
        connectivity_errors = self.check_service_connectivity()
        warnings.extend(connectivity_errors)
        
        # Additional validation checks
        if self.config:
            # Check memory length
            memory_length = self.config.get("memory_length", 0)
            if memory_length <= 0:
                errors.append("memory_length must be positive")
            elif memory_length > 32768:
                warnings.append("Large memory_length may impact performance")
            
            # Check top_k_retrieve
            top_k = self.config.get("top_k_retrieve", 0)
            if top_k <= 0:
                errors.append("top_k_retrieve must be positive")
            elif top_k > 20:
                warnings.append("Large top_k_retrieve may impact performance")
        
        is_valid = len(errors) == 0
        
        return ConfigValidationResult(is_valid, errors, warnings)


def validate_config(config_path: Optional[str] = None) -> ConfigValidationResult:
    """
    Convenience function to validate configuration.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Configuration validation result
    """
    validator = ConfigValidator(config_path or "/app/app/config.yaml")
    return validator.validate()