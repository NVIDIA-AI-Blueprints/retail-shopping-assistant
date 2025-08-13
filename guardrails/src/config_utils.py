# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import yaml
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def apply_endpoint_overrides(config, config_dir: str = "/app/shared/configs"):
    """
    Apply endpoint overrides to the RailsConfig if CONFIG_OVERRIDE is set.
    
    Args:
        config: RailsConfig object to modify
        config_dir: Directory containing config files
    """
    override_file = os.environ.get("CONFIG_OVERRIDE")
    
    if not override_file:
        logger.info("Using local endpoints for guardrails configuration")
        return
    
    # Load the override config file to get the base_url values
    override_path = os.path.join(config_dir, override_file)
    
    if not os.path.exists(override_path):
        logger.warning(f"Guardrails override config file not found at {override_path}")
        return
    
    logger.info(f"Loading guardrails override config from {override_path}")
    
    with open(override_path, 'r') as f:
        override_config = yaml.safe_load(f)
    
    # Extract base_url values from the override config
    if 'models' in override_config:
        for model_config in override_config['models']:
            if 'type' in model_config and 'parameters' in model_config:
                model_type = model_config['type']
                base_url = model_config['parameters'].get('base_url')
                
                if base_url:
                    # Update the corresponding model in RailsConfig
                    for model in config.models:
                        if model.type == model_type:
                            model.parameters['base_url'] = base_url
                            logger.info(f"Updated {model_type} base_url to {base_url}")
                            break

    logger.info("Applied endpoint overrides to guardrails configuration") 