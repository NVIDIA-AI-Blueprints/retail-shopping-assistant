# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import yaml
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def apply_endpoint_overrides(config, config_dir: str = "/app/shared/configs"):
    """
    Apply model and parameter overrides to RailsConfig when configured.

    Args:
        config: RailsConfig object to modify
        config_dir: Directory containing config files
    """
    override_file = os.environ.get("CONFIG_OVERRIDE")
    
    if not override_file:
        logger.info("Using local endpoints for guardrails configuration")
        return

    # Load the override config file.
    override_path = os.path.join(config_dir, override_file)

    if not os.path.exists(override_path):
        logger.warning(f"Guardrails override config file not found at {override_path}")
        return

    logger.info(f"Loading guardrails override config from {override_path}")
    
    with open(override_path, 'r') as f:
        override_config = yaml.safe_load(f)

    if 'models' in override_config:
        for model_config in override_config['models']:
            model_type = model_config.get('type')
            if not model_type:
                continue

            for model in config.models:
                if model.type != model_type:
                    continue

                override_model = model_config.get('model')
                if override_model:
                    model.model = override_model
                    logger.info(f"Updated {model_type} model to {override_model}")

                override_parameters = model_config.get('parameters', {})
                if override_parameters:
                    model.parameters.update(override_parameters)
                    logger.info(f"Updated {model_type} model parameters")
                break

    logger.info("Applied endpoint overrides to guardrails configuration")
