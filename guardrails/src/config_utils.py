# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os

from shared.model_config import resolve_model_config, validate_model_config

logger = logging.getLogger(__name__)


def apply_model_config(config, config_dir: str = "/app/shared/configs/rails"):
    """Apply shared model endpoint config to a RailsConfig object."""

    config_root = os.environ.get("SHARED_CONFIG_ROOT") or os.path.dirname(config_dir.rstrip("/"))
    model_config = resolve_model_config(config_root=config_root)
    validate_model_config(model_config, roles=("app_llm", "content_safety", "topic_control"))

    role_by_type = {
        "main": model_config.require("app_llm"),
        "content_safety": model_config.require("content_safety"),
        "topic_control": model_config.require("topic_control"),
    }

    for model in config.models:
        endpoint = role_by_type.get(model.type)
        if not endpoint:
            continue

        model.model = endpoint.model
        if getattr(model, "parameters", None) is None:
            model.parameters = {}
        model.parameters["base_url"] = endpoint.base_url

        api_key = os.environ.get(endpoint.api_key_env, "") if endpoint.api_key_env else ""
        if api_key:
            os.environ["NVIDIA_API_KEY"] = api_key
        elif not endpoint.api_key_env:
            os.environ.setdefault("NVIDIA_API_KEY", "not-needed")

        logger.info(
            "Applied model config role %s to guardrails model type %s",
            endpoint.role,
            model.type,
        )
