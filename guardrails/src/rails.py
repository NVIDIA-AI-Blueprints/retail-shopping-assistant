# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import re
from typing import Any

from nemoguardrails import RailsConfig, LLMRails

from config_utils import apply_endpoint_overrides

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

_PARSE_FAILURE = "content safety response parsing failed"


def _safety_categories(value: Any) -> list[str]:
    if isinstance(value, str):
        return [category.strip() for category in value.split(",") if category.strip()]
    if isinstance(value, list):
        return [str(category).strip() for category in value if str(category).strip()]
    return []


def _parse_content_safety(response: str, field: str) -> list[bool | str]:
    """Parse JSON or line-oriented safety labels, failing closed on ambiguity."""
    result: Any = None
    categories: list[str] = []

    try:
        parsed = json.loads(response)
    except (json.JSONDecodeError, TypeError):
        parsed = None

    if isinstance(parsed, dict):
        result = parsed.get(field)
        categories = _safety_categories(parsed.get("Safety Categories"))
    elif isinstance(response, str):
        ratings = {
            match.lower()
            for match in re.findall(
                rf"^\s*{re.escape(field)}\s*:\s*(safe|unsafe)\s*$",
                response,
                flags=re.IGNORECASE | re.MULTILINE,
            )
        }
        if "unsafe" in ratings:
            result = "unsafe"
        elif ratings == {"safe"}:
            result = "safe"

        category_match = re.search(
            r"^\s*Safety Categories\s*:\s*(.+?)\s*$",
            response,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if category_match:
            categories = _safety_categories(category_match.group(1))

    if not isinstance(result, str):
        return [False, _PARSE_FAILURE]

    rating = result.strip().lower()
    if rating == "safe":
        return [True]
    if rating == "unsafe":
        return [False, *categories]
    return [False, _PARSE_FAILURE]


def parse_user_safety(response: str) -> list[bool | str]:
    return _parse_content_safety(response, "User Safety")


def parse_response_safety(response: str) -> list[bool | str]:
    return _parse_content_safety(response, "Response Safety")

class BaseRails():

    async def call_input_content_rails(self, user_input: str):
        pass

    async def call_output_content_rails(self, user_input: str):
        pass

# Define the GuardRails class
class GuardRails(BaseRails):
    def __init__(self, config_path: str):

        # Load the base configuration
        self.config = RailsConfig.from_path(config_path)
        
        # Apply endpoint overrides if CONFIG_OVERRIDE is set
        apply_endpoint_overrides(self.config, config_path)
        
        # Initialize the LLM Rails with the modified configuration
        self.app = LLMRails(self.config)
        self.app.register_output_parser(
            parse_user_safety, "retail_parse_user_safety"
        )
        self.app.register_output_parser(
            parse_response_safety, "retail_parse_response_safety"
        )

    async def call_input_content_rails(self, user_input: str):
        """Generate a response to user input using the LLM"""
        options = {"rails": ["input"]}
        messages = [{"role": "user", "content": user_input}]
        response = await self.app.generate_async(messages=messages, options=options)
        return response

    async def call_output_content_rails(self, bot_response: str):
        """Generate a response to user input using the LLM"""
        options = {"rails": ["output"]}
        messages = [{"role": "user", "content": ""}, {"role": "assistant", "content": bot_response}]
        response = await self.app.generate_async(messages=messages, options=options)
        return response
    
# Load configuration
config_path = os.path.join(os.environ.get("SHARED_CONFIG_ROOT", "/app/shared/configs"), "rails")
guardRails = GuardRails(config_path)

class Rails():
    def getGuardRails(self):
        return guardRails
