# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemoguardrails import RailsConfig, LLMRails
import logging
from config_utils import apply_endpoint_overrides

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

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
config_path = "/app/shared/configs/rails"
guardRails = GuardRails(config_path)

class Rails():
    def getGuardRails(self):
        return guardRails