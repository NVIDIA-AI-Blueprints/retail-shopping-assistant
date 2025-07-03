from nemoguardrails import RailsConfig, LLMRails
import os
import yaml
import logging
import sys
from openai import OpenAI

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class BaseRails():

    async def call_input_content_rails(self, user_input: str):
        pass

    async def call_input_topic_rails(self, user_input: str):
        pass

    async def call_output_content_rails(self, user_input: str):
        pass

# Define the GuardRails class
class GuardRails(BaseRails):
    def __init__(self, config_path: str):
        self.config = RailsConfig.from_path(config_path)
        self.app = LLMRails(self.config)

    async def call_input_content_rails(self, user_input: str):
        """Generate a response to user input using the LLM"""
        options = {
            "rails": ["input"], 
            "output_vars": True, 
            "log": {
                "activated_rails": True,
                "llm_calls": True,
                "internal_events": True,
                "colang_history": True
            }
        }
        messages = [{"role": "user", "content": user_input}]
        response = await self.app.generate_async(messages=messages, options=options)
        return response

    async def call_output_content_rails(self, bot_response: str):
        """Generate a response to user input using the LLM"""
        options = {"rails": ["output"]}
        messages = [{"role": "user", "content": ""}, {"role": "assistant", "content": bot_response}]
        response = await self.app.generate_async(messages=messages, options=options)
        return response
    
class GenericRails(BaseRails):
    def __init__(self, config_path: str):
        # Load configuration
        config_path = os.path.join("/app", "app", "llm_config.yaml")
        if not os.path.exists(config_path):
            logging.error(f"Config file not found at {config_path}")
            raise FileNotFoundError(f"Config file not found at {config_path}")

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        logging.info(f"PlannerAgent | __init__ | Initializing with llm_name={self.config.llm_name}, llm_port={self.config.llm_port}")
        self.app = OpenAI(
            base_url=self.config.llm_port,
            api_key=os.environ.get("NVIDIA_API_KEY")
        )

    async def call_input_content_rails(self, user_input: str):
        response = self.app.chat.completions.create(
            model=self.config.llm_name,
            messages=[
                {
                    "role": "system",
                    "content": self.config.system_prompt
                },
                {
                    "role": "user",
                    "content": f"Customer Query: {user_input}"
                }
            ],
            temperature=0.0,
            max_tokens=100
        )
        return response

    async def call_output_content_rails(self, bot_response: str):
        """Generate a response to user input using the LLM"""
        options = {"rails": ["output"]}
        messages = [{"role": "user", "content": ""}, {"role": "assistant", "content": bot_response}]
        response = await self.app.generate_async(messages=messages, options=options)
        return response
    
guardRails = GuardRails("config")

class Rails():
    def getGuardRails(self):
        return guardRails