# tests/test_rails.py
import os
import sys
PROJECT_PATH = os.getcwd()
SOURCE_PATH = os.path.join(
    PROJECT_PATH,"src"
)
sys.path.append(SOURCE_PATH)

import unittest
from unittest.mock import Mock
from nemoguardrails import RailsConfig, LLMRails
from rails import BaseRails  # assuming rails.py is in the same directory

class TestBaseRails(unittest.TestCase):
    def test_call_input_content_rails(self):
        # Create a mock implementation of the abstract method
        class MockRails(BaseRails):
            async def call_input_content_rails(self, user_input: str):
                return "Mock response"

        # Create an instance of the mock class
        rails = MockRails()

        # Test the method
        user_input = "Hello"
        response = rails.call_input_content_rails(user_input)
        self.assertEqual(response, "Mock response")

    def test_call_input_topic_rails(self):
        # Create a mock implementation of the abstract method
        class MockRails(BaseRails):
            async def call_input_topic_rails(self, user_input: str):
                return "Mock response"

        # Create an instance of the mock class
        rails = MockRails()

        # Test the method
        user_input = "Hello"
        response = rails.call_input_topic_rails(user_input)
        self.assertEqual(response, "Mock response")

    def test_call_output_content_rails(self):
        # Create a mock implementation of the abstract method
        class MockRails(BaseRails):
            async def call_output_content_rails(self, user_input: str):
                return "Mock response"

        # Create an instance of the mock class
        rails = MockRails()

        # Test the method
        user_input = "Hello"
        response = rails.call_output_content_rails(user_input)
        self.assertEqual(response, "Mock response")

if __name__ == "__main__":
    unittest.main()