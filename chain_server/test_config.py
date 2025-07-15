#!/usr/bin/env python3
"""
Test script for the centralized configuration system.

This script tests the ChainServerConfig class and its integration
with the config override system.
"""

import os
import sys
import tempfile
import yaml

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.config import ChainServerConfig, load_config, get_config


def test_config_validation():
    """Test that the config validation works correctly."""
    print("Testing config validation...")
    
    # Test valid config
    valid_config = {
        "llm_port": "http://localhost:8000/v1",
        "llm_name": "meta/llama-3.1-70b-instruct",
        "retriever_port": "http://localhost:8010",
        "memory_port": "http://localhost:8011",
        "rails_port": "http://localhost:8012",
        "routing_prompt": "You are a routing assistant.",
        "chatter_prompt": "You are a helpful assistant.",
        "categories": ["bag", "shoes"],
        "agent_choices": ["cart", "retriever", "chatter"],
        "memory_length": 16384,
        "top_k_retrieve": 4,
        "multimodal": True,
        "unsafe_message": "Sorry, I cannot help with that."
    }
    
    try:
        config = ChainServerConfig(**valid_config)
        print("✓ Valid config created successfully")
        print(f"  - LLM Port: {config.llm_port}")
        print(f"  - Categories: {config.categories}")
        print(f"  - Memory Length: {config.memory_length}")
    except Exception as e:
        print(f"✗ Valid config failed: {e}")
        return False
    
    # Test invalid config (missing required field)
    invalid_config = valid_config.copy()
    del invalid_config["llm_port"]
    
    try:
        config = ChainServerConfig(**invalid_config)
        print("✗ Invalid config should have failed")
        return False
    except Exception as e:
        print("✓ Invalid config correctly rejected")
    
    # Test invalid URL
    invalid_url_config = valid_config.copy()
    invalid_url_config["llm_port"] = "not-a-url"
    
    try:
        config = ChainServerConfig(**invalid_url_config)
        print("✗ Invalid URL should have failed")
        return False
    except Exception as e:
        print("✓ Invalid URL correctly rejected")
    
    return True


def test_config_loading():
    """Test that config loading works with override system."""
    print("\nTesting config loading...")
    
    # Create a temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        base_config = {
            "llm_port": "http://localhost:8000/v1",
            "llm_name": "meta/llama-3.1-70b-instruct",
            "retriever_port": "http://localhost:8010",
            "memory_port": "http://localhost:8011",
            "rails_port": "http://localhost:8012",
            "routing_prompt": "You are a routing assistant.",
            "chatter_prompt": "You are a helpful assistant.",
            "categories": ["bag", "shoes"],
            "agent_choices": ["cart", "retriever", "chatter"],
            "memory_length": 16384,
            "top_k_retrieve": 4,
            "multimodal": True,
            "unsafe_message": "Sorry, I cannot help with that."
        }
        yaml.dump(base_config, f)
        config_path = f.name
    
    try:
        # Test loading without override
        config = load_config(config_path)
        print("✓ Config loaded successfully")
        print(f"  - LLM Port: {config.llm_port}")
        
        # Test getting config after loading
        retrieved_config = get_config()
        print("✓ Config retrieved successfully")
        print(f"  - Same LLM Port: {retrieved_config.llm_port}")
        
        if config.llm_port == retrieved_config.llm_port:
            print("✓ Config instances match")
        else:
            print("✗ Config instances don't match")
            return False
            
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False
    finally:
        # Clean up
        os.unlink(config_path)
    
    return True


def test_config_usage():
    """Test that config can be used in agent-like scenarios."""
    print("\nTesting config usage in agents...")
    
    # Simulate agent initialization
    config = ChainServerConfig(
        llm_port="http://localhost:8000/v1",
        llm_name="meta/llama-3.1-70b-instruct",
        retriever_port="http://localhost:8010",
        memory_port="http://localhost:8011",
        rails_port="http://localhost:8012",
        routing_prompt="You are a routing assistant.",
        chatter_prompt="You are a helpful assistant.",
        categories=["bag", "shoes", "dress"],
        agent_choices=["cart", "retriever", "chatter"],
        memory_length=16384,
        top_k_retrieve=4,
        multimodal=True,
        unsafe_message="Sorry, I cannot help with that."
    )
    
    # Simulate planner agent usage
    print("✓ Planner agent config:")
    print(f"  - Agent choices: {config.agent_choices}")
    print(f"  - Routing prompt length: {len(config.routing_prompt)}")
    
    # Simulate retriever agent usage
    print("✓ Retriever agent config:")
    print(f"  - Categories: {config.categories}")
    print(f"  - Top K retrieve: {config.top_k_retrieve}")
    print(f"  - Retriever port: {config.retriever_port}")
    
    # Simulate chatter agent usage
    print("✓ Chatter agent config:")
    print(f"  - Memory length: {config.memory_length}")
    print(f"  - Chatter prompt length: {len(config.chatter_prompt)}")
    
    # Simulate cart agent usage
    print("✓ Cart agent config:")
    print(f"  - Memory port: {config.memory_port}")
    
    # Simulate graph usage
    print("✓ Graph config:")
    print(f"  - Rails port: {config.rails_port}")
    print(f"  - Unsafe message: {config.unsafe_message}")
    
    return True


def main():
    """Run all tests."""
    print("Testing ChainServerConfig centralized configuration system")
    print("=" * 60)
    
    tests = [
        test_config_validation,
        test_config_loading,
        test_config_usage
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"✗ Test {test.__name__} failed")
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 60)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The centralized config system is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 