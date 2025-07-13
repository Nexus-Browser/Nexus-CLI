#!/usr/bin/env python3
"""
Test script for Nexus AI model.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.nexus_model import NexusModel
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def test_model():
    """Test the Nexus model."""
    print("Testing Nexus AI Model...")
    
    try:
        # Initialize model
        print("Loading model...")
        model = NexusModel()
        
        if not model.model:
            print("❌ Model failed to load!")
            return False
        
        print("✅ Model loaded successfully!")
        
        # Test chat generation
        print("\nTesting chat generation...")
        chat_response = model.generate_response("Hello, how are you?")
        print(f"Chat response: {chat_response}")
        
        # Test code generation
        print("\nTesting code generation...")
        code_response = model.generate_code("create a function to add two numbers", "python")
        print(f"Code response: {code_response}")
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_model()
    sys.exit(0 if success else 1) 