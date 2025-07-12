#!/usr/bin/env python3
"""
Test script to debug the Nexus model output.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.nexus_model import NexusModel

def test_model():
    """Test the model with various inputs."""
    print("Loading Nexus model...")
    model = NexusModel()
    
    # Test conversation
    print("\n=== Testing Conversation ===")
    conversation_prompts = [
        "Hello, how are you?",
        "What can you help me with?",
        "What is a variable?",
        "How do I create a function?"
    ]
    
    for prompt in conversation_prompts:
        print(f"\nPrompt: {prompt}")
        response = model.generate_response(f"User: {prompt}\nAssistant:", max_length=128, temperature=0.7)
        print(f"Response: {response}")
        print("-" * 50)
    
    # Test code generation
    print("\n=== Testing Code Generation ===")
    code_prompts = [
        "Create a function to add two numbers",
        "Write a simple calculator class",
        "Create a function to check if a number is even"
    ]
    
    for prompt in code_prompts:
        print(f"\nPrompt: {prompt}")
        code = model.generate_code(prompt, "python")
        print(f"Generated Code:\n{code}")
        print("-" * 50)

if __name__ == "__main__":
    test_model() 