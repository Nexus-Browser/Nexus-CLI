#!/usr/bin/env python3
"""
Test script for the new iLLuMinator-powered Nexus CLI
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from model.illuminator_api import iLLuMinatorAPI
    print("✓ iLLuMinator API import successful")
    
    # Test API initialization
    api = iLLuMinatorAPI()
    print("✓ iLLuMinator API initialized")
    
    # Test model info
    info = api.get_model_info()
    print(f"✓ Model info: {info['name']} v{info['version']} by {info['author']}")
    
    # Test basic functionality (this will make a real API call)
    print("Testing API connection...")
    if api.is_available():
        print("✓ iLLuMinator API is available!")
        
        # Test simple code generation
        print("Testing code generation...")
        code = api.generate_code("create a function that adds two numbers", "python")
        if not code.startswith("Error:"):
            print("Code generation working!")
            print("Generated code preview:")
            print(code[:200] + "..." if len(code) > 200 else code)
        else:
            print(f"WARNING: Code generation test failed: {code}")
    else:
        print("WARNING: iLLuMinator API connection failed - check your API key")
    
    print("\n" + "="*50)
    print("Now testing full Nexus CLI...")
    
    from model.nexus_model import NexusModel
    print("✓ NexusModel import successful")
    
    model = NexusModel()
    print("✓ NexusModel initialized")
    
    if model.is_available():
        print("Nexus model is available!")
        print("Ready to run Nexus CLI with iLLuMinator-4.7B!")
    else:
        print("WARNING: Nexus model not available")

except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Make sure all dependencies are installed: pip install -r requirements.txt")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "="*50)
print("To start Nexus CLI, run: python nexus_cli.py")
