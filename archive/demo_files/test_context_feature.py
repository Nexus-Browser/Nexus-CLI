#!/usr/bin/env python3
"""
Test script to demonstrate the context feature in Nexus CLI
"""

import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nexus_cli import IntelligentNexusCLI

def test_context_feature():
    """Test the context feature functionality."""
    print("=== Testing Nexus CLI Context Feature ===\n")
    
    # Initialize CLI
    cli = IntelligentNexusCLI()
    
    # Test 1: Show empty context
    print("1. Testing empty context:")
    result = cli._handle_show_context([])
    print(f"   Result: {result}")
    
    # Test 2: Read a file to add to context
    print("\n2. Testing read command with context:")
    result = cli._handle_read_file(["setup.py"])
    print(f"   Result: {result}")
    
    # Test 3: Show context with file
    print("\n3. Testing context display after reading file:")
    result = cli._handle_show_context([])
    print(f"   Result: {result}")
    
    # Test 4: Test natural language with context (without API call to avoid rate limits)
    print("\n4. Testing context-enhanced prompt preparation:")
    enhanced_prompt = cli._prepare_context_enhanced_prompt("What is this project about?")
    print(f"   Enhanced prompt length: {len(enhanced_prompt)} characters")
    print(f"   Contains context: {'CONTEXT:' in enhanced_prompt}")
    print(f"   Contains setup.py: {'setup.py' in enhanced_prompt}")
    
    # Test 5: Add another file
    print("\n5. Testing reading another file:")
    result = cli._handle_read_file(["README.md"])
    print(f"   Result: {result}")
    
    # Test 6: Show updated context
    print("\n6. Testing context with multiple files:")
    result = cli._handle_show_context([])
    print(f"   Result: {result}")
    
    # Test 7: Clear context
    print("\n7. Testing clear context:")
    result = cli._handle_clear_context([])
    print(f"   Result: {result}")
    
    # Test 8: Show empty context again
    print("\n8. Testing context after clearing:")
    result = cli._handle_show_context([])
    print(f"   Result: {result}")
    
    print("\n=== Context Feature Test Complete! ===")

if __name__ == "__main__":
    test_context_feature()
