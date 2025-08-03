#!/usr/bin/env python3
"""
Demo script showing Nexus CLI capabilities
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the parent directory to the path so we can import nexus_cli
sys.path.insert(0, str(Path(__file__).parent.parent))

from nexus_cli import NexusCLI

def demo_code_generation():
    """Demonstrate code generation capabilities."""
    print(" Demo: Code Generation")
    print("-" * 40)
    
    cli = NexusCLI()
    
    # Test code generation
    test_instructions = [
        "create a function to calculate fibonacci numbers",
        "write a class for a simple calculator",
        "create a function to check if a string is a palindrome"
    ]
    
    for instruction in test_instructions:
        print(f"\n Instruction: {instruction}")
        response = cli._handle_code_generation([instruction])
        print(f" Response: {response[:200]}...")

def demo_file_operations():
    """Demonstrate file operation capabilities."""
    print("\n Demo: File Operations")
    print("-" * 40)
    
    cli = NexusCLI()
    
    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("""def hello_world():
    print("Hello, World!")

def add_numbers(a, b):
    return a + b
""")
        temp_file = f.name
    
    try:
        # Test file reading
        print(f"\nReading file: {temp_file}")
        response = cli._handle_read_file([temp_file])
        print(f" {response}")
        
        # Test file writing
        new_file = "demo_output.py"
        content = 'print("This is a demo file created by Nexus CLI")'
        response = cli._handle_write_file([new_file, content])
        print(f"\n  Writing to {new_file}")
        print(f" {response}")
        
        # Clean up
        if os.path.exists(new_file):
            os.remove(new_file)
            
    finally:
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)

def demo_code_analysis():
    """Demonstrate code analysis capabilities."""
    print("\n Demo: Code Analysis")
    print("-" * 40)
    
    cli = NexusCLI()
    
    # Create a sample Python file for analysis
    sample_code = """class Calculator:
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def subtract(self, a, b):
        result = a - b
        self.history.append(f"{a} - {b} = {result}")
        return result

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(sample_code)
        temp_file = f.name
    
    try:
        # Test code analysis
        print(f"\n Analyzing file: {temp_file}")
        response = cli._handle_analyze_code([temp_file])
        print(f" {response}")
        
        # Test function extraction
        print(f"\nExtracting functions from: {temp_file}")
        response = cli._handle_extract_functions([temp_file])
        print(f" {response}")
        
        # Test class extraction
        print(f"\n  Extracting classes from: {temp_file}")
        response = cli._handle_extract_classes([temp_file])
        print(f" {response}")
        
    finally:
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)

def demo_project_management():
    """Demonstrate project management capabilities."""
    print("\n  Demo: Project Management")
    print("-" * 40)
    
    cli = NexusCLI()
    
    # Test project structure
    print("\nProject structure:")
    response = cli._handle_project_tree([])
    print(f" {response}")
    
    # Test file listing
    print("\nFile listing:")
    response = cli._handle_list_files([])
    print(f" {response}")

def demo_natural_language():
    """Demonstrate natural language processing."""
    print("\nDemo: Natural Language Processing")
    print("-" * 40)
    
    cli = NexusCLI()
    
    # Test natural language queries
    test_queries = [
        "How do I create a web server in Python?",
        "What's the difference between a list and a tuple?",
        "Help me understand decorators in Python"
    ]
    
    for query in test_queries:
        print(f"\n Query: {query}")
        response = cli._handle_natural_language(query)
        print(f" Response: {response[:200]}...")

def main():
    """Run all demos."""
    print(" Nexus CLI Demo")
    print("=" * 50)
    
    try:
        # Run demos
        demo_code_generation()
        demo_file_operations()
        demo_code_analysis()
        demo_project_management()
        demo_natural_language()
        
        print("\n All demos completed successfully!")
        print("\n To try the interactive CLI, run: python nexus_cli.py")
        
    except Exception as e:
        print(f"\n Demo failed: {e}")
        print("Make sure you have trained the model first: python train_nexus_model.py --create-data")

if __name__ == "__main__":
    main() 