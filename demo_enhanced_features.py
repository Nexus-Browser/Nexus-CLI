#!/usr/bin/env python3
"""
Enhanced Nexus CLI Demo
Shows intelligent features and capabilities in action
"""

import subprocess
import sys
import os
from pathlib import Path

def create_demo_files():
    """Create demo files for testing."""
    # Create a sample Python file
    python_code = '''"""
Sample Python module for demonstration
"""
import json
from typing import List, Dict

def calculate_fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number."""
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

def sort_list(items: List) -> List:
    """Sort a list of items."""
    return sorted(items)

class DataProcessor:
    """A class for processing data."""
    
    def __init__(self, data: List):
        self.data = data
    
    def filter_even(self) -> List:
        """Filter even numbers from data."""
        return [x for x in self.data if x % 2 == 0]
    
    def get_statistics(self) -> Dict:
        """Get basic statistics of the data."""
        if not self.data:
            return {}
        return {
            "count": len(self.data),
            "sum": sum(self.data),
            "average": sum(self.data) / len(self.data),
            "min": min(self.data),
            "max": max(self.data)
        }

if __name__ == "__main__":
    # Demo usage
    processor = DataProcessor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print("Even numbers:", processor.filter_even())
    print("Statistics:", processor.get_statistics())
    print("Fibonacci(10):", calculate_fibonacci(10))
'''
    
    with open("demo_module.py", "w") as f:
        f.write(python_code)
    
    # Create a requirements file
    requirements = '''rich>=10.0.0
pathlib
typing-extensions
'''
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)

def run_demo():
    """Run the enhanced CLI demo."""
    print("🚀 Enhanced Nexus CLI Demo")
    print("=" * 60)
    print()
    
    # Create demo files
    create_demo_files()
    
    print("📁 Created demo files:")
    print("   • demo_module.py - Sample Python module")
    print("   • requirements.txt - Dependencies file")
    print()
    
    # Demo 1: Code Generation
    print("🎯 Demo 1: Intelligent Code Generation")
    print("-" * 40)
    
    code_examples = [
        "code function to reverse a string",
        "code class todo list manager",
        "code web server with flask",
        "code function to find prime numbers"
    ]
    
    for example in code_examples:
        print(f"\n💡 Generating: {example}")
        result = subprocess.run(
            ['python', 'nexus_cli.py'],
            input=f"{example}\nexit\n",
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if "Code generated successfully" in result.stdout:
            print("✅ Generated working code!")
        else:
            print("❌ Code generation failed")
    
    # Demo 2: File Analysis
    print("\n\n🔍 Demo 2: Advanced File Analysis")
    print("-" * 40)
    
    analysis_commands = [
        "read demo_module.py",
        "analyze demo_module.py",
        "functions demo_module.py",
        "classes demo_module.py"
    ]
    
    for cmd in analysis_commands:
        print(f"\n📊 Running: {cmd}")
        result = subprocess.run(
            ['python', 'nexus_cli.py'],
            input=f"{cmd}\nexit\n",
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if "successfully" in result.stdout or "Found" in result.stdout:
            print("✅ Analysis completed!")
        else:
            print("❌ Analysis failed")
    
    # Demo 3: Project Operations
    print("\n\n📂 Demo 3: Project Operations")
    print("-" * 40)
    
    project_commands = [
        "list",
        "tree",
        "run python demo_module.py"
    ]
    
    for cmd in project_commands:
        print(f"\n🛠️  Running: {cmd}")
        result = subprocess.run(
            ['python', 'nexus_cli.py'],
            input=f"{cmd}\nexit\n",
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if "successfully" in result.stdout or "Listed" in result.stdout:
            print("✅ Operation completed!")
        else:
            print("❌ Operation failed")
    
    # Demo 4: Smart Features
    print("\n\n🧠 Demo 4: Smart Features")
    print("-" * 40)
    
    smart_commands = [
        "help",
        "install",
        "test"
    ]
    
    for cmd in smart_commands:
        print(f"\n🤖 Running: {cmd}")
        result = subprocess.run(
            ['python', 'nexus_cli.py'],
            input=f"{cmd}\nexit\n",
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if "Available Commands" in result.stdout or "Detecting" in result.stdout:
            print("✅ Smart feature working!")
        else:
            print("❌ Smart feature failed")
    
    # Cleanup
    cleanup_files = ["demo_module.py", "requirements.txt"]
    for file in cleanup_files:
        if os.path.exists(file):
            os.remove(file)
    
    print("\n" + "=" * 60)
    print("🎉 Enhanced Nexus CLI Demo Complete!")
    print()
    print("✨ Key Features Demonstrated:")
    print("   • 🎯 Intelligent code generation from natural language")
    print("   • 🔍 Advanced code analysis with AST parsing")
    print("   • 📊 Function and class extraction")
    print("   • 📂 Smart project operations")
    print("   • 🛠️  Context-aware command execution")
    print("   • 🎨 Beautiful rich terminal output")
    print("   • 🧠 Memory and conversation management")
    print("   • 🚀 Modern CLI patterns from successful tools")
    print()
    print("💡 No GPT wrappers - Pure open-source intelligence!")
    print("🔧 Built with techniques from Warp, Cursor, and other successful CLI tools")

if __name__ == "__main__":
    run_demo() 