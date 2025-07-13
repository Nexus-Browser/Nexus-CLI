#!/usr/bin/env python3
"""
Test script for the enhanced Nexus CLI
Demonstrates intelligent features and capabilities
"""

import subprocess
import sys
import os
from pathlib import Path

def run_cli_command(command):
    """Run a CLI command and return the output."""
    try:
        result = subprocess.run(
            ['python', 'nexus_cli.py'],
            input=command + '\nexit\n',
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return "Command timed out", ""

def test_code_generation():
    """Test intelligent code generation."""
    print("🧪 Testing Code Generation...")
    
    test_cases = [
        "code function to multiply numbers",
        "code class calculator",
        "code web server flask",
        "code function to check if number is prime"
    ]
    
    for test_case in test_cases:
        print(f"\n📝 Testing: {test_case}")
        stdout, stderr = run_cli_command(test_case)
        
        if "Code generated successfully" in stdout:
            print("✅ Code generation working")
        else:
            print("❌ Code generation failed")
            print(f"Error: {stderr}")

def test_file_operations():
    """Test file operations."""
    print("\n🧪 Testing File Operations...")
    
    # Create a test file
    test_content = """def hello_world():
    print("Hello, World!")
    return "Hello, World!"

class TestClass:
    def __init__(self):
        self.name = "Test"
    
    def greet(self):
        return f"Hello, {self.name}!"
"""
    
    with open("test_file.py", "w") as f:
        f.write(test_content)
    
    # Test read operation
    print("📖 Testing file read...")
    stdout, stderr = run_cli_command("read test_file.py")
    
    if "File test_file.py read successfully" in stdout:
        print("✅ File read working")
    else:
        print("❌ File read failed")
    
    # Test analyze operation
    print("🔍 Testing code analysis...")
    stdout, stderr = run_cli_command("analyze test_file.py")
    
    if "Code analysis completed" in stdout:
        print("✅ Code analysis working")
    else:
        print("❌ Code analysis failed")
    
    # Test functions extraction
    print("📋 Testing functions extraction...")
    stdout, stderr = run_cli_command("functions test_file.py")
    
    if "Found" in stdout and "functions" in stdout:
        print("✅ Functions extraction working")
    else:
        print("❌ Functions extraction failed")
    
    # Clean up
    os.remove("test_file.py")

def test_project_operations():
    """Test project operations."""
    print("\n🧪 Testing Project Operations...")
    
    # Test list files
    print("📁 Testing file listing...")
    stdout, stderr = run_cli_command("list")
    
    if "Listed" in stdout and "files" in stdout:
        print("✅ File listing working")
    else:
        print("❌ File listing failed")
    
    # Test project tree
    print("🌳 Testing project tree...")
    stdout, stderr = run_cli_command("tree")
    
    if "Project tree displayed successfully" in stdout:
        print("✅ Project tree working")
    else:
        print("❌ Project tree failed")

def test_conversation():
    """Test conversation features."""
    print("\n🧪 Testing Conversation Features...")
    
    # Test natural language processing
    print("💬 Testing natural language...")
    stdout, stderr = run_cli_command("What is a variable in programming?")
    
    if "variable" in stdout.lower() or "Variable" in stdout:
        print("✅ Natural language processing working")
    else:
        print("❌ Natural language processing failed")

def test_help_and_system():
    """Test help and system commands."""
    print("\n🧪 Testing Help and System Commands...")
    
    # Test help
    print("❓ Testing help command...")
    stdout, stderr = run_cli_command("help")
    
    if "Available Commands" in stdout:
        print("✅ Help system working")
    else:
        print("❌ Help system failed")

def main():
    """Run all tests."""
    print("🚀 Starting Enhanced Nexus CLI Tests")
    print("=" * 50)
    
    try:
        test_code_generation()
        test_file_operations()
        test_project_operations()
        test_conversation()
        test_help_and_system()
        
        print("\n" + "=" * 50)
        print("🎉 All tests completed!")
        print("\n✨ Enhanced Nexus CLI Features:")
        print("   • Intelligent code generation with AST analysis")
        print("   • Beautiful rich terminal output")
        print("   • Context-aware command suggestions")
        print("   • Advanced file operations with syntax highlighting")
        print("   • Project detection and analysis")
        print("   • Memory and conversation management")
        print("   • Modern CLI patterns from successful tools")
        print("   • No GPT wrappers - pure open-source intelligence")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 