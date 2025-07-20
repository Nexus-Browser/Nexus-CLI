#!/usr/bin/env python3
"""
Simple test script for Nexus CLI components
"""

import os
import sys
import json
import tempfile
from pathlib import Path

def test_imports():
    """Test that all modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        from model.nexus_model import NexusModel
        print(" NexusModel imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import NexusModel: {e}")
        return False
    
    try:
        from tools import FileTools, CodeTools, ProjectTools, MemoryTools
        print(" Tools imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import tools: {e}")
        return False
    
    try:
        from nexus_cli import NexusCLI
        print(" NexusCLI imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import NexusCLI: {e}")
        return False
    
    return True

def test_file_tools():
    """Test file operations."""
    print("\n Testing file tools...")
    
    from tools import FileTools
    file_tools = FileTools()
    
    # Test file writing and reading
    test_content = "def hello():\n    print('Hello, World!')"
    test_file = "test_file.py"
    
    # Write file
    success = file_tools.write_file(test_file, test_content)
    if success:
        print(" File writing successful")
    else:
        print("❌ File writing failed")
        return False
    
    # Read file
    content = file_tools.read_file(test_file)
    if content == test_content:
        print(" File reading successful")
    else:
        print("❌ File reading failed")
        return False
    
    # Clean up
    if os.path.exists(test_file):
        os.remove(test_file)
    
    return True

def test_code_tools():
    """Test code analysis tools."""
    print("\n Testing code tools...")
    
    from tools import CodeTools
    code_tools = CodeTools()
    
    # Test syntax validation
    valid_code = "def test(): pass"
    invalid_code = "def test( return"
    
    is_valid, message = code_tools.validate_python_syntax(valid_code)
    if is_valid:
        print(" Valid syntax detection working")
    else:
        print(f"❌ Valid syntax detection failed: {message}")
        return False
    
    is_valid, message = code_tools.validate_python_syntax(invalid_code)
    if not is_valid:
        print(" Invalid syntax detection working")
    else:
        print(f"❌ Invalid syntax detection failed: {message}")
        return False
    
    # Test function extraction
    test_code = """
def hello():
    print("Hello")

def add(a, b):
    return a + b
"""
    functions = code_tools.extract_functions(test_code)
    if len(functions) == 2:
        print(" Function extraction working")
    else:
        print(f"❌ Function extraction failed: found {len(functions)} functions")
        return False
    
    return True

def test_project_tools():
    """Test project management tools."""
    print("\n  Testing project tools...")
    
    from tools import ProjectTools
    project_tools = ProjectTools()
    
    # Test command execution
    returncode, stdout, stderr = project_tools.run_command("echo 'test'")
    if returncode == 0 and "test" in stdout:
        print(" Command execution working")
    else:
        print(f"❌ Command execution failed: {stderr}")
        return False
    
    # Test project structure
    structure = project_tools.get_project_structure(".", max_depth=1)
    if isinstance(structure, dict):
        print(" Project structure analysis working")
    else:
        print("❌ Project structure analysis failed")
        return False
    
    return True

def test_memory_tools():
    """Test memory management."""
    print("\n Testing memory tools...")
    
    from tools import MemoryTools
    memory_tools = MemoryTools("test_memory.json")
    
    # Test conversation memory
    memory_tools.add_conversation("Hello", "Hi there!", "test context")
    context = memory_tools.get_recent_context()
    if "Hello" in context and "Hi there!" in context:
        print(" Conversation memory working")
    else:
        print("❌ Conversation memory failed")
        return False
    
    # Test project context
    memory_tools.update_project_context("test_key", "test_value")
    value = memory_tools.get_project_context("test_key")
    if value == "test_value":
        print(" Project context working")
    else:
        print("❌ Project context failed")
        return False
    
    # Clean up
    if os.path.exists("test_memory.json"):
        os.remove("test_memory.json")
    
    return True

def test_config():
    """Test configuration loading."""
    print("\n  Testing configuration...")
    
    if not os.path.exists("model_config.json"):
        print("❌ model_config.json not found")
        return False
    
    try:
        with open("model_config.json", "r") as f:
            config = json.load(f)
        
        required_keys = ["model_name", "vocab_size", "n_positions"]
        for key in required_keys:
            if key not in config:
                print(f"❌ Missing required config key: {key}")
                return False
        
        print(" Configuration loading successful")
        return True
        
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False

def main():
    """Run all tests."""
    print(" Nexus CLI Component Tests")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_file_tools,
        test_code_tools,
        test_project_tools,
        test_memory_tools,
        test_config
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"❌ {test.__name__} failed")
        except Exception as e:
            print(f"❌ {test.__name__} failed with exception: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print(" All tests passed! Nexus CLI is ready to use.")
        print("\n Next steps:")
        print("1. Create training data: python train_nexus_model.py --create-data")
        print("2. Train the model: python train_nexus_model.py")
        print("3. Run the CLI: python nexus_cli.py")
    else:
        print("  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 