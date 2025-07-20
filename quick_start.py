#!/usr/bin/env python3
"""
Quick Start Script for Nexus CLI
Sets up the environment and trains a basic model for immediate use
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f" {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f" Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def create_directories():
    """Create necessary directories."""
    directories = ["data", "memory", "model"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print(" Directories created")

def install_dependencies():
    """Install required dependencies."""
    return run_command("pip install -r requirements.txt", "Installing dependencies")

def create_sample_data():
    """Create sample training data."""
    sample_data = [
        {
            "prompt": "Create a function to calculate the factorial of a number",
            "completion": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"
        },
        {
            "prompt": "Write a function to check if a string is a palindrome",
            "completion": "def is_palindrome(s):\n    s = s.lower().replace(' ', '')\n    return s == s[::-1]"
        },
        {
            "prompt": "Create a class for a simple calculator",
            "completion": "class Calculator:\n    def add(self, a, b):\n        return a + b\n    \n    def subtract(self, a, b):\n        return a - b\n    \n    def multiply(self, a, b):\n        return a * b\n    \n    def divide(self, a, b):\n        if b == 0:\n            raise ValueError('Cannot divide by zero')\n        return a / b"
        },
        {
            "prompt": "Write a function to find the maximum element in a list",
            "completion": "def find_max(lst):\n    if not lst:\n        return None\n    return max(lst)"
        },
        {
            "prompt": "Create a function to reverse a string",
            "completion": "def reverse_string(s):\n    return s[::-1]"
        },
        {
            "prompt": "Write a function to check if a number is prime",
            "completion": "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True"
        },
        {
            "prompt": "Create a function to generate Fibonacci sequence",
            "completion": "def fibonacci(n):\n    if n <= 0:\n        return []\n    elif n == 1:\n        return [0]\n    elif n == 2:\n        return [0, 1]\n    \n    fib = [0, 1]\n    for i in range(2, n):\n        fib.append(fib[i-1] + fib[i-2])\n    return fib"
        },
        {
            "prompt": "Write a function to sort a list using bubble sort",
            "completion": "def bubble_sort(lst):\n    n = len(lst)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if lst[j] > lst[j+1]:\n                lst[j], lst[j+1] = lst[j+1], lst[j]\n    return lst"
        },
        {
            "prompt": "Create a function to count vowels in a string",
            "completion": "def count_vowels(s):\n    vowels = 'aeiouAEIOU'\n    return sum(1 for char in s if char in vowels)"
        },
        {
            "prompt": "Write a function to check if two strings are anagrams",
            "completion": "def are_anagrams(s1, s2):\n    return sorted(s1.lower()) == sorted(s2.lower())"
        }
    ]
    
    with open("data/custom_training_data.json", "w") as f:
        json.dump(sample_data, f, indent=2)
    
    print(f" Created {len(sample_data)} sample training examples")

def train_basic_model():
    """Train a basic model for immediate use."""
    print(" Training basic model (this may take a few minutes)...")
    return run_command(
        "python train_nexus_model.py --epochs 2 --batch-size 2 --max-length 256",
        "Training basic model"
    )

def test_cli():
    """Test the CLI with a simple command."""
    print("🧪 Testing CLI...")
    test_command = 'echo "code create a hello world function" | python nexus_cli.py --debug'
    return run_command(test_command, "Testing CLI")

def main():
    """Main setup function."""
    print(" Nexus CLI Quick Start Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies. Please check your internet connection and try again.")
        sys.exit(1)
    
    # Create sample data
    create_sample_data()
    
    # Train basic model
    if not train_basic_model():
        print("  Model training failed, but you can still use the CLI with the base model")
    
    # Test CLI
    test_cli()
    
    print("\n Setup completed!")
    print("\n📖 Next steps:")
    print("1. Run the CLI: python nexus_cli.py")
    print("2. Try some commands:")
    print("   - code create a function to calculate fibonacci numbers")
    print("   - read nexus_cli.py")
    print("   - help")
    print("3. Train a better model: python train_nexus_model.py --epochs 5")
    
    print("\n For more information, see README.md")

if __name__ == "__main__":
    main() 