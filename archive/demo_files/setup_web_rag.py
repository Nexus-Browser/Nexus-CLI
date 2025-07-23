#!/usr/bin/env python3
"""
Setup script for Nexus CLI Web-RAG System
Installs dependencies and configures the system
"""

import subprocess
import sys
import os
import json
from pathlib import Path

def install_dependencies():
    """Install required Python packages"""
    print("📦 Installing Python dependencies...")
    
    try:
        # Install PyTorch (CPU version for compatibility)
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"
        ])
        print("✅ PyTorch installed")
        
        # Install transformers and other dependencies
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "transformers", "requests", "accelerate", "sentencepiece", "protobuf"
        ])
        print("✅ Transformers and dependencies installed")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        return False
    
    return True

def download_sample_models():
    """Download and cache small models for testing"""
    print("🤖 Downloading sample models...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Download DialoGPT-small (lightweight for testing)
        model_name = "microsoft/DialoGPT-small"
        print(f"Downloading {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        print("✅ Sample model downloaded and cached")
        return True
        
    except Exception as e:
        print(f"⚠️  Model download failed: {e}")
        print("Models will be downloaded on first use")
        return False

def create_config():
    """Create configuration file"""
    config = {
        "default_model": "microsoft/DialoGPT-medium",
        "cache_ttl": 3600,
        "max_results": 8,
        "api_keys": {
            "github_token": "your_github_token_here",
            "note": "Add your API keys here for enhanced search capabilities"
        }
    }
    
    config_path = Path("web_rag_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"⚙️  Configuration created: {config_path}")

def test_installation():
    """Test the installation"""
    print("🧪 Testing installation...")
    
    try:
        # Test imports
        import torch
        import transformers
        import requests
        
        print("✅ All imports successful")
        
        # Test basic functionality
        from web_rag_cli import WebRAGSearchEngine
        
        search_engine = WebRAGSearchEngine()
        print("✅ Search engine initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main setup process"""
    print("🚀 Nexus CLI Web-RAG System Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed during dependency installation")
        sys.exit(1)
    
    # Create config
    create_config()
    
    # Download sample models (optional)
    download_sample_models()
    
    # Test installation
    if test_installation():
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. python web_rag_cli.py --interactive")
        print("2. python web_rag_cli.py 'What is machine learning?'")
        print("\nFor better performance, add API keys to web_rag_config.json")
    else:
        print("⚠️  Setup completed with warnings")
        print("The system should still work, but some features may be limited")

if __name__ == "__main__":
    main()
