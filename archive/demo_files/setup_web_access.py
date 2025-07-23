#!/usr/bin/env python3
"""
Nexus CLI - Complete Web Access Setup Guide
Set up your CLI to access the entire web for data and code generation
"""

import os
import subprocess
import sys
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    
    print("🔍 Checking Web Access Dependencies...")
    
    required_packages = [
        ('requests', 'Web requests'),
        ('beautifulsoup4', 'HTML parsing'),
        ('lxml', 'XML/HTML parser'),
    ]
    
    optional_packages = [
        ('google-api-python-client', 'Google Search API'),
        ('selenium', 'Browser automation'),
    ]
    
    missing_required = []
    missing_optional = []
    
    for package, description in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} - {description}")
        except ImportError:
            print(f"❌ {package} - {description}")
            missing_required.append(package)
    
    for package, description in optional_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} - {description}")
        except ImportError:
            print(f"⚠️  {package} - {description} (optional)")
            missing_optional.append(package)
    
    return missing_required, missing_optional

def install_dependencies():
    """Install required dependencies for web access."""
    
    print("\n🚀 Installing Web Access Dependencies...")
    
    # Install basic requirements
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "requests", "beautifulsoup4", "lxml"
        ])
        print("✅ Basic web access dependencies installed!")
    except subprocess.CalledProcessError:
        print("❌ Failed to install basic dependencies")
        return False
    
    # Offer to install optional dependencies
    print("\n🤔 Install optional enhanced features?")
    print("   • Google Search API support")
    print("   • Browser automation for complex sites")
    
    choice = input("Install optional features? (y/n): ").lower().strip()
    
    if choice in ['y', 'yes']:
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "google-api-python-client", "selenium", "webdriver-manager"
            ])
            print("✅ Optional dependencies installed!")
        except subprocess.CalledProcessError:
            print("⚠️  Some optional dependencies failed to install")
    
    return True

def setup_api_keys():
    """Guide user through API key setup."""
    
    print("\n🔑 API Key Setup (Optional but Recommended)")
    print("=" * 50)
    
    env_file = Path(".env")
    env_content = []
    
    if env_file.exists():
        with open(env_file, 'r') as f:
            env_content = f.readlines()
    
    # Google Search API
    print("\n1. 🔍 Google Custom Search API (FREE: 100 searches/day)")
    print("   • Go to: https://developers.google.com/custom-search/v1/introduction")
    print("   • Create an API key and Custom Search Engine")
    
    google_key = input("   Enter Google Search API Key (or press Enter to skip): ").strip()
    google_engine = input("   Enter Custom Search Engine ID (or press Enter to skip): ").strip()
    
    if google_key:
        env_content.append(f"GOOGLE_SEARCH_API_KEY={google_key}\n")
        if google_engine:
            env_content.append(f"GOOGLE_SEARCH_ENGINE_ID={google_engine}\n")
    
    # Serper API (alternative)
    print("\n2. 🔎 Serper API - Google Search Alternative (FREE: 2,500/month)")
    print("   • Go to: https://serper.dev")
    print("   • Sign up and get your API key")
    
    serper_key = input("   Enter Serper API Key (or press Enter to skip): ").strip()
    if serper_key:
        env_content.append(f"SERPER_API_KEY={serper_key}\n")
    
    # GitHub Token
    print("\n3. 🐙 GitHub Personal Access Token (Better rate limits)")
    print("   • Go to: https://github.com/settings/tokens")
    print("   • Create a token with 'public_repo' scope")
    
    github_token = input("   Enter GitHub Token (or press Enter to skip): ").strip()
    if github_token:
        env_content.append(f"GITHUB_TOKEN={github_token}\n")
    
    # Save to .env file
    if env_content:
        with open(env_file, 'w') as f:
            f.writelines(env_content)
        print(f"\n✅ API keys saved to {env_file}")
    else:
        print("\n⚠️  No API keys provided - using free alternatives")

def test_web_access():
    """Test the web access functionality."""
    
    print("\n🧪 Testing Web Access...")
    
    try:
        import requests
        
        # Test basic web request
        response = requests.get("https://httpbin.org/json", timeout=5)
        if response.status_code == 200:
            print("✅ Basic web requests working")
        else:
            print("❌ Web requests failed")
            return False
        
        # Test API endpoints
        test_endpoints = [
            ("Stack Overflow", "https://api.stackexchange.com/2.3/info?site=stackoverflow"),
            ("GitHub API", "https://api.github.com/zen"),
            ("NPM Registry", "https://registry.npmjs.org/express"),
            ("PyPI", "https://pypi.org/pypi/requests/json"),
        ]
        
        for name, url in test_endpoints:
            try:
                response = requests.get(url, timeout=3)
                if response.status_code == 200:
                    print(f"✅ {name} API accessible")
                else:
                    print(f"⚠️  {name} API returned {response.status_code}")
            except requests.RequestException:
                print(f"❌ {name} API not accessible")
        
        print("\n🎉 Web access is configured and working!")
        return True
        
    except ImportError:
        print("❌ Required packages not installed")
        return False

def show_usage_examples():
    """Show examples of how to use web access in Nexus CLI."""
    
    print("\n📖 How to Use Web Access in Nexus CLI")
    print("=" * 45)
    
    examples = [
        ("Latest Information", "python nexus_cli.py 'what are the latest Python 3.12 features'"),
        ("Best Practices", "python nexus_cli.py 'FastAPI best practices 2024'"),
        ("Tutorials", "python nexus_cli.py 'how to learn React hooks with examples'"),
        ("Code Examples", "python nexus_cli.py 'show me Rust async programming examples'"),
        ("Documentation", "python nexus_cli.py 'explain Django ORM with examples'"),
        ("Web-Enhanced Code", "python nexus_cli.py 'code create a modern REST API with authentication'"),
    ]
    
    for category, command in examples:
        print(f"\n📌 {category}:")
        print(f"   {command}")
    
    print("\n💡 The CLI will automatically:")
    print("   • Search the web for latest information")
    print("   • Find relevant documentation and examples") 
    print("   • Include Stack Overflow solutions")
    print("   • Pull from GitHub repositories")
    print("   • Provide comprehensive, up-to-date answers")

def main():
    """Main setup function."""
    
    print("🌐 Nexus CLI - Complete Web Access Setup")
    print("=" * 50)
    print("Transform your CLI into a web-powered AI assistant!")
    print("Access the entire internet for code generation and information.")
    
    # Check current state
    missing_required, missing_optional = check_dependencies()
    
    if missing_required:
        print(f"\n⚠️  Missing required dependencies: {', '.join(missing_required)}")
        install_choice = input("Install required dependencies? (y/n): ").lower().strip()
        
        if install_choice in ['y', 'yes']:
            if not install_dependencies():
                print("❌ Setup failed. Please install dependencies manually.")
                return
        else:
            print("❌ Cannot proceed without required dependencies.")
            return
    
    # Set up API keys
    api_choice = input("\n🔑 Set up API keys for enhanced features? (y/n): ").lower().strip()
    if api_choice in ['y', 'yes']:
        setup_api_keys()
    
    # Test everything
    print("\n🧪 Running final tests...")
    if test_web_access():
        print("\n🎉 SUCCESS! Your Nexus CLI now has complete web access!")
        show_usage_examples()
    else:
        print("\n⚠️  Some tests failed, but basic functionality should work.")
    
    print("\n" + "=" * 50)
    print("🚀 Your CLI can now search the ENTIRE WEB!")
    print("   • Real-time information from any website")
    print("   • Latest documentation and tutorials")  
    print("   • Code examples from GitHub")
    print("   • Community discussions and solutions")
    print("   • Up-to-date best practices and news")

if __name__ == "__main__":
    main()
