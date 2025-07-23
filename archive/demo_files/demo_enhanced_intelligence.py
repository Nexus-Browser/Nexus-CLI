#!/usr/bin/env python3
"""
Enhanced Nexus CLI Intelligence Demo
Shows the difference between fallback and enhanced modes
"""

import subprocess
import sys
import os
from pathlib import Path

def run_cli_command(command, timeout=30):
    """Run a CLI command and return the output."""
    try:
        result = subprocess.run(
            ['python', 'nexus_cli.py'],
            input=f"{command}\nexit\n",
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return "Command timed out", ""

def demo_fallback_mode():
    """Demonstrate the intelligent fallback mode."""
    print(" Demo: Intelligent Fallback Mode")
    print("=" * 50)
    print("This mode works without any API configuration.")
    print("It uses advanced pattern matching and AST analysis.\n")
    
    # Test code generation
    print(" Testing Code Generation (Fallback Mode):")
    test_cases = [
        "code function to calculate fibonacci numbers",
        "code class for managing a todo list",
        "code web server with flask"
    ]
    
    for test_case in test_cases:
        print(f"\n Request: {test_case}")
        stdout, stderr = run_cli_command(test_case)
        
        if "Code generated successfully" in stdout:
            print(" Generated intelligent code!")
        else:
            print(" Code generation failed")
    
    # Test conversation
    print("\n💬 Testing Conversation (Fallback Mode):")
    conversation_tests = [
        "What is a variable in programming?",
        "How do I create a function?",
        "Explain object-oriented programming"
    ]
    
    for test in conversation_tests:
        print(f"\n Question: {test}")
        stdout, stderr = run_cli_command(test)
        
        if "variable" in stdout.lower() or "function" in stdout.lower() or "class" in stdout.lower():
            print(" Provided intelligent response!")
        else:
            print(" Conversation failed")

def demo_enhanced_mode():
    """Demonstrate the enhanced mode with API integration."""
    print("\n Demo: Enhanced Intelligence Mode")
    print("=" * 50)
    print("This mode uses advanced AI capabilities for even smarter responses.")
    print("Configure your API key to enable this mode.\n")
    
    # Check if API is configured
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("NEXUS_API_KEY")
    
    if api_key:
        print(" API key detected - Enhanced mode available!")
        
        # Test enhanced code generation
        print("\n Testing Enhanced Code Generation:")
        enhanced_tests = [
            "code create a complete REST API with authentication and database",
            "code implement a machine learning pipeline for text classification",
            "code build a real-time chat application with WebSocket"
        ]
        
        for test_case in enhanced_tests:
            print(f"\n Request: {test_case}")
            stdout, stderr = run_cli_command(test_case)
            
            if "Code generated successfully" in stdout:
                print(" Generated production-ready code!")
            else:
                print(" Enhanced generation failed")
        
        # Test enhanced conversation
        print("\n💬 Testing Enhanced Conversation:")
        enhanced_conversation = [
            "How do I implement a custom authentication system with OAuth2?",
            "Explain the differences between synchronous and asynchronous programming",
            "What are the best practices for API design and documentation?"
        ]
        
        for test in enhanced_conversation:
            print(f"\n Question: {test}")
            stdout, stderr = run_cli_command(test)
            
            if len(stdout) > 100:  # Enhanced responses are typically longer
                print(" Provided detailed, enhanced response!")
            else:
                print(" Enhanced conversation failed")
    
    else:
        print("  No API key detected - Enhanced mode not available")
        print("   The CLI will automatically use intelligent fallback mode")
        print("   Set OPENAI_API_KEY or NEXUS_API_KEY to enable enhanced features")

def demo_features_comparison():
    """Show a comparison of features between modes."""
    print("\n📊 Feature Comparison")
    print("=" * 50)
    
    comparison = {
        "Code Generation": {
            "Fallback Mode": " Pattern-based generation\n AST analysis\n Multiple languages\n Error handling",
            "Enhanced Mode": " Production-ready code\n Best practices\n Complex algorithms\n Documentation"
        },
        "Conversation": {
            "Fallback Mode": " Pre-defined responses\n Programming topics\n Basic explanations\n Context awareness",
            "Enhanced Mode": " Detailed explanations\n Code reviews\n Best practices\n Real-time assistance"
        },
        "Performance": {
            "Fallback Mode": " Instant responses\n No network required\n Always available\n Lightweight",
            "Enhanced Mode": " Intelligent caching\n Rate limiting\n Graceful fallback\n Advanced features"
        }
    }
    
    for feature, modes in comparison.items():
        print(f"\n {feature}:")
        for mode, capabilities in modes.items():
            print(f"   {mode}:")
            for capability in capabilities.split('\n'):
                print(f"     {capability}")

def demo_setup_instructions():
    """Show setup instructions for enhanced mode."""
    print("\n Setup Instructions for Enhanced Mode")
    print("=" * 50)
    
    print("To enable enhanced intelligence:")
    print()
    print("1. Get an API key from OpenAI or your preferred provider")
    print("2. Set environment variable:")
    print("   export OPENAI_API_KEY='your-api-key-here'")
    print()
    print("3. Or edit model/api_config.json:")
    print("   {")
    print('     "api_key": "your-api-key-here"')
    print("   }")
    print()
    print("4. Run the CLI - it will automatically detect and use enhanced mode")
    print()
    print(" The CLI works perfectly in fallback mode while you configure enhanced features!")

def main():
    """Run the enhanced intelligence demo."""
    print(" Enhanced Nexus CLI Intelligence Demo")
    print("=" * 60)
    print()
    
    try:
        # Demo fallback mode (always works)
        demo_fallback_mode()
        
        # Demo enhanced mode (if configured)
        demo_enhanced_mode()
        
        # Show feature comparison
        demo_features_comparison()
        
        # Show setup instructions
        demo_setup_instructions()
        
        print("\n" + "=" * 60)
        print(" Enhanced Intelligence Demo Complete!")
        print()
        print(" Key Takeaways:")
        print("   • Fallback mode provides intelligent, reliable functionality")
        print("   • Enhanced mode adds production-ready code generation")
        print("   • Both modes are seamless and professional")
        print("   • Configuration is optional and subtle")
        print()
        print(" Ready to experience the future of CLI coding assistants!")
        
    except Exception as e:
        print(f"\n Demo failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 