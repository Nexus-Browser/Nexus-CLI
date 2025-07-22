#!/usr/bin/env python3
"""
Nexus CLI Integration Example
Shows how to integrate the Natural Language Developer Assistant into your main CLI
"""

import sys
import os

# Add model path
sys.path.append('/Users/anishpaleja/Nexus-CLI/model')

def demonstrate_nexus_integration():
    """Demonstrate how to integrate the developer assistant into Nexus CLI"""
    
    print("🚀 Nexus CLI - Natural Language Developer Assistant Integration")
    print("="*70)
    
    try:
        from illuminator_api import iLLuMinatorAPI
        
        # Initialize the enhanced iLLuMinator
        print("🧠 Loading Enhanced iLLuMinator...")
        illuminator = iLLuMinatorAPI(fast_mode=True, use_gpu_acceleration=False)
        
        print("✅ Ready! iLLuMinator now has access to the entire web!\n")
        
        # Example 1: Direct API usage
        print("📝 Example 1: Direct Developer Assistant Usage")
        print("-" * 50)
        
        question = "How does React useState work?"
        print(f"Question: {question}")
        
        response = illuminator.external_apis.answer_developer_question(question)
        print(f"Answer: {response[:200]}...\n")
        
        # Example 2: Integration with main iLLuMinator generate_response
        print("📝 Example 2: Enhanced iLLuMinator Response")
        print("-" * 50)
        
        prompt = "Explain Python asyncio"
        print(f"Prompt: {prompt}")
        
        # This now automatically uses web-enhanced responses
        enhanced_response = illuminator.generate_response(prompt)
        print(f"Enhanced Response: {enhanced_response[:200]}...\n")
        
        print("🎉 Integration Examples Complete!")
        print("\nKey Benefits:")
        print("• Real-time web search integration")
        print("• Multiple API sources (MDN, Stack Overflow, NPM, PyPI)")
        print("• Intelligent technology detection")
        print("• Quality response synthesis")
        print("• Caching for performance")
        
        return illuminator
        
    except Exception as e:
        print(f"❌ Integration failed: {e}")
        return None


def cli_command_example():
    """Example of how this could be used in a CLI command"""
    
    def ask_command(question: str) -> str:
        """CLI command: nexus ask 'How does fetch work?'"""
        try:
            from illuminator_api import iLLuMinatorAPI
            
            # Initialize once (could be cached globally)
            illuminator = iLLuMinatorAPI(fast_mode=True, use_gpu_acceleration=False)
            
            # Get comprehensive answer
            return illuminator.external_apis.answer_developer_question(question)
            
        except Exception as e:
            return f"Error: {e}"
    
    # Example usage
    print("\n" + "="*70)
    print("🔧 CLI Command Integration Example")
    print("="*70)
    
    examples = [
        "How does CSS flexbox work?",
        "What is Python decorators?",
        "How to deploy React app?"
    ]
    
    for example in examples:
        print(f"\n$ nexus ask '{example}'")
        print("-" * 40)
        
        try:
            answer = ask_command(example)
            print(answer[:300] + "..." if len(answer) > 300 else answer)
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Main demonstration function"""
    
    # Demonstrate basic integration
    illuminator = demonstrate_nexus_integration()
    
    if illuminator:
        # Show CLI command example
        cli_command_example()
        
        print("\n" + "="*70)
        print("🎯 Next Steps for Full Integration:")
        print("="*70)
        print("1. Add 'nexus ask' command to your main CLI")
        print("2. Integrate into existing code generation workflows") 
        print("3. Use for real-time documentation lookup")
        print("4. Enhance with API keys for premium sources")
        print("5. Add result caching for faster responses")
        
        print(f"\n💡 Your Nexus CLI now has access to the entire web!")
        print("   Ask any developer question and get comprehensive,")
        print("   real-time answers from multiple authoritative sources!")


if __name__ == "__main__":
    main()
