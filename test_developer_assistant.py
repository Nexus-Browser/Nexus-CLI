#!/usr/bin/env python3
"""
Natural Language Developer Assistant Test
Demonstrates the enhanced iLLuMinator with real-time web-powered question answering
"""

import sys
import os
import time
import argparse

# Add model path
sys.path.append('/Users/anishpaleja/Nexus-CLI/model')

def test_natural_language_assistant():
    """Test the natural language developer assistant functionality"""
    try:
        from illuminator_api import iLLuMinatorAPI
        
        print("🚀 Initializing Enhanced iLLuMinator with Natural Language Developer Assistant...")
        
        # Initialize with fast mode for testing
        illuminator = iLLuMinatorAPI(fast_mode=True, use_gpu_acceleration=False)
        
        # Test questions to demonstrate capabilities
        test_questions = [
            "How does fetch() work in JavaScript?",
            "What is async/await in Python?",
            "Explain CSS flexbox",
            "How to create a React component?",
            "What is the difference between npm and yarn?",
            "How to handle errors in FastAPI?",
            "What are Rust ownership rules?",
            "How to optimize Python performance?"
        ]
        
        print("\n" + "="*80)
        print("🧠 Enhanced iLLuMinator - Natural Language Developer Assistant")
        print("="*80)
        print("This demo shows how iLLuMinator now answers developer questions using:")
        print("• Real-time web search")
        print("• Official documentation (MDN, Python docs, etc.)")
        print("• Package registries (NPM, PyPI, Crates.io)")
        print("• Stack Overflow solutions")
        print("• GitHub code examples")
        print("• Wikipedia technical concepts")
        print("="*80)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n🔍 Test {i}: {question}")
            print("-" * 60)
            
            start_time = time.time()
            
            try:
                # Use the external APIs directly for demonstration
                response = illuminator.external_apis.answer_developer_question(question)
                
                print(response)
                
            except Exception as e:
                print(f"❌ Error: {e}")
                # Fallback to basic response
                try:
                    basic_response = illuminator.generate_response(question)
                    print(f"🔄 Fallback response: {basic_response}")
                except Exception as e2:
                    print(f"❌ Fallback also failed: {e2}")
            
            search_time = time.time() - start_time
            print(f"\n⏱️  Answered in {search_time:.2f} seconds")
            print("="*80)
            
            # Pause between questions for rate limiting
            time.sleep(1)
        
        print("\n🎉 Natural Language Developer Assistant Test Complete!")
        print("\nThe enhanced iLLuMinator can now answer complex developer questions")
        print("using real-time web sources instead of just local knowledge!")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure you're in the Nexus-CLI directory and the model files exist.")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")


def interactive_mode():
    """Interactive mode for testing the assistant"""
    try:
        from illuminator_api import iLLuMinatorAPI
        
        print("🚀 Loading Enhanced iLLuMinator...")
        illuminator = iLLuMinatorAPI(fast_mode=True, use_gpu_acceleration=False)
        
        print("\n" + "="*60)
        print("🧠 Enhanced iLLuMinator - Interactive Developer Assistant")
        print("="*60)
        print("Ask any developer question! Type 'exit' to quit.")
        print("Examples:")
        print("  • How does React useState work?")
        print("  • What is Python async/await?")
        print("  • Explain CSS grid vs flexbox")
        print("  • How to handle errors in Node.js?")
        print("="*60)
        
        while True:
            try:
                question = input("\n💬 Your question: ").strip()
                
                if question.lower() in ['exit', 'quit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not question:
                    continue
                
                print("\n🔍 Researching your question...")
                start_time = time.time()
                
                try:
                    # Use the enhanced developer assistant
                    response = illuminator.external_apis.answer_developer_question(question)
                    print(f"\n{'='*60}")
                    print(response)
                    
                except Exception as e:
                    print(f"❌ Error with developer assistant: {e}")
                    # Fallback to basic response
                    try:
                        basic_response = illuminator.generate_response(question)
                        print(f"\n🔄 Basic response: {basic_response}")
                    except Exception as e2:
                        print(f"❌ All systems failed: {e2}")
                
                search_time = time.time() - start_time
                print(f"\n⏱️  Response generated in {search_time:.2f} seconds")
                print("="*60)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure you're in the Nexus-CLI directory and dependencies are installed.")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")


def quick_test():
    """Quick test with a single question"""
    try:
        from illuminator_api import iLLuMinatorAPI
        
        question = "How does fetch() work in JavaScript?"
        
        print(f"🚀 Quick Test: {question}")
        print("="*60)
        
        # Initialize assistant
        illuminator = iLLuMinatorAPI(fast_mode=True, use_gpu_acceleration=False)
        
        # Test the developer assistant
        start_time = time.time()
        response = illuminator.external_apis.answer_developer_question(question)
        search_time = time.time() - start_time
        
        print(response)
        print(f"\n⏱️  Completed in {search_time:.2f} seconds")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False


def main():
    """Main function with command line argument support"""
    parser = argparse.ArgumentParser(description="Natural Language Developer Assistant Test")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--quick", "-q", action="store_true", help="Quick test with one question")
    parser.add_argument("--full", "-f", action="store_true", help="Full test suite")
    parser.add_argument("question", nargs="*", help="Ask a specific question")
    
    args = parser.parse_args()
    
    if args.quick:
        success = quick_test()
        if success:
            print("\n✅ Quick test passed! The system is working.")
        else:
            print("\n❌ Quick test failed. Check your setup.")
    
    elif args.interactive:
        interactive_mode()
    
    elif args.full:
        test_natural_language_assistant()
    
    elif args.question:
        # Ask a specific question
        question = " ".join(args.question)
        try:
            from illuminator_api import iLLuMinatorAPI
            
            print(f"🔍 Question: {question}")
            print("="*60)
            
            illuminator = iLLuMinatorAPI(fast_mode=True, use_gpu_acceleration=False)
            
            start_time = time.time()
            response = illuminator.external_apis.answer_developer_question(question)
            search_time = time.time() - start_time
            
            print(response)
            print(f"\n⏱️  Answered in {search_time:.2f} seconds")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    else:
        print("🧠 Natural Language Developer Assistant")
        print("Usage examples:")
        print("  python test_developer_assistant.py --quick")
        print("  python test_developer_assistant.py --interactive")
        print("  python test_developer_assistant.py --full")
        print("  python test_developer_assistant.py 'How does React work?'")


if __name__ == "__main__":
    main()
