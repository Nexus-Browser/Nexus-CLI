#!/usr/bin/env python3
"""
Test the enhanced Nexus CLI with web intelligence
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from nexus_cli import IntelligentNexusCLI

def test_enhanced_nexus():
    """Test the enhanced Nexus CLI capabilities"""
    
    print("🚀 Testing Enhanced Nexus CLI with Web Intelligence")
    print("=" * 60)
    
    try:
        # Initialize CLI
        cli = IntelligentNexusCLI()
        
        # Test queries that should use web intelligence
        test_queries = [
            "ask What is React hooks?",
            "search Python asyncio tutorial",
            "How to use Docker containers?",
            "What are the best practices for JavaScript?",
            "explain machine learning"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🧪 Test {i}: {query}")
            print("-" * 40)
            
            try:
                result = cli.process_command(query)
                
                # Show truncated result
                if len(result) > 200:
                    print(result[:200] + "...[truncated]")
                else:
                    print(result)
                    
            except Exception as e:
                print(f"❌ Test failed: {e}")
            
            print()
    
    except Exception as e:
        print(f"❌ CLI initialization failed: {e}")
        print("\n💡 Make sure all dependencies are installed and the iLLuMinator model is available")

if __name__ == "__main__":
    test_enhanced_nexus()
