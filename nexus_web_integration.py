#!/usr/bin/env python3
"""
Nexus CLI Integration - Web-Enhanced RAG System
Integrates the web RAG system with the existing Nexus CLI
"""

import sys
import os
import logging
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

RAG_AVAILABLE = False
try:
    from web_rag_cli import WebRAGCLI
    RAG_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  RAG system not available: {e}")
    print("Run 'python setup_web_rag.py' to install dependencies")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NexusWebRAGIntegration:
    """
    Integration layer between Nexus CLI and Web-RAG system
    """
    
    def __init__(self):
        self.rag_system = None
        
        if RAG_AVAILABLE:
            try:
                self.rag_system = WebRAGCLI()
                logger.info("✅ Web-RAG system initialized")
            except Exception as e:
                logger.warning(f"⚠️  RAG initialization failed: {e}")
                self.rag_system = None
    
    def enhanced_query(self, query: str) -> str:
        """
        Process query with web-enhanced RAG
        Falls back to basic search if RAG is unavailable
        """
        if self.rag_system:
            try:
                return self.rag_system.process_query(query)
            except Exception as e:
                logger.error(f"RAG processing failed: {e}")
                return self._fallback_search(query)
        else:
            return self._fallback_search(query)
    
    def _fallback_search(self, query: str) -> str:
        """Fallback to basic web search without LLM"""
        try:
            from web_rag_cli import WebRAGSearchEngine
            
            search_engine = WebRAGSearchEngine()
            results = search_engine.search_comprehensive(query)
            
            if not results:
                return "I couldn't find relevant information for your query."
            
            # Format results without LLM processing
            response = f"🔍 **Search Results for: {query}**\n\n"
            
            for i, result in enumerate(results[:3], 1):
                response += f"**{i}. {result.title}**\n"
                response += f"{result.content[:200]}...\n"
                if result.url:
                    response += f"🔗 {result.url}\n"
                response += f"📍 Source: {result.source}\n\n"
            
            return response
            
        except Exception as e:
            return f"Search failed: {e}"
    
    def is_available(self) -> bool:
        """Check if RAG system is available"""
        return self.rag_system is not None

# CLI Enhancement Functions
def enhance_nexus_cli():
    """Add web-RAG commands to existing Nexus CLI"""
    
    def web_search_command(query: str):
        """Enhanced web search command"""
        integration = NexusWebRAGIntegration()
        
        print("🌐 Nexus CLI - Enhanced Web Search")
        print("=" * 50)
        
        if integration.is_available():
            print("🧠 Using AI-powered RAG system")
        else:
            print("🔍 Using basic web search")
        
        print(f"🎯 Query: {query}\n")
        
        response = integration.enhanced_query(query)
        print(response)
    
    def interactive_web_mode():
        """Interactive web-enhanced mode"""
        integration = NexusWebRAGIntegration()
        
        print("🌐 Nexus CLI - Interactive Web Mode")
        print("=" * 50)
        
        if integration.is_available():
            print("🧠 AI-powered responses available")
        else:
            print("🔍 Basic web search mode")
        
        print("Ask any question! Type 'exit' to quit.\n")
        
        while True:
            try:
                query = input("💭 Your question: ").strip()
                
                if query.lower() in ['exit', 'quit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not query:
                    continue
                
                print("\n🔄 Processing...")
                response = integration.enhanced_query(query)
                print("\n" + "="*60)
                print(response)
                print("="*60 + "\n")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    # Return enhanced functions
    return {
        'web_search': web_search_command,
        'interactive_web': interactive_web_mode
    }

def demo_integration():
    """Demonstrate the integrated system"""
    print("🚀 Nexus CLI Web-RAG Integration Demo")
    print("=" * 60)
    
    # Initialize integration
    integration = NexusWebRAGIntegration()
    
    if integration.is_available():
        print("✅ Full RAG system available")
        model_info = "with AI-powered response generation"
    else:
        print("⚠️  Basic search mode (install dependencies for full features)")
        model_info = "with web search only"
    
    print(f"🔧 System status: {model_info}\n")
    
    # Demo queries
    demo_queries = [
        "How does React useEffect work?",
        "Python asyncio best practices",
        "What is Docker compose?",
        "JavaScript async await explained"
    ]
    
    print("🧪 Running demo queries...")
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n{'='*60}")
        print(f"📝 Demo {i}: {query}")
        print("-" * 40)
        
        try:
            response = integration.enhanced_query(query)
            
            # Show truncated response for demo
            if len(response) > 300:
                print(response[:300] + "...\n[Response truncated for demo]")
            else:
                print(response)
                
        except Exception as e:
            print(f"❌ Demo query failed: {e}")
    
    print(f"\n{'='*60}")
    print("🎉 Demo complete!")
    print("\nTo use the system:")
    print("• python nexus_web_integration.py --interactive")
    print("• python nexus_web_integration.py --search 'your question'")

def main():
    """Main CLI for integration"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Nexus CLI Web-RAG Integration")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--search", "-s", help="Search query")
    parser.add_argument("--demo", "-d", action="store_true", help="Run demo")
    parser.add_argument("--status", action="store_true", help="Check system status")
    
    args = parser.parse_args()
    
    if args.status:
        integration = NexusWebRAGIntegration()
        print("🔧 Nexus CLI Web-RAG System Status")
        print("=" * 40)
        print(f"RAG Available: {'✅ Yes' if integration.is_available() else '❌ No'}")
        if RAG_AVAILABLE:
            print("Dependencies: ✅ Installed")
        else:
            print("Dependencies: ❌ Missing (run setup_web_rag.py)")
        
    elif args.demo:
        demo_integration()
        
    elif args.interactive:
        commands = enhance_nexus_cli()
        commands['interactive_web']()
        
    elif args.search:
        commands = enhance_nexus_cli()
        commands['web_search'](args.search)
        
    else:
        print("Nexus CLI Web-RAG Integration")
        print("Usage:")
        print("  --interactive    Interactive web mode")
        print("  --search 'query' Single search query")
        print("  --demo          Run demonstration")
        print("  --status        Check system status")

if __name__ == "__main__":
    main()
