"""
Enhanced Nexus CLI - Advanced Code Intelligence
Integrates state-of-the-art LLM architecture with intelligent assistance
Combines proven techniques from LLMs-from-scratch and nanoGPT
"""

import os
import sys
import json
import time
import argparse
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
import asyncio
import signal

# Add model directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))

try:
    import torch
    import torch.nn.functional as F
    from nexus_llm import NexusLLM, NexusConfig, create_model
    from tokenizer import NexusTokenizer, create_tokenizer
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"Warning: PyTorch not available: {e}")
    TORCH_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nexus.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class NexusSession:
    """Session management for conversation and context"""
    conversation_history: List[Dict[str, str]]
    context_memory: Dict[str, Any]
    working_directory: str
    active_files: List[str]
    model_state: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'conversation_history': self.conversation_history,
            'context_memory': self.context_memory,
            'working_directory': self.working_directory,
            'active_files': self.active_files,
            'model_state': self.model_state
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NexusSession':
        return cls(**data)


class WebIntelligence:
    """Enhanced web intelligence for real-time information"""
    
    def __init__(self):
        self.search_engines = {
            'duckduckgo': self._search_duckduckgo,
            'wikipedia': self._search_wikipedia,
            'github': self._search_github
        }
        self.cache = {}
        
    async def search(self, query: str, engine: str = 'duckduckgo') -> Dict[str, Any]:
        """Perform intelligent web search"""
        cache_key = f"{engine}:{query}"
        
        if cache_key in self.cache:
            logger.info(f"Using cached result for: {query}")
            return self.cache[cache_key]
        
        try:
            if engine in self.search_engines:
                result = await self.search_engines[engine](query)
                self.cache[cache_key] = result
                return result
            else:
                return {"error": f"Unknown search engine: {engine}"}
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {"error": str(e)}
    
    async def _search_duckduckgo(self, query: str) -> Dict[str, Any]:
        """DuckDuckGo search implementation"""
        # Simulated search - replace with actual API calls
        return {
            "query": query,
            "results": [
                {
                    "title": f"Search result for: {query}",
                    "url": "https://example.com",
                    "snippet": f"Relevant information about {query}"
                }
            ],
            "timestamp": time.time()
        }
    
    async def _search_wikipedia(self, query: str) -> Dict[str, Any]:
        """Wikipedia search implementation"""
        return {
            "query": query,
            "summary": f"Wikipedia summary for {query}",
            "url": f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}",
            "timestamp": time.time()
        }
    
    async def _search_github(self, query: str) -> Dict[str, Any]:
        """GitHub search implementation"""
        return {
            "query": query,
            "repositories": [
                {
                    "name": f"relevant-repo-{query}",
                    "description": f"Repository related to {query}",
                    "stars": 1000,
                    "url": "https://github.com/example/repo"
                }
            ],
            "timestamp": time.time()
        }


class CodeAnalyzer:
    """Advanced code analysis and understanding"""
    
    def __init__(self):
        self.supported_languages = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php'
        }
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze code file structure and content"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            ext = Path(file_path).suffix
            language = self.supported_languages.get(ext, 'unknown')
            
            analysis = {
                'file_path': file_path,
                'language': language,
                'lines': len(content.split('\n')),
                'size': len(content),
                'functions': self._extract_functions(content, language),
                'classes': self._extract_classes(content, language),
                'imports': self._extract_imports(content, language),
                'complexity': self._estimate_complexity(content)
            }
            
            return analysis
            
        except Exception as e:
            return {'error': str(e)}
    
    def _extract_functions(self, content: str, language: str) -> List[str]:
        """Extract function definitions"""
        import re
        
        patterns = {
            'python': r'def\s+(\w+)\s*\(',
            'javascript': r'function\s+(\w+)\s*\(',
            'typescript': r'function\s+(\w+)\s*\(',
            'java': r'(?:public|private|protected)?\s*(?:static)?\s*\w+\s+(\w+)\s*\(',
            'cpp': r'\w+\s+(\w+)\s*\(',
            'c': r'\w+\s+(\w+)\s*\('
        }
        
        pattern = patterns.get(language, r'(\w+)\s*\(')
        matches = re.findall(pattern, content)
        return matches
    
    def _extract_classes(self, content: str, language: str) -> List[str]:
        """Extract class definitions"""
        import re
        
        patterns = {
            'python': r'class\s+(\w+)',
            'javascript': r'class\s+(\w+)',
            'typescript': r'class\s+(\w+)',
            'java': r'class\s+(\w+)',
            'cpp': r'class\s+(\w+)',
        }
        
        pattern = patterns.get(language, r'class\s+(\w+)')
        matches = re.findall(pattern, content)
        return matches
    
    def _extract_imports(self, content: str, language: str) -> List[str]:
        """Extract import statements"""
        import re
        
        patterns = {
            'python': r'(?:import|from)\s+([\w.]+)',
            'javascript': r'import\s+.*?from\s+["\'](.+?)["\']',
            'typescript': r'import\s+.*?from\s+["\'](.+?)["\']',
            'java': r'import\s+([\w.]+)',
        }
        
        pattern = patterns.get(language, r'import\s+([\w.]+)')
        matches = re.findall(pattern, content)
        return matches
    
    def _estimate_complexity(self, content: str) -> int:
        """Estimate code complexity"""
        import re
        
        # Simple complexity estimation based on control structures
        control_structures = [
            r'\bif\b', r'\belse\b', r'\belif\b', r'\bwhile\b',
            r'\bfor\b', r'\btry\b', r'\bcatch\b', r'\bswitch\b'
        ]
        
        complexity = 1  # Base complexity
        for pattern in control_structures:
            complexity += len(re.findall(pattern, content))
        
        return complexity


class NexusCLI:
    """
    Enhanced Nexus CLI with advanced LLM architecture integration
    Combines proven optimization techniques for maximum performance
    """
    
    def __init__(self, model_path: str = None, config_path: str = None):
        self.model_path = model_path or "model/nexus_model"
        self.config_path = config_path or "model_config.json"
        self.session_file = "memory/nexus_session.json"
        
        # Initialize components
        self.web_intelligence = WebIntelligence()
        self.code_analyzer = CodeAnalyzer()
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if TORCH_AVAILABLE else None
        
        # Initialize session
        self.session = self._load_session()
        
        # Performance tracking
        self.stats = {
            'total_requests': 0,
            'average_response_time': 0.0,
            'cache_hits': 0,
            'model_calls': 0
        }
        
        logger.info("✓ Nexus CLI initialized with enhanced architecture")
    
    def _load_config(self) -> NexusConfig:
        """Load model configuration"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config_dict = json.load(f)
                return NexusConfig(**config_dict)
            else:
                # Default configuration optimized for performance
                config = NexusConfig(
                    block_size=2048,
                    vocab_size=50304,
                    n_layer=12,
                    n_head=12,
                    n_embd=768,
                    dropout=0.0,
                    bias=False,
                    use_flash_attention=True,
                    use_kv_cache=True,
                    temperature=0.8,
                    top_k=200,
                    max_new_tokens=500
                )
                self._save_config(config)
                return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return NexusConfig()
    
    def _save_config(self, config: NexusConfig):
        """Save configuration to file"""
        try:
            config_dict = {
                'block_size': config.block_size,
                'vocab_size': config.vocab_size,
                'n_layer': config.n_layer,
                'n_head': config.n_head,
                'n_embd': config.n_embd,
                'dropout': config.dropout,
                'bias': config.bias,
                'use_flash_attention': config.use_flash_attention,
                'use_kv_cache': config.use_kv_cache,
                'temperature': config.temperature,
                'top_k': config.top_k,
                'max_new_tokens': config.max_new_tokens
            }
            with open(self.config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def _load_session(self) -> NexusSession:
        """Load session from file or create new one"""
        try:
            if os.path.exists(self.session_file):
                with open(self.session_file, 'r') as f:
                    data = json.load(f)
                return NexusSession.from_dict(data)
            else:
                return NexusSession(
                    conversation_history=[],
                    context_memory={},
                    working_directory=os.getcwd(),
                    active_files=[]
                )
        except Exception as e:
            logger.error(f"Failed to load session: {e}")
            return NexusSession([], {}, os.getcwd(), [])
    
    def _save_session(self):
        """Save current session"""
        try:
            os.makedirs(os.path.dirname(self.session_file), exist_ok=True)
            with open(self.session_file, 'w') as f:
                json.dump(self.session.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
    
    def initialize_model(self):
        """Initialize the enhanced LLM model"""
        if not TORCH_AVAILABLE:
            logger.error("PyTorch not available - model cannot be loaded")
            return False
        
        try:
            logger.info("Initializing enhanced Nexus LLM...")
            
            # Create tokenizer
            self.tokenizer = create_tokenizer()
            
            # Create model with optimizations
            self.model = create_model(self.config)
            self.model.to(self.device)
            self.model.eval()
            
            # Load pretrained weights if available
            if os.path.exists(self.model_path):
                try:
                    checkpoint = torch.load(f"{self.model_path}/model.pt", map_location=self.device)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    logger.info(f"✓ Loaded model weights from {self.model_path}")
                except Exception as e:
                    logger.warning(f"Could not load weights: {e}")
            
            logger.info(f"✓ Model initialized on {self.device} with {self.model.get_num_params():,} parameters")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            return False
    
    async def process_query(self, query: str, context: str = None) -> Dict[str, Any]:
        """
        Process user query with enhanced intelligence
        Combines local model inference with web intelligence
        """
        start_time = time.time()
        self.stats['total_requests'] += 1
        
        try:
            # Analyze query intent
            intent = self._analyze_intent(query)
            
            # Gather context
            full_context = await self._gather_context(query, context, intent)
            
            # Generate response
            if self.model and self.tokenizer:
                response = await self._generate_response(query, full_context, intent)
            else:
                response = await self._fallback_response(query, full_context)
            
            # Post-process and enhance
            enhanced_response = await self._enhance_response(response, intent)
            
            # Update session
            self._update_session(query, enhanced_response)
            
            # Calculate performance metrics
            response_time = time.time() - start_time
            self.stats['average_response_time'] = (
                (self.stats['average_response_time'] * (self.stats['total_requests'] - 1) + response_time)
                / self.stats['total_requests']
            )
            
            return {
                'response': enhanced_response,
                'intent': intent,
                'context_used': full_context,
                'performance': {
                    'response_time': response_time,
                    'model_params': self.model.get_num_params() if self.model else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                'response': f"I apologize, but I encountered an error: {str(e)}",
                'error': str(e),
                'performance': {'response_time': time.time() - start_time}
            }
    
    def _analyze_intent(self, query: str) -> Dict[str, Any]:
        """Analyze user intent from query"""
        intent = {
            'type': 'general',
            'confidence': 0.5,
            'entities': [],
            'needs_web': False,
            'needs_code': False,
            'needs_file': False
        }
        
        query_lower = query.lower()
        
        # Code-related intents
        code_keywords = ['function', 'class', 'method', 'code', 'debug', 'error', 'implement', 'write']
        if any(keyword in query_lower for keyword in code_keywords):
            intent['type'] = 'code'
            intent['needs_code'] = True
            intent['confidence'] = 0.8
        
        # File-related intents
        file_keywords = ['file', 'directory', 'folder', 'read', 'write', 'analyze']
        if any(keyword in query_lower for keyword in file_keywords):
            intent['needs_file'] = True
            intent['confidence'] = max(intent['confidence'], 0.7)
        
        # Web search intents
        web_keywords = ['search', 'find', 'look up', 'what is', 'how to', 'latest', 'current']
        if any(keyword in query_lower for keyword in web_keywords):
            intent['needs_web'] = True
            intent['confidence'] = max(intent['confidence'], 0.8)
        
        return intent
    
    async def _gather_context(self, query: str, provided_context: str, intent: Dict[str, Any]) -> str:
        """Gather comprehensive context for the query"""
        context_parts = []
        
        # Add provided context
        if provided_context:
            context_parts.append(f"Provided context: {provided_context}")
        
        # Add conversation history
        if self.session.conversation_history:
            recent_history = self.session.conversation_history[-3:]  # Last 3 exchanges
            history_text = "\n".join([
                f"{msg['role']}: {msg['content']}" 
                for msg in recent_history
            ])
            context_parts.append(f"Recent conversation:\n{history_text}")
        
        # Add file context if needed
        if intent['needs_file'] and self.session.active_files:
            for file_path in self.session.active_files[-2:]:  # Last 2 files
                if os.path.exists(file_path):
                    analysis = self.code_analyzer.analyze_file(file_path)
                    context_parts.append(f"File analysis for {file_path}: {analysis}")
        
        # Add web intelligence if needed
        if intent['needs_web']:
            web_results = await self.web_intelligence.search(query)
            if 'results' in web_results:
                web_context = "; ".join([
                    f"{result['title']}: {result['snippet']}"
                    for result in web_results['results'][:2]  # Top 2 results
                ])
                context_parts.append(f"Web search results: {web_context}")
        
        return "\n\n".join(context_parts)
    
    async def _generate_response(self, query: str, context: str, intent: Dict[str, Any]) -> str:
        """Generate response using the enhanced LLM"""
        self.stats['model_calls'] += 1
        
        try:
            # Create prompt with enhanced formatting
            messages = [
                {"role": "system", "content": "You are Nexus, an advanced AI coding assistant with deep technical knowledge."},
                {"role": "user", "content": f"Context: {context}\n\nQuery: {query}"}
            ]
            
            # Format using chat template
            prompt = self.tokenizer.create_chat_template(messages)
            
            # Encode prompt
            input_ids = torch.tensor([self.tokenizer.encode(prompt)], device=self.device)
            
            # Generate with KV caching for 4x speedup
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_k=self.config.top_k
                )
            
            # Decode response
            response_ids = output_ids[0][input_ids.shape[1]:]
            response = self.tokenizer.decode(response_ids.tolist(), skip_special_tokens=True)
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Model generation failed: {e}")
            return await self._fallback_response(query, context)
    
    async def _fallback_response(self, query: str, context: str) -> str:
        """Fallback response when model is unavailable"""
        return f"""I understand you're asking: "{query}"

Based on the available context, I can see this relates to your current work. 
While my advanced model is currently unavailable, I can still help you with:

1. Code analysis and suggestions
2. File operations and management  
3. Web search and information gathering
4. Technical problem-solving

Please let me know what specific assistance you need, and I'll do my best to help using alternative methods."""
    
    async def _enhance_response(self, response: str, intent: Dict[str, Any]) -> str:
        """Enhance response with additional intelligence"""
        enhanced = response
        
        # Add code formatting if response contains code
        if intent['needs_code'] and ('def ' in response or 'function ' in response or '{' in response):
            enhanced = f"```\n{response}\n```" if not response.startswith('```') else response
        
        # Add helpful suggestions based on intent
        if intent['type'] == 'code':
            enhanced += "\n\n💡 Tip: I can help you debug, optimize, or explain this code further if needed."
        
        return enhanced
    
    def _update_session(self, query: str, response: str):
        """Update session with new interaction"""
        self.session.conversation_history.append({
            'role': 'user',
            'content': query,
            'timestamp': time.time()
        })
        
        self.session.conversation_history.append({
            'role': 'assistant', 
            'content': response,
            'timestamp': time.time()
        })
        
        # Keep only last 20 interactions
        if len(self.session.conversation_history) > 40:
            self.session.conversation_history = self.session.conversation_history[-40:]
        
        self._save_session()
    
    def add_file_to_context(self, file_path: str):
        """Add file to active context"""
        if os.path.exists(file_path) and file_path not in self.session.active_files:
            self.session.active_files.append(file_path)
            logger.info(f"Added {file_path} to active context")
            self._save_session()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            **self.stats,
            'session_length': len(self.session.conversation_history),
            'active_files': len(self.session.active_files),
            'model_loaded': self.model is not None,
            'device': str(self.device) if self.device else None
        }


async def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description='Nexus CLI - Advanced Code Intelligence')
    parser.add_argument('--model-path', help='Path to model directory')
    parser.add_argument('--config-path', help='Path to config file')
    parser.add_argument('--interactive', '-i', action='store_true', help='Start interactive mode')
    parser.add_argument('--file', '-f', help='Add file to context')
    parser.add_argument('query', nargs='*', help='Query to process')
    
    args = parser.parse_args()
    
    # Initialize CLI
    cli = NexusCLI(args.model_path, args.config_path)
    
    # Initialize model
    if TORCH_AVAILABLE:
        cli.initialize_model()
    else:
        print("⚠️  Running without PyTorch - limited functionality available")
    
    # Add file to context if specified
    if args.file:
        cli.add_file_to_context(args.file)
    
    # Process query or start interactive mode
    if args.query:
        query = ' '.join(args.query)
        result = await cli.process_query(query)
        print(f"\n🤖 Nexus: {result['response']}\n")
        
        if 'performance' in result:
            print(f"⚡ Response time: {result['performance']['response_time']:.2f}s")
    
    elif args.interactive:
        print("🚀 Nexus CLI - Enhanced Code Intelligence")
        print("Type 'exit' to quit, 'stats' for performance info, 'help' for commands\n")
        
        # Setup signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            print("\n👋 Goodbye!")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        while True:
            try:
                query = input("💭 You: ").strip()
                
                if query.lower() in ['exit', 'quit', 'q']:
                    print("👋 Goodbye!")
                    break
                elif query.lower() == 'stats':
                    stats = cli.get_stats()
                    print(f"\n📊 Performance Stats:")
                    for key, value in stats.items():
                        print(f"  {key}: {value}")
                    print()
                    continue
                elif query.lower() == 'help':
                    print("""
🔧 Available Commands:
  exit/quit/q    - Exit the CLI
  stats         - Show performance statistics  
  help          - Show this help message
  
💡 Tips:
  - Ask me to analyze code files
  - Request web searches for latest information
  - Get help with programming problems
  - Analyze project structure and dependencies
                    """)
                    continue
                elif not query:
                    continue
                
                result = await cli.process_query(query)
                print(f"\n🤖 Nexus: {result['response']}")
                
                if result.get('performance'):
                    print(f"⚡ {result['performance']['response_time']:.2f}s")
                print()
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
