#!/usr/bin/env python3
"""
Nexus CLI - Web-Enhanced RAG System with Small Language Model
Combines comprehensive web search with local LLM for personalized responses
"""

import requests
import json
import os
import time
import torch
import logging
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import quote
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Structured web search result"""
    title: str
    url: str
    content: str
    source: str
    relevance_score: float = 0.0
    category: str = "general"

@dataclass
class RAGContext:
    """Context for RAG generation"""
    query: str
    search_results: List[SearchResult]
    best_results: List[SearchResult]
    synthesized_context: str

class WebRAGSearchEngine:
    """
    Comprehensive web search engine for RAG system
    Searches multiple sources and ranks results by relevance
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Nexus-CLI/3.0) RAG-Enhanced System'
        })
        
        # Load API credentials
        self.github_token = os.getenv('GITHUB_TOKEN')
        
        # Search configuration
        self.max_results_per_source = 3
        self.timeout = 10
        
        # Result cache
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour
    
    def search_comprehensive(self, query: str) -> List[SearchResult]:
        """
        Perform comprehensive web search across multiple authoritative sources
        Returns ranked and filtered results for RAG processing
        """
        logger.info(f"🔍 Searching web for: {query}")
        
        # Check cache first
        cache_key = f"rag_search_{hash(query)}"
        if self._get_cached_result(cache_key):
            logger.info("📦 Using cached results")
            return self._get_cached_result(cache_key)
        
        all_results = []
        
        # Use ThreadPoolExecutor for parallel searching
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {
                'stackoverflow': executor.submit(self._search_stackoverflow, query),
                'documentation': executor.submit(self._search_documentation, query),
                'github': executor.submit(self._search_github, query),
                'packages': executor.submit(self._search_packages, query),
                'wikipedia': executor.submit(self._search_wikipedia, query),
                'duckduckgo': executor.submit(self._search_duckduckgo, query)
            }
            
            # Collect results as they complete
            for source_name, future in futures.items():
                try:
                    results = future.result(timeout=self.timeout)
                    if results:
                        all_results.extend(results)
                        logger.info(f"✅ {source_name}: {len(results)} results")
                except Exception as e:
                    logger.warning(f"❌ {source_name} search failed: {e}")
        
        # Rank and filter results
        ranked_results = self._rank_and_filter_results(query, all_results)
        
        # Cache results
        self._cache_result(cache_key, ranked_results)
        
        logger.info(f"🎯 Found {len(ranked_results)} high-quality results")
        return ranked_results
    
    def _search_stackoverflow(self, query: str) -> List[SearchResult]:
        """Search Stack Overflow for developer solutions"""
        try:
            params = {
                'order': 'desc',
                'sort': 'relevance',
                'q': query,
                'site': 'stackoverflow',
                'pagesize': self.max_results_per_source,
                'filter': 'withbody'
            }
            
            response = self.session.get(
                'https://api.stackexchange.com/2.3/search/advanced',
                params=params,
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                for item in data.get('items', []):
                    # Extract clean text from body
                    body = item.get('body_markdown', item.get('body', ''))[:1000]
                    
                    results.append(SearchResult(
                        title=item.get('title', ''),
                        url=item.get('link', ''),
                        content=f"Q: {item.get('title', '')}\n\n{body}",
                        source='Stack Overflow',
                        relevance_score=min(item.get('score', 0) / 20, 1.0),
                        category='community'
                    ))
                
                return results
                
        except Exception as e:
            logger.debug(f"Stack Overflow search failed: {e}")
        return []
    
    def _search_documentation(self, query: str) -> List[SearchResult]:
        """Search official documentation sources"""
        results = []
        
        # Detect technology and search appropriate docs
        query_lower = query.lower()
        
        # MDN for web technologies
        if any(term in query_lower for term in ['javascript', 'css', 'html', 'web', 'fetch', 'dom']):
            mdn_results = self._search_mdn(query)
            results.extend(mdn_results)
        
        # Python documentation
        if any(term in query_lower for term in ['python', 'django', 'flask', 'fastapi']):
            python_results = self._search_python_docs(query)
            results.extend(python_results)
        
        return results
    
    def _search_mdn(self, query: str) -> List[SearchResult]:
        """Search MDN Web Docs"""
        try:
            # Use DuckDuckGo site search for MDN
            search_query = f"site:developer.mozilla.org {query}"
            
            response = self.session.get(
                f"https://api.duckduckgo.com/?q={quote(search_query)}&format=json&no_html=1",
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                # Get abstract if available
                if data.get('Abstract') and 'mozilla' in data.get('AbstractURL', '').lower():
                    results.append(SearchResult(
                        title=f"MDN: {data.get('AbstractSource', 'Web Documentation')}",
                        url=data.get('AbstractURL', ''),
                        content=data.get('Abstract', ''),
                        source='MDN Web Docs',
                        relevance_score=0.9,
                        category='documentation'
                    ))
                
                # Get related topics
                for topic in data.get('RelatedTopics', [])[:2]:
                    if isinstance(topic, dict) and 'mozilla' in topic.get('FirstURL', ''):
                        results.append(SearchResult(
                            title=f"MDN: {topic.get('Text', '')[:50]}...",
                            url=topic.get('FirstURL', ''),
                            content=topic.get('Text', ''),
                            source='MDN Web Docs',
                            relevance_score=0.8,
                            category='documentation'
                        ))
                
                return results
                
        except Exception as e:
            logger.debug(f"MDN search failed: {e}")
        return []
    
    def _search_python_docs(self, query: str) -> List[SearchResult]:
        """Search Python official documentation"""
        try:
            # Create a helpful documentation reference
            return [SearchResult(
                title=f"Python Documentation: {query}",
                url=f"https://docs.python.org/3/search.html?q={quote(query)}",
                content=f"Official Python documentation and reference for: {query}. "
                       f"Check the official docs for the most accurate and up-to-date information.",
                source='Python Docs',
                relevance_score=0.85,
                category='documentation'
            )]
        except Exception:
            return []
    
    def _search_github(self, query: str) -> List[SearchResult]:
        """Search GitHub for code examples and repositories"""
        try:
            headers = {}
            if self.github_token:
                headers['Authorization'] = f'token {self.github_token}'
            
            # Search repositories
            response = self.session.get(
                f"https://api.github.com/search/repositories?q={quote(query)}&sort=stars&per_page=2",
                headers=headers,
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                for item in data.get('items', []):
                    description = item.get('description', 'No description')
                    language = item.get('language', 'Unknown')
                    stars = item.get('stargazers_count', 0)
                    
                    results.append(SearchResult(
                        title=f"GitHub: {item.get('full_name')} ({stars} ⭐)",
                        url=item.get('html_url', ''),
                        content=f"{description}\n\nLanguage: {language}\nStars: {stars}",
                        source='GitHub',
                        relevance_score=min(stars / 1000, 1.0),
                        category='code'
                    ))
                
                return results
                
        except Exception as e:
            logger.debug(f"GitHub search failed: {e}")
        return []
    
    def _search_packages(self, query: str) -> List[SearchResult]:
        """Search package registries"""
        results = []
        query_lower = query.lower()
        
        # NPM for JavaScript
        if any(term in query_lower for term in ['javascript', 'node', 'js', 'npm', 'react', 'vue']):
            npm_results = self._search_npm(query)
            results.extend(npm_results)
        
        # PyPI for Python
        if any(term in query_lower for term in ['python', 'pip', 'django', 'flask']):
            pypi_results = self._search_pypi(query)
            results.extend(pypi_results)
        
        return results
    
    def _search_npm(self, query: str) -> List[SearchResult]:
        """Search NPM registry"""
        try:
            response = self.session.get(
                f"https://registry.npmjs.org/-/v1/search?text={quote(query)}&size=2",
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                for obj in data.get('objects', []):
                    pkg = obj.get('package', {})
                    score = obj.get('score', {}).get('final', 0)
                    
                    results.append(SearchResult(
                        title=f"NPM: {pkg.get('name')}",
                        url=f"https://www.npmjs.com/package/{pkg.get('name')}",
                        content=f"{pkg.get('description', 'No description')}\n\nVersion: {pkg.get('version', 'Unknown')}",
                        source='NPM Registry',
                        relevance_score=score,
                        category='package'
                    ))
                
                return results
                
        except Exception as e:
            logger.debug(f"NPM search failed: {e}")
        return []
    
    def _search_pypi(self, query: str) -> List[SearchResult]:
        """Search PyPI registry"""
        try:
            # Try exact package name search
            response = self.session.get(f"https://pypi.org/pypi/{query}/json", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                info = data.get('info', {})
                
                return [SearchResult(
                    title=f"PyPI: {info.get('name')}",
                    url=info.get('project_url', f"https://pypi.org/project/{query}"),
                    content=f"{info.get('summary', 'No description')}\n\nVersion: {info.get('version', 'Unknown')}\nAuthor: {info.get('author', 'Unknown')}",
                    source='PyPI Registry',
                    relevance_score=0.8,
                    category='package'
                )]
                
        except Exception as e:
            logger.debug(f"PyPI search failed: {e}")
        return []
    
    def _search_wikipedia(self, query: str) -> List[SearchResult]:
        """Search Wikipedia for concepts"""
        try:
            # Search for articles
            search_response = self.session.get(
                f"https://en.wikipedia.org/api/rest_v1/page/search/{quote(query)}",
                timeout=5
            )
            
            if search_response.status_code == 200:
                search_data = search_response.json()
                pages = search_data.get('pages', [])
                
                if pages:
                    # Get summary of first result
                    page_title = pages[0].get('title')
                    summary_response = self.session.get(
                        f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(page_title)}",
                        timeout=5
                    )
                    
                    if summary_response.status_code == 200:
                        summary_data = summary_response.json()
                        
                        return [SearchResult(
                            title=f"Wikipedia: {summary_data.get('title', page_title)}",
                            url=summary_data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                            content=summary_data.get('extract', ''),
                            source='Wikipedia',
                            relevance_score=0.7,
                            category='concept'
                        )]
                        
        except Exception as e:
            logger.debug(f"Wikipedia search failed: {e}")
        return []
    
    def _search_duckduckgo(self, query: str) -> List[SearchResult]:
        """Search DuckDuckGo for general information"""
        try:
            response = self.session.get(
                f"https://api.duckduckgo.com/?q={quote(query)}&format=json&no_html=1&skip_disambig=1",
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                # Instant answer
                if data.get('Answer'):
                    results.append(SearchResult(
                        title="DuckDuckGo Instant Answer",
                        url='',
                        content=data.get('Answer'),
                        source='DuckDuckGo',
                        relevance_score=0.9,
                        category='general'
                    ))
                
                # Abstract
                if data.get('Abstract'):
                    results.append(SearchResult(
                        title=f"Overview: {data.get('AbstractSource', 'General')}",
                        url=data.get('AbstractURL', ''),
                        content=data.get('Abstract'),
                        source='DuckDuckGo',
                        relevance_score=0.8,
                        category='general'
                    ))
                
                return results
                
        except Exception as e:
            logger.debug(f"DuckDuckGo search failed: {e}")
        return []
    
    def _rank_and_filter_results(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Rank results by relevance and filter for quality"""
        if not results:
            return []
        
        # Calculate enhanced relevance scores
        for result in results:
            # Base relevance score from source
            base_score = result.relevance_score
            
            # Boost for query term matches in title and content
            query_terms = set(query.lower().split())
            title_terms = set(result.title.lower().split())
            content_terms = set(result.content.lower().split())
            
            title_overlap = len(query_terms.intersection(title_terms)) / len(query_terms)
            content_overlap = len(query_terms.intersection(content_terms)) / len(query_terms)
            
            # Category weights
            category_weights = {
                'documentation': 1.2,  # Prefer official docs
                'community': 1.0,      # Stack Overflow etc.
                'code': 0.9,           # GitHub repos
                'package': 0.8,        # Package registries
                'concept': 0.7,        # Wikipedia
                'general': 0.6         # General web search
            }
            
            category_weight = category_weights.get(result.category, 1.0)
            
            # Calculate final score
            result.relevance_score = (
                base_score * 0.4 +
                title_overlap * 0.3 +
                content_overlap * 0.2 +
                (category_weight - 1.0) * 0.1
            )
        
        # Sort by relevance and filter
        sorted_results = sorted(results, key=lambda x: x.relevance_score, reverse=True)
        
        # Filter out low-quality results
        filtered_results = [
            r for r in sorted_results 
            if r.relevance_score > 0.3 and len(r.content.strip()) > 20
        ]
        
        # Return top results
        return filtered_results[:8]  # Top 8 results for RAG context
    
    def _get_cached_result(self, key: str) -> Optional[List[SearchResult]]:
        """Get cached search results"""
        if key in self.cache:
            result, timestamp = self.cache[key]
            if time.time() - timestamp < self.cache_ttl:
                return result
        return None
    
    def _cache_result(self, key: str, result: List[SearchResult]):
        """Cache search results"""
        self.cache[key] = (result, time.time())


class SmallLanguageModel:
    """
    Small Language Model for RAG response generation
    Optimized for efficiency and quality
    """
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.max_length = 512
        
        logger.info(f"🧠 Initializing {model_name} on {self.device}")
        self._load_model()
    
    def _load_model(self):
        """Load the small language model"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            logger.info("📦 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Add pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("🤖 Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            logger.info(f"✅ Model loaded successfully on {self.device}")
            
        except ImportError:
            logger.error("❌ transformers library not found. Please install: pip install transformers torch")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def generate_response(self, context: RAGContext) -> str:
        """Generate response using RAG context"""
        if not self.model or not self.tokenizer:
            return "Model not available"
        
        try:
            # Create RAG prompt
            prompt = self._create_rag_prompt(context)
            
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=self.max_length)
            inputs = inputs.to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 150,  # Add 150 tokens for response
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=3
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part
            generated_response = response[len(prompt):].strip()
            
            # Clean up the response
            cleaned_response = self._clean_response(generated_response)
            
            return cleaned_response
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return f"I encountered an error generating a response: {e}"
    
    def _create_rag_prompt(self, context: RAGContext) -> str:
        """Create RAG prompt with web search context"""
        
        # Build context from search results
        context_parts = []
        
        for i, result in enumerate(context.best_results[:3], 1):
            context_parts.append(f"Source {i} ({result.source}): {result.content[:300]}...")
        
        context_text = "\n\n".join(context_parts)
        
        # Create prompt
        prompt = f"""Based on the following information from authoritative sources, provide a comprehensive and accurate answer to the user's question.

Context Information:
{context_text}

User Question: {context.query}

Please provide a clear, helpful answer based on the above information. Be specific and practical:"""

        return prompt
    
    def _clean_response(self, response: str) -> str:
        """Clean and format the generated response"""
        if not response:
            return "I couldn't generate a response for your question."
        
        # Remove common artifacts
        response = response.strip()
        
        # Remove repetitive patterns
        lines = response.split('\n')
        clean_lines = []
        prev_line = ""
        
        for line in lines:
            line = line.strip()
            if line and line != prev_line:  # Remove empty and duplicate lines
                clean_lines.append(line)
                prev_line = line
        
        cleaned = '\n'.join(clean_lines)
        
        # Ensure reasonable length
        if len(cleaned) > 800:
            sentences = cleaned.split('. ')
            cleaned = '. '.join(sentences[:4]) + '.'
        
        return cleaned if cleaned else "I couldn't generate a clear response for your question."


class WebRAGCLI:
    """
    Main CLI system combining web search with RAG generation
    """
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        logger.info("🚀 Initializing Nexus CLI Web-RAG System")
        
        self.search_engine = WebRAGSearchEngine()
        
        try:
            self.llm = SmallLanguageModel(model_name)
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            logger.info("💡 Falling back to rule-based responses")
            self.llm = None
    
    def process_query(self, query: str) -> str:
        """Main query processing pipeline"""
        logger.info(f"🎯 Processing query: {query}")
        
        try:
            # Step 1: Search the web
            search_results = self.search_engine.search_comprehensive(query)
            
            if not search_results:
                return "I couldn't find relevant information for your query. Please try rephrasing your question."
            
            # Step 2: Select best results for RAG context
            best_results = search_results[:3]  # Top 3 results
            
            # Step 3: Create synthesized context
            synthesized_context = self._synthesize_context(best_results)
            
            # Step 4: Create RAG context
            rag_context = RAGContext(
                query=query,
                search_results=search_results,
                best_results=best_results,
                synthesized_context=synthesized_context
            )
            
            # Step 5: Generate response
            if self.llm:
                response = self.llm.generate_response(rag_context)
            else:
                response = self._fallback_response(rag_context)
            
            # Step 6: Add sources
            response_with_sources = self._add_sources(response, best_results)
            
            logger.info("✅ Query processed successfully")
            return response_with_sources
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return f"I encountered an error processing your query: {e}"
    
    def _synthesize_context(self, results: List[SearchResult]) -> str:
        """Synthesize context from search results"""
        context_parts = []
        
        for result in results:
            context_parts.append(f"From {result.source}: {result.content}")
        
        return "\n\n".join(context_parts)
    
    def _fallback_response(self, context: RAGContext) -> str:
        """Fallback response when LLM is not available"""
        if not context.best_results:
            return "I couldn't find relevant information for your question."
        
        # Simple summarization
        best_result = context.best_results[0]
        
        response = f"Based on information from {best_result.source}:\n\n"
        response += best_result.content[:400]
        
        if len(best_result.content) > 400:
            response += "..."
        
        return response
    
    def _add_sources(self, response: str, results: List[SearchResult]) -> str:
        """Add source citations to response"""
        if not results:
            return response
        
        sources_section = "\n\n📚 **Sources:**\n"
        
        for i, result in enumerate(results, 1):
            if result.url:
                sources_section += f"{i}. [{result.source}]({result.url})\n"
            else:
                sources_section += f"{i}. {result.source}\n"
        
        return response + sources_section
    
    def interactive_mode(self):
        """Interactive CLI mode"""
        print("🌐 Nexus CLI - Web-Enhanced RAG System")
        print("=" * 50)
        print("Ask any question and I'll search the web and provide a comprehensive answer!")
        print("Type 'exit' to quit.\n")
        
        while True:
            try:
                query = input("🤔 Your question: ").strip()
                
                if query.lower() in ['exit', 'quit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not query:
                    continue
                
                print("\n🔍 Searching the web and generating response...")
                start_time = time.time()
                
                response = self.process_query(query)
                
                processing_time = time.time() - start_time
                
                print(f"\n{'='*60}")
                print(response)
                print(f"\n⏱️  Processed in {processing_time:.2f} seconds")
                print("="*60)
                print()
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Nexus CLI Web-Enhanced RAG System")
    parser.add_argument("query", nargs="*", help="Your question")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--model", "-m", default="microsoft/DialoGPT-medium", 
                       help="Language model to use")
    
    args = parser.parse_args()
    
    try:
        # Initialize the system
        cli = WebRAGCLI(model_name=args.model)
        
        if args.interactive:
            cli.interactive_mode()
        elif args.query:
            # Single query mode
            query = " ".join(args.query)
            print(f"🔍 Processing: {query}")
            print("="*60)
            
            start_time = time.time()
            response = cli.process_query(query)
            processing_time = time.time() - start_time
            
            print(response)
            print(f"\n⏱️  Processed in {processing_time:.2f} seconds")
        else:
            print("Usage:")
            print("  python web_rag_cli.py --interactive")
            print("  python web_rag_cli.py 'How does React work?'")
            print("  python web_rag_cli.py --model microsoft/DialoGPT-small 'What is Python?'")
            
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        print(f"❌ Failed to start: {e}")
        print("\nTo install required dependencies:")
        print("pip install torch transformers requests")


if __name__ == "__main__":
    main()
