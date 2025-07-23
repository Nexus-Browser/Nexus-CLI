#!/usr/bin/env python3
"""
Nexus CLI - Lightweight Web Search System
Provides comprehensive web search without heavy ML dependencies
"""

import requests
import json
import os
import time
import logging
from typing import Dict, List, Optional, Any
from urllib.parse import quote
from dataclasses import dataclass
import argparse
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Web search result"""
    title: str
    url: str
    content: str
    source: str
    relevance_score: float = 0.0
    category: str = "general"

class LightweightWebSearch:
    """
    Lightweight web search system that works without ML dependencies
    Provides intelligent search result synthesis
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Nexus-CLI/3.0) Lightweight Web Search'
        })
        
        # Configuration
        self.timeout = 10
        self.max_results = 8
        
        # Result cache
        self.cache = {}
        self.cache_ttl = 3600
    
    def search_and_synthesize(self, query: str) -> str:
        """
        Main search function that returns a synthesized response
        """
        logger.info(f"🔍 Searching for: {query}")
        
        # Get search results
        results = self._comprehensive_search(query)
        
        if not results:
            return "I couldn't find relevant information for your query. Please try rephrasing your question."
        
        # Synthesize response
        response = self._synthesize_response(query, results)
        
        return response
    
    def _comprehensive_search(self, query: str) -> List[SearchResult]:
        """Perform comprehensive web search"""
        
        # Check cache
        cache_key = f"search_{hash(query)}"
        if self._get_cached_result(cache_key):
            return self._get_cached_result(cache_key)
        
        all_results = []
        
        # Search multiple sources
        sources = [
            ('duckduckgo', self._search_duckduckgo),
            ('stackoverflow', self._search_stackoverflow),
            ('wikipedia', self._search_wikipedia),
            ('github', self._search_github),
            ('documentation', self._search_documentation),
            ('packages', self._search_packages)
        ]
        
        for source_name, search_func in sources:
            try:
                results = search_func(query)
                if results:
                    all_results.extend(results)
                    logger.info(f"✅ {source_name}: {len(results)} results")
            except Exception as e:
                logger.debug(f"❌ {source_name} failed: {e}")
        
        # Rank and filter results
        ranked_results = self._rank_results(query, all_results)
        
        # Cache results
        self._cache_result(cache_key, ranked_results)
        
        return ranked_results
    
    def _search_duckduckgo(self, query: str) -> List[SearchResult]:
        """Search DuckDuckGo"""
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
                        title="Instant Answer",
                        url='',
                        content=data.get('Answer'),
                        source='DuckDuckGo',
                        relevance_score=0.95,
                        category='answer'
                    ))
                
                # Abstract
                if data.get('Abstract'):
                    results.append(SearchResult(
                        title=data.get('AbstractSource', 'Overview'),
                        url=data.get('AbstractURL', ''),
                        content=data.get('Abstract'),
                        source='DuckDuckGo',
                        relevance_score=0.9,
                        category='overview'
                    ))
                
                # Related topics
                for topic in data.get('RelatedTopics', [])[:3]:
                    if isinstance(topic, dict) and topic.get('Text'):
                        results.append(SearchResult(
                            title=topic.get('FirstURL', '').split('/')[-1] if topic.get('FirstURL') else 'Related',
                            url=topic.get('FirstURL', ''),
                            content=topic.get('Text'),
                            source='DuckDuckGo',
                            relevance_score=0.7,
                            category='related'
                        ))
                
                return results
                
        except Exception as e:
            logger.debug(f"DuckDuckGo search failed: {e}")
        return []
    
    def _search_stackoverflow(self, query: str) -> List[SearchResult]:
        """Search Stack Overflow"""
        try:
            params = {
                'order': 'desc',
                'sort': 'relevance',
                'q': query,
                'site': 'stackoverflow',
                'pagesize': 3,
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
                    # Clean body text
                    body = item.get('body_markdown', item.get('body', ''))
                    # Remove HTML tags and limit length
                    import re
                    clean_body = re.sub(r'<[^>]+>', '', body)[:500]
                    
                    answer_status = ""
                    if item.get('accepted_answer_id'):
                        answer_status = " ✓"
                    elif item.get('answer_count', 0) > 0:
                        answer_status = f" ({item['answer_count']} answers)"
                    
                    results.append(SearchResult(
                        title=f"{item.get('title', '')}{answer_status}",
                        url=item.get('link', ''),
                        content=clean_body,
                        source='Stack Overflow',
                        relevance_score=min(item.get('score', 0) / 10, 1.0),
                        category='community'
                    ))
                
                return results
                
        except Exception as e:
            logger.debug(f"Stack Overflow search failed: {e}")
        return []
    
    def _search_wikipedia(self, query: str) -> List[SearchResult]:
        """Search Wikipedia"""
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
                            relevance_score=0.8,
                            category='encyclopedia'
                        )]
                        
        except Exception as e:
            logger.debug(f"Wikipedia search failed: {e}")
        return []
    
    def _search_github(self, query: str) -> List[SearchResult]:
        """Search GitHub repositories"""
        try:
            headers = {}
            github_token = os.getenv('GITHUB_TOKEN')
            if github_token:
                headers['Authorization'] = f'token {github_token}'
            
            response = self.session.get(
                f"https://api.github.com/search/repositories?q={quote(query)}&sort=stars&per_page=2",
                headers=headers,
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                for item in data.get('items', []):
                    stars = item.get('stargazers_count', 0)
                    language = item.get('language', 'Unknown')
                    description = item.get('description', 'No description')
                    
                    results.append(SearchResult(
                        title=f"GitHub: {item.get('full_name')} ({stars}⭐)",
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
    
    def _search_documentation(self, query: str) -> List[SearchResult]:
        """Search documentation sites"""
        query_lower = query.lower()
        results = []
        
        # Technology-specific documentation
        if any(term in query_lower for term in ['javascript', 'html', 'css', 'web']):
            results.append(SearchResult(
                title="MDN Web Docs",
                url=f"https://developer.mozilla.org/en-US/search?q={quote(query)}",
                content=f"Official Mozilla documentation for web technologies: {query}",
                source='MDN',
                relevance_score=0.9,
                category='documentation'
            ))
        
        if any(term in query_lower for term in ['python']):
            results.append(SearchResult(
                title="Python Documentation",
                url=f"https://docs.python.org/3/search.html?q={quote(query)}",
                content=f"Official Python documentation: {query}",
                source='Python Docs',
                relevance_score=0.9,
                category='documentation'
            ))
        
        return results
    
    def _search_packages(self, query: str) -> List[SearchResult]:
        """Search package registries"""
        results = []
        query_lower = query.lower()
        
        # NPM for JavaScript
        if any(term in query_lower for term in ['javascript', 'node', 'js', 'npm']):
            try:
                response = self.session.get(
                    f"https://registry.npmjs.org/-/v1/search?text={quote(query)}&size=1",
                    timeout=5
                )
                
                if response.status_code == 200:
                    data = response.json()
                    for obj in data.get('objects', []):
                        pkg = obj.get('package', {})
                        results.append(SearchResult(
                            title=f"NPM: {pkg.get('name')}",
                            url=f"https://www.npmjs.com/package/{pkg.get('name')}",
                            content=f"{pkg.get('description', 'No description')}\nVersion: {pkg.get('version', 'Unknown')}",
                            source='NPM Registry',
                            relevance_score=obj.get('score', {}).get('final', 0),
                            category='package'
                        ))
            except Exception:
                pass
        
        # PyPI for Python
        if any(term in query_lower for term in ['python', 'pip']):
            try:
                response = self.session.get(f"https://pypi.org/pypi/{query}/json", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    info = data.get('info', {})
                    results.append(SearchResult(
                        title=f"PyPI: {info.get('name')}",
                        url=info.get('project_url', f"https://pypi.org/project/{query}"),
                        content=f"{info.get('summary', 'No description')}\nVersion: {info.get('version', 'Unknown')}",
                        source='PyPI Registry',
                        relevance_score=0.8,
                        category='package'
                    ))
            except Exception:
                pass
        
        return results
    
    def _rank_results(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Rank and filter search results"""
        if not results:
            return []
        
        # Calculate relevance scores
        query_terms = set(query.lower().split())
        
        for result in results:
            # Base score from source
            base_score = result.relevance_score
            
            # Term matching bonus
            title_terms = set(result.title.lower().split())
            content_terms = set(result.content.lower().split())
            
            title_overlap = len(query_terms.intersection(title_terms)) / len(query_terms) if query_terms else 0
            content_overlap = len(query_terms.intersection(content_terms)) / len(query_terms) if query_terms else 0
            
            # Category weights
            category_weights = {
                'answer': 1.3,
                'documentation': 1.2,
                'overview': 1.1,
                'community': 1.0,
                'code': 0.9,
                'package': 0.8,
                'encyclopedia': 0.7,
                'related': 0.6
            }
            
            category_weight = category_weights.get(result.category, 1.0)
            
            # Final score
            result.relevance_score = (
                base_score * 0.4 +
                title_overlap * 0.3 +
                content_overlap * 0.2 +
                (category_weight - 1.0) * 0.1
            )
        
        # Sort and filter
        sorted_results = sorted(results, key=lambda x: x.relevance_score, reverse=True)
        return [r for r in sorted_results if r.relevance_score > 0.3][:self.max_results]
    
    def _synthesize_response(self, query: str, results: List[SearchResult]) -> str:
        """Synthesize a comprehensive response from search results"""
        
        if not results:
            return "No relevant information found."
        
        # Start with the best result
        best_result = results[0]
        
        # Build response
        response_parts = []
        
        # Main answer section
        if best_result.category == 'answer' and best_result.content:
            response_parts.append(f"**Answer:** {best_result.content}")
        elif best_result.category == 'overview':
            response_parts.append(f"**Overview:** {best_result.content[:400]}...")
        else:
            response_parts.append(f"**Key Information:** {best_result.content[:300]}...")
        
        # Add additional context from other high-quality results
        documentation_results = [r for r in results if r.category == 'documentation']
        code_results = [r for r in results if r.category == 'code']
        community_results = [r for r in results if r.category == 'community']
        
        # Documentation section
        if documentation_results:
            response_parts.append(f"\n**📚 Documentation:** {documentation_results[0].title} - {documentation_results[0].content[:200]}...")
        
        # Code examples section
        if code_results:
            response_parts.append(f"\n**💻 Code Examples:** {code_results[0].title} - {code_results[0].content[:150]}...")
        
        # Community insights
        if community_results:
            response_parts.append(f"\n**💬 Community Discussion:** {community_results[0].title[:80]}...")
        
        # Add sources
        response_parts.append("\n**📚 Sources:**")
        for i, result in enumerate(results[:3], 1):
            if result.url:
                response_parts.append(f"{i}. [{result.source}]({result.url})")
            else:
                response_parts.append(f"{i}. {result.source}")
        
        return "\n".join(response_parts)
    
    def _get_cached_result(self, key: str) -> Optional[List[SearchResult]]:
        """Get cached results"""
        if key in self.cache:
            result, timestamp = self.cache[key]
            if time.time() - timestamp < self.cache_ttl:
                return result
        return None
    
    def _cache_result(self, key: str, result: List[SearchResult]):
        """Cache results"""
        self.cache[key] = (result, time.time())
    
    def interactive_mode(self):
        """Interactive search mode"""
        print("🌐 Nexus CLI - Lightweight Web Search")
        print("=" * 50)
        print("Ask any question and I'll search the web for comprehensive answers!")
        print("Type 'exit' to quit.\n")
        
        while True:
            try:
                query = input("🤔 Your question: ").strip()
                
                if query.lower() in ['exit', 'quit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not query:
                    continue
                
                print("\n🔍 Searching...")
                start_time = time.time()
                
                response = self.search_and_synthesize(query)
                
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
    parser = argparse.ArgumentParser(description="Nexus CLI Lightweight Web Search")
    parser.add_argument("query", nargs="*", help="Your search query")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    search_system = LightweightWebSearch()
    
    if args.interactive:
        search_system.interactive_mode()
    elif args.query:
        query = " ".join(args.query)
        print(f"🔍 Searching for: {query}")
        print("="*60)
        
        start_time = time.time()
        response = search_system.search_and_synthesize(query)
        processing_time = time.time() - start_time
        
        print(response)
        print(f"\n⏱️  Processed in {processing_time:.2f} seconds")
    else:
        print("Usage:")
        print("  python lightweight_web_search.py --interactive")
        print("  python lightweight_web_search.py 'How does Python work?'")

if __name__ == "__main__":
    main()
