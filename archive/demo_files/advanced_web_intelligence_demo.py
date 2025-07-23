#!/usr/bin/env python3
"""
Nexus CLI - Advanced Web Integration Demonstration
Shows the complete web-powered capabilities with real-world examples
"""

import requests
import json
import os
import time
from typing import Dict, List, Optional, Any
from urllib.parse import quote
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class WebSearchResult:
    """Structured representation of web search results"""
    title: str
    url: str
    snippet: str
    source: str
    relevance_score: float = 0.0

class AdvancedWebIntelligence:
    """
    Production-ready web intelligence system for Nexus CLI
    Integrates multiple APIs and data sources for comprehensive information retrieval
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Nexus-CLI/2.0) Advanced Intelligence System'
        })
        
        # Load API credentials from environment
        self.google_api_key = os.getenv('GOOGLE_SEARCH_API_KEY')
        self.google_engine_id = os.getenv('GOOGLE_SEARCH_ENGINE_ID')
        self.serper_api_key = os.getenv('SERPER_API_KEY')
        self.github_token = os.getenv('GITHUB_TOKEN')
        
        # Result cache with TTL
        self.cache = {}
        self.cache_ttl = 3600
        
        # Rate limiting
        self.last_request_times = {}
        self.min_request_interval = 0.1  # 100ms between requests
        
    def search_comprehensive(self, query: str, max_results: int = 10) -> Dict[str, List[WebSearchResult]]:
        """
        Perform comprehensive web search across multiple sources
        Returns categorized results from different APIs
        """
        
        # Check cache first
        cache_key = f"comprehensive_{hash(query)}"
        if self._get_cached_result(cache_key):
            return self._get_cached_result(cache_key)
        
        results = {
            'web_search': [],
            'documentation': [],
            'code_examples': [],
            'community_discussions': [],
            'packages': [],
            'technical_resources': []
        }
        
        # Use ThreadPoolExecutor for parallel API calls
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {}
            
            # Submit all search tasks
            if self.google_api_key and self.google_engine_id:
                futures['google'] = executor.submit(self._search_google, query, max_results // 2)
            elif self.serper_api_key:
                futures['serper'] = executor.submit(self._search_serper, query, max_results // 2)
            else:
                futures['duckduckgo'] = executor.submit(self._search_duckduckgo, query, max_results // 2)
            
            futures['stackoverflow'] = executor.submit(self._search_stackoverflow, query)
            futures['github'] = executor.submit(self._search_github_comprehensive, query)
            futures['documentation'] = executor.submit(self._search_documentation_sites, query)
            futures['packages'] = executor.submit(self._search_package_registries, query)
            
            # Collect results as they complete
            for future_name, future in futures.items():
                try:
                    result = future.result(timeout=10)
                    if future_name in ['google', 'serper', 'duckduckgo']:
                        results['web_search'].extend(result)
                    elif future_name == 'stackoverflow':
                        results['community_discussions'].extend(result)
                    elif future_name == 'github':
                        results['code_examples'].extend(result)
                    elif future_name == 'documentation':
                        results['documentation'].extend(result)
                    elif future_name == 'packages':
                        results['packages'].extend(result)
                except Exception as e:
                    logger.warning(f"Search {future_name} failed: {e}")
        
        # Cache results
        self._cache_result(cache_key, results)
        
        return results
    
    def _search_google(self, query: str, max_results: int) -> List[WebSearchResult]:
        """Search using Google Custom Search API with advanced parameters"""
        if not self.google_api_key or not self.google_engine_id:
            return []
        
        try:
            self._rate_limit('google')
            
            params = {
                'key': self.google_api_key,
                'cx': self.google_engine_id,
                'q': query,
                'num': min(max_results, 10),
                'safe': 'active',
                'fields': 'items(title,link,snippet,displayLink)'
            }
            
            response = self.session.get(
                'https://www.googleapis.com/customsearch/v1',
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                for item in data.get('items', []):
                    results.append(WebSearchResult(
                        title=item.get('title', ''),
                        url=item.get('link', ''),
                        snippet=item.get('snippet', ''),
                        source=f"Google ({item.get('displayLink', '')})",
                        relevance_score=self._calculate_relevance(query, item.get('title', '') + ' ' + item.get('snippet', ''))
                    ))
                
                return sorted(results, key=lambda x: x.relevance_score, reverse=True)
                
        except Exception as e:
            logger.warning(f"Google search failed: {e}")
        
        return []
    
    def _search_serper(self, query: str, max_results: int) -> List[WebSearchResult]:
        """Search using Serper API (Google alternative) with enhanced parameters"""
        if not self.serper_api_key:
            return []
        
        try:
            self._rate_limit('serper')
            
            payload = {
                'q': query,
                'num': min(max_results, 10),
                'hl': 'en',
                'gl': 'us'
            }
            
            response = self.session.post(
                'https://google.serper.dev/search',
                headers={'X-API-KEY': self.serper_api_key},
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                for item in data.get('organic', []):
                    results.append(WebSearchResult(
                        title=item.get('title', ''),
                        url=item.get('link', ''),
                        snippet=item.get('snippet', ''),
                        source='Serper',
                        relevance_score=self._calculate_relevance(query, item.get('title', '') + ' ' + item.get('snippet', ''))
                    ))
                
                return sorted(results, key=lambda x: x.relevance_score, reverse=True)
                
        except Exception as e:
            logger.warning(f"Serper search failed: {e}")
        
        return []
    
    def _search_duckduckgo(self, query: str, max_results: int) -> List[WebSearchResult]:
        """Enhanced DuckDuckGo search with instant answers and related topics"""
        try:
            self._rate_limit('duckduckgo')
            
            response = self.session.get(
                f"https://api.duckduckgo.com/?q={quote(query)}&format=json&no_html=1&skip_disambig=1",
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                # Main abstract/definition
                if data.get('Abstract'):
                    results.append(WebSearchResult(
                        title=data.get('AbstractSource', 'DuckDuckGo'),
                        url=data.get('AbstractURL', ''),
                        snippet=data.get('Abstract'),
                        source='DuckDuckGo Abstract',
                        relevance_score=0.9
                    ))
                
                # Instant answer
                if data.get('Answer'):
                    results.append(WebSearchResult(
                        title='Instant Answer',
                        url='',
                        snippet=data.get('Answer'),
                        source='DuckDuckGo Answer',
                        relevance_score=0.95
                    ))
                
                # Related topics
                for topic in data.get('RelatedTopics', [])[:max_results-len(results)]:
                    if isinstance(topic, dict) and topic.get('Text'):
                        results.append(WebSearchResult(
                            title=topic.get('FirstURL', '').split('/')[-1] if topic.get('FirstURL') else 'Related',
                            url=topic.get('FirstURL', ''),
                            snippet=topic.get('Text'),
                            source='DuckDuckGo Related',
                            relevance_score=0.7
                        ))
                
                return results
                
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}")
        
        return []
    
    def _search_stackoverflow(self, query: str) -> List[WebSearchResult]:
        """Advanced Stack Overflow search with answer scores and tags"""
        try:
            self._rate_limit('stackoverflow')
            
            # Search questions first
            params = {
                'order': 'desc',
                'sort': 'relevance',
                'q': query,
                'site': 'stackoverflow',
                'pagesize': 5,
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
                    # Get accepted answer if available
                    answer_info = ""
                    if item.get('accepted_answer_id'):
                        answer_info = " ✓ Accepted Answer"
                    elif item.get('answer_count', 0) > 0:
                        answer_info = f" ({item['answer_count']} answers)"
                    
                    # Format tags
                    tags = ", ".join(item.get('tags', [])[:3])
                    
                    results.append(WebSearchResult(
                        title=f"{item.get('title', '')} {answer_info}",
                        url=item.get('link', ''),
                        snippet=f"Tags: {tags} | Score: {item.get('score', 0)} | Views: {item.get('view_count', 0)}",
                        source='Stack Overflow',
                        relevance_score=min(item.get('score', 0) / 10, 1.0)
                    ))
                
                return sorted(results, key=lambda x: x.relevance_score, reverse=True)
                
        except Exception as e:
            logger.warning(f"Stack Overflow search failed: {e}")
        
        return []
    
    def _search_github_comprehensive(self, query: str) -> List[WebSearchResult]:
        """Comprehensive GitHub search including repositories, code, and issues"""
        results = []
        
        try:
            headers = {}
            if self.github_token:
                headers['Authorization'] = f'token {self.github_token}'
            
            self._rate_limit('github')
            
            # Search repositories
            repo_response = self.session.get(
                f"https://api.github.com/search/repositories?q={quote(query)}&sort=stars&per_page=3",
                headers=headers,
                timeout=10
            )
            
            if repo_response.status_code == 200:
                repo_data = repo_response.json()
                for item in repo_data.get('items', []):
                    results.append(WebSearchResult(
                        title=f"📁 {item.get('full_name')} ({item.get('stargazers_count')} ⭐)",
                        url=item.get('html_url', ''),
                        snippet=item.get('description', 'No description') + f" | Language: {item.get('language', 'N/A')}",
                        source='GitHub Repository',
                        relevance_score=min(item.get('stargazers_count', 0) / 1000, 1.0)
                    ))
            
            # Search code examples
            code_response = self.session.get(
                f"https://api.github.com/search/code?q={quote(query)}&sort=indexed&per_page=2",
                headers=headers,
                timeout=10
            )
            
            if code_response.status_code == 200:
                code_data = code_response.json()
                for item in code_data.get('items', []):
                    repo = item.get('repository', {})
                    results.append(WebSearchResult(
                        title=f"💻 {item.get('name')} - {repo.get('full_name')}",
                        url=item.get('html_url', ''),
                        snippet=f"Code example from {repo.get('full_name')} | Language: {repo.get('language', 'N/A')}",
                        source='GitHub Code',
                        relevance_score=0.8
                    ))
                    
        except Exception as e:
            logger.warning(f"GitHub search failed: {e}")
        
        return results
    
    def _search_documentation_sites(self, query: str) -> List[WebSearchResult]:
        """Search official documentation sites"""
        doc_sources = [
            ('MDN', 'https://developer.mozilla.org/en-US/search?q='),
            ('Python Docs', 'https://docs.python.org/3/search.html?q='),
            ('React Docs', 'https://react.dev/learn?q='),
            ('Node.js Docs', 'https://nodejs.org/api/'),
            ('Rust Docs', 'https://doc.rust-lang.org/std/?search='),
        ]
        
        results = []
        query_lower = query.lower()
        
        # Determine relevant documentation based on query
        for name, base_url in doc_sources:
            relevance = 0.0
            
            if 'javascript' in query_lower or 'html' in query_lower or 'css' in query_lower:
                if 'MDN' in name:
                    relevance = 0.9
            elif 'python' in query_lower:
                if 'Python' in name:
                    relevance = 0.9
            elif 'react' in query_lower:
                if 'React' in name:
                    relevance = 0.9
            elif 'node' in query_lower:
                if 'Node.js' in name:
                    relevance = 0.9
            elif 'rust' in query_lower:
                if 'Rust' in name:
                    relevance = 0.9
            else:
                relevance = 0.3  # General relevance
            
            if relevance > 0.3:
                results.append(WebSearchResult(
                    title=f"📚 {name} Documentation",
                    url=base_url + quote(query),
                    snippet=f"Official {name} documentation for: {query}",
                    source=f"{name} Docs",
                    relevance_score=relevance
                ))
        
        return sorted(results, key=lambda x: x.relevance_score, reverse=True)
    
    def _search_package_registries(self, query: str) -> List[WebSearchResult]:
        """Search package registries (NPM, PyPI, Crates.io)"""
        results = []
        
        # Determine which registries to search based on query
        query_lower = query.lower()
        
        try:
            # NPM Registry
            if any(term in query_lower for term in ['javascript', 'node', 'npm', 'js', 'typescript']):
                npm_results = self._search_npm_registry(query)
                results.extend(npm_results)
            
            # PyPI Registry
            if any(term in query_lower for term in ['python', 'pip', 'pypi']):
                pypi_results = self._search_pypi_registry(query)
                results.extend(pypi_results)
            
            # Crates.io Registry
            if any(term in query_lower for term in ['rust', 'cargo', 'crate']):
                crates_results = self._search_crates_registry(query)
                results.extend(crates_results)
                
        except Exception as e:
            logger.warning(f"Package registry search failed: {e}")
        
        return results
    
    def _search_npm_registry(self, query: str) -> List[WebSearchResult]:
        """Search NPM registry with detailed package information"""
        try:
            response = self.session.get(
                f"https://registry.npmjs.org/-/v1/search?text={quote(query)}&size=3",
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                for obj in data.get('objects', []):
                    pkg = obj.get('package', {})
                    results.append(WebSearchResult(
                        title=f"📦 {pkg.get('name')} (NPM)",
                        url=f"https://www.npmjs.com/package/{pkg.get('name')}",
                        snippet=f"{pkg.get('description', 'No description')} | Downloads: {obj.get('score', {}).get('detail', {}).get('popularity', 0):.2f}",
                        source='NPM Registry',
                        relevance_score=obj.get('score', {}).get('final', 0)
                    ))
                
                return results
                
        except Exception as e:
            logger.warning(f"NPM search failed: {e}")
        
        return []
    
    def _search_pypi_registry(self, query: str) -> List[WebSearchResult]:
        """Search PyPI registry with package metadata"""
        try:
            # First, try exact match
            response = self.session.get(
                f"https://pypi.org/pypi/{query}/json",
                timeout=5
            )
            
            results = []
            
            if response.status_code == 200:
                data = response.json()
                info = data.get('info', {})
                results.append(WebSearchResult(
                    title=f"🐍 {info.get('name')} (PyPI)",
                    url=info.get('project_url', f"https://pypi.org/project/{query}"),
                    snippet=f"{info.get('summary', 'No description')} | Version: {info.get('version')}",
                    source='PyPI Registry',
                    relevance_score=0.9
                ))
            
            return results
            
        except Exception as e:
            logger.warning(f"PyPI search failed: {e}")
        
        return []
    
    def _search_crates_registry(self, query: str) -> List[WebSearchResult]:
        """Search Crates.io registry for Rust packages"""
        try:
            response = self.session.get(
                f"https://crates.io/api/v1/crates?q={quote(query)}&per_page=3",
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                for crate in data.get('crates', []):
                    results.append(WebSearchResult(
                        title=f"🦀 {crate.get('name')} (Crates.io)",
                        url=f"https://crates.io/crates/{crate.get('name')}",
                        snippet=f"{crate.get('description', 'No description')} | Downloads: {crate.get('downloads', 0)}",
                        source='Crates.io Registry',
                        relevance_score=min(crate.get('downloads', 0) / 10000, 1.0)
                    ))
                
                return results
                
        except Exception as e:
            logger.warning(f"Crates.io search failed: {e}")
        
        return []
    
    def _calculate_relevance(self, query: str, text: str) -> float:
        """Calculate relevance score based on query terms in text"""
        query_terms = set(query.lower().split())
        text_terms = set(text.lower().split())
        
        if not query_terms:
            return 0.0
        
        # Calculate term overlap
        overlap = len(query_terms.intersection(text_terms))
        relevance = overlap / len(query_terms)
        
        # Boost for exact phrase matches
        if query.lower() in text.lower():
            relevance += 0.3
        
        return min(relevance, 1.0)
    
    def _rate_limit(self, service: str):
        """Simple rate limiting to avoid hitting API limits"""
        current_time = time.time()
        last_time = self.last_request_times.get(service, 0)
        
        if current_time - last_time < self.min_request_interval:
            time.sleep(self.min_request_interval - (current_time - last_time))
        
        self.last_request_times[service] = time.time()
    
    def _get_cached_result(self, key: str) -> Optional[Dict]:
        """Get cached result if still valid"""
        if key in self.cache:
            result, timestamp = self.cache[key]
            if time.time() - timestamp < self.cache_ttl:
                return result
        return None
    
    def _cache_result(self, key: str, result: Dict):
        """Cache result with timestamp"""
        self.cache[key] = (result, time.time())
    
    def format_comprehensive_response(self, query: str, results: Dict[str, List[WebSearchResult]]) -> str:
        """Format search results into a comprehensive, readable response"""
        response_parts = []
        
        # Header
        response_parts.append(f"🌐 **Comprehensive Web Intelligence Report for: {query}**\n")
        
        # Statistics
        total_results = sum(len(category_results) for category_results in results.values())
        active_sources = len([k for k, v in results.items() if v])
        response_parts.append(f"📊 **Summary:** {total_results} results from {active_sources} categories\n")
        
        # Web Search Results
        if results.get('web_search'):
            response_parts.append("🔍 **Web Search Results:**")
            for result in results['web_search'][:3]:
                response_parts.append(f"• **{result.title}**")
                response_parts.append(f"  {result.snippet}")
                response_parts.append(f"  🔗 {result.url}")
                response_parts.append(f"  📍 Source: {result.source} | Relevance: {result.relevance_score:.2f}\n")
        
        # Documentation
        if results.get('documentation'):
            response_parts.append("📚 **Official Documentation:**")
            for doc in results['documentation'][:2]:
                response_parts.append(f"• {doc.title}")
                response_parts.append(f"  🔗 {doc.url}\n")
        
        # Code Examples
        if results.get('code_examples'):
            response_parts.append("💻 **Code Examples & Repositories:**")
            for example in results['code_examples'][:3]:
                response_parts.append(f"• {example.title}")
                response_parts.append(f"  {example.snippet}")
                response_parts.append(f"  🔗 {example.url}\n")
        
        # Packages
        if results.get('packages'):
            response_parts.append("📦 **Packages & Libraries:**")
            for package in results['packages'][:2]:
                response_parts.append(f"• {package.title}")
                response_parts.append(f"  {package.snippet}")
                response_parts.append(f"  🔗 {package.url}\n")
        
        # Community Discussions
        if results.get('community_discussions'):
            response_parts.append("💬 **Community Discussions:**")
            for discussion in results['community_discussions'][:2]:
                response_parts.append(f"• {discussion.title}")
                response_parts.append(f"  {discussion.snippet}")
                response_parts.append(f"  🔗 {discussion.url}\n")
        
        # Footer with API status
        api_status = []
        if self.google_api_key:
            api_status.append("Google ✅")
        if self.serper_api_key:
            api_status.append("Serper ✅")
        if self.github_token:
            api_status.append("GitHub ✅")
        
        response_parts.append(f"🔧 **API Status:** {', '.join(api_status) if api_status else 'Free APIs Only'}")
        
        return "\n".join(response_parts)


def demo_advanced_web_intelligence():
    """Demonstrate the advanced web intelligence capabilities"""
    
    print("🚀 Nexus CLI - Advanced Web Intelligence Demo")
    print("=" * 60)
    
    web_intel = AdvancedWebIntelligence()
    
    # Test queries that showcase different capabilities
    test_queries = [
        "React hooks TypeScript examples",
        "Python FastAPI authentication best practices",
        "Rust async programming tutorial",
        "Node.js performance optimization",
        "Docker microservices architecture"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test {i}: Searching for '{query}'")
        print("-" * 50)
        
        start_time = time.time()
        results = web_intel.search_comprehensive(query, max_results=15)
        search_time = time.time() - start_time
        
        # Show comprehensive formatted response
        formatted_response = web_intel.format_comprehensive_response(query, results)
        print(formatted_response)
        
        print(f"⏱️  Search completed in {search_time:.2f} seconds")
        print("\n" + "="*60)
    
    print("\n🎉 Advanced Web Intelligence Demo Complete!")
    print("Your Nexus CLI now has access to the entire web with:")
    print("• Multi-source parallel search")
    print("• Intelligent relevance scoring")
    print("• Rate limiting and caching")
    print("• Comprehensive result formatting")


if __name__ == "__main__":
    demo_advanced_web_intelligence()
