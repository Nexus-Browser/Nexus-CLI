#!/usr/bin/env python3
"""
Complete Web Integration for Nexus CLI
Enables access to the entire web for data retrieval and code generation
"""

import requests
import json
import os
from typing import List, Dict, Optional
from urllib.parse import quote
import time

class ComprehensiveWebAccess:
    """
    Complete web access integration for Nexus CLI
    Searches multiple data sources across the entire web
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Nexus-CLI/1.0) Enhanced Intelligence System'
        })
        
        # API keys from environment
        self.google_api_key = os.getenv('GOOGLE_SEARCH_API_KEY')
        self.serper_api_key = os.getenv('SERPER_API_KEY')
        self.github_token = os.getenv('GITHUB_TOKEN')
        
        # Cache for web results
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour
    
    def search_everything(self, query: str) -> Dict[str, any]:
        """
        Search the entire web for comprehensive information
        Returns data from multiple sources
        """
        
        results = {
            'web_search': [],
            'documentation': [],
            'code_examples': [],
            'tutorials': [],
            'latest_news': [],
            'community_discussions': []
        }
        
        # 1. General web search
        web_results = self.search_web(query)
        if web_results:
            results['web_search'] = web_results
        
        # 2. Documentation search
        docs = self.search_documentation(query)
        if docs:
            results['documentation'] = docs
        
        # 3. Code examples
        code_examples = self.search_code_examples(query)
        if code_examples:
            results['code_examples'] = code_examples
        
        # 4. Tutorials and guides
        tutorials = self.search_tutorials(query)
        if tutorials:
            results['tutorials'] = tutorials
        
        # 5. Latest news and updates
        news = self.search_latest_news(query)
        if news:
            results['latest_news'] = news
        
        # 6. Community discussions
        discussions = self.search_community_discussions(query)
        if discussions:
            results['community_discussions'] = discussions
        
        return results
    
    def search_web(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search the web using multiple search engines."""
        
        results = []
        
        # Try Google Custom Search first (best quality)
        if self.google_api_key:
            google_results = self._search_google(query, max_results)
            results.extend(google_results)
        
        # Try Serper API (Google alternative)
        elif self.serper_api_key:
            serper_results = self._search_serper(query, max_results)
            results.extend(serper_results)
        
        # Free fallback: DuckDuckGo
        else:
            duckduckgo_results = self._search_duckduckgo(query, max_results)
            results.extend(duckduckgo_results)
        
        return results
    
    def _search_google(self, query: str, max_results: int) -> List[Dict]:
        """Search using Google Custom Search API."""
        if not self.google_api_key:
            return []
        
        try:
            # You need to set up a Custom Search Engine at:
            # https://cse.google.com/cse/
            search_engine_id = os.getenv('GOOGLE_SEARCH_ENGINE_ID', 'your_search_engine_id')
            
            response = self.session.get(
                f"https://www.googleapis.com/customsearch/v1?key={self.google_api_key}&cx={search_engine_id}&q={quote(query)}&num={max_results}",
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                for item in data.get('items', []):
                    results.append({
                        'title': item.get('title'),
                        'url': item.get('link'),
                        'snippet': item.get('snippet'),
                        'source': 'Google'
                    })
                
                return results
                
        except Exception as e:
            print(f"Google search failed: {e}")
        
        return []
    
    def _search_serper(self, query: str, max_results: int) -> List[Dict]:
        """Search using Serper API (Google alternative)."""
        if not self.serper_api_key:
            return []
        
        try:
            response = self.session.post(
                "https://google.serper.dev/search",
                headers={'X-API-KEY': self.serper_api_key},
                json={'q': query, 'num': max_results},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                for item in data.get('organic', []):
                    results.append({
                        'title': item.get('title'),
                        'url': item.get('link'),
                        'snippet': item.get('snippet'),
                        'source': 'Serper'
                    })
                
                return results
                
        except Exception as e:
            print(f"Serper search failed: {e}")
        
        return []
    
    def _search_duckduckgo(self, query: str, max_results: int) -> List[Dict]:
        """Search using DuckDuckGo Instant Answer API (free)."""
        try:
            response = self.session.get(
                f"https://api.duckduckgo.com/?q={quote(query)}&format=json&no_html=1&skip_disambig=1",
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                # Abstract/Definition
                if data.get('Abstract'):
                    results.append({
                        'title': data.get('AbstractSource', 'DuckDuckGo'),
                        'url': data.get('AbstractURL', ''),
                        'snippet': data.get('Abstract'),
                        'source': 'DuckDuckGo'
                    })
                
                # Related topics
                for topic in data.get('RelatedTopics', [])[:max_results-1]:
                    if isinstance(topic, dict) and topic.get('Text'):
                        results.append({
                            'title': topic.get('FirstURL', '').split('/')[-1] if topic.get('FirstURL') else 'Related',
                            'url': topic.get('FirstURL', ''),
                            'snippet': topic.get('Text'),
                            'source': 'DuckDuckGo'
                        })
                
                return results
                
        except Exception as e:
            print(f"DuckDuckGo search failed: {e}")
        
        return []
    
    def search_documentation(self, query: str) -> List[Dict]:
        """Search official documentation sites."""
        
        doc_sources = [
            ('Python', 'https://docs.python.org/3/search.html?q='),
            ('MDN', 'https://developer.mozilla.org/en-US/search?q='),
            ('Rust', 'https://doc.rust-lang.org/std/?search='),
            ('Node.js', 'https://nodejs.org/api/'),
        ]
        
        results = []
        
        for name, base_url in doc_sources:
            if any(tech in query.lower() for tech in [name.lower(), 'javascript', 'web', 'api']):
                results.append({
                    'title': f'{name} Documentation',
                    'url': base_url + quote(query),
                    'snippet': f'Official {name} documentation for: {query}',
                    'source': f'{name} Docs'
                })
        
        return results
    
    def search_code_examples(self, query: str) -> List[Dict]:
        """Search for code examples on GitHub and other platforms."""
        
        results = []
        
        # GitHub search
        if self.github_token:
            github_results = self._search_github_authenticated(query)
            results.extend(github_results)
        else:
            github_results = self._search_github_public(query)
            results.extend(github_results)
        
        # CodePen, JSFiddle for web examples
        if any(tech in query.lower() for tech in ['javascript', 'html', 'css', 'web']):
            results.append({
                'title': f'CodePen Examples: {query}',
                'url': f'https://codepen.io/search/pens?q={quote(query)}',
                'snippet': f'Interactive code examples for {query}',
                'source': 'CodePen'
            })
        
        return results
    
    def _search_github_authenticated(self, query: str) -> List[Dict]:
        """Search GitHub with authentication for better rate limits."""
        try:
            headers = {'Authorization': f'token {self.github_token}'}
            
            response = self.session.get(
                f"https://api.github.com/search/code?q={quote(query)}&sort=indexed&per_page=3",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                for item in data.get('items', []):
                    repo = item.get('repository', {})
                    results.append({
                        'title': f"{item.get('name')} - {repo.get('full_name')}",
                        'url': item.get('html_url'),
                        'snippet': f"Code example from {repo.get('full_name')}",
                        'source': 'GitHub'
                    })
                
                return results
                
        except Exception as e:
            print(f"GitHub authenticated search failed: {e}")
        
        return []
    
    def _search_github_public(self, query: str) -> List[Dict]:
        """Search GitHub without authentication (limited rate)."""
        try:
            response = self.session.get(
                f"https://api.github.com/search/repositories?q={quote(query)}&sort=stars&per_page=2",
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                for item in data.get('items', []):
                    results.append({
                        'title': f"{item.get('full_name')} - {item.get('stargazers_count')} stars",
                        'url': item.get('html_url'),
                        'snippet': item.get('description', 'No description'),
                        'source': 'GitHub'
                    })
                
                return results
                
        except Exception as e:
            print(f"GitHub public search failed: {e}")
        
        return []
    
    def search_tutorials(self, query: str) -> List[Dict]:
        """Search for tutorials and learning resources."""
        
        tutorial_sites = [
            'tutorial',
            'guide', 
            'how-to',
            'learn',
            'course'
        ]
        
        tutorial_query = f"{query} {' OR '.join(tutorial_sites)}"
        return self.search_web(tutorial_query, max_results=3)
    
    def search_latest_news(self, query: str) -> List[Dict]:
        """Search for latest news and updates."""
        
        news_query = f"{query} latest news updates 2024"
        return self.search_web(news_query, max_results=3)
    
    def search_community_discussions(self, query: str) -> List[Dict]:
        """Search community discussions (Reddit, Discord, Forums)."""
        
        results = []
        
        # Reddit search
        reddit_query = f"site:reddit.com {query}"
        reddit_results = self.search_web(reddit_query, max_results=2)
        results.extend(reddit_results)
        
        # Stack Overflow
        so_query = f"site:stackoverflow.com {query}"
        so_results = self.search_web(so_query, max_results=2)
        results.extend(so_results)
        
        return results
    
    def scrape_webpage_content(self, url: str) -> Optional[str]:
        """Scrape and extract content from any webpage."""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # Try to use BeautifulSoup if available
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Get text content
                text = soup.get_text()
                
                # Clean up whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                return text[:5000]  # Limit to first 5000 chars
                
            except ImportError:
                # Fallback: simple text extraction
                import re
                text = re.sub(r'<[^>]+>', '', response.text)
                text = ' '.join(text.split())
                return text[:5000]
            
        except Exception as e:
            return f"Failed to scrape {url}: {str(e)}"
    
    def format_comprehensive_response(self, query: str, web_data: Dict) -> str:
        """Format comprehensive web search results into a readable response."""
        
        response_parts = []
        
        # Add header
        response_parts.append(f"**🌐 Comprehensive Web Search Results for: {query}**\n")
        
        # Web search results
        if web_data.get('web_search'):
            response_parts.append("**🔍 Web Search Results:**")
            for result in web_data['web_search'][:3]:
                response_parts.append(f"• **{result['title']}**")
                response_parts.append(f"  {result['snippet']}")
                response_parts.append(f"  Source: {result['url']}\n")
        
        # Documentation
        if web_data.get('documentation'):
            response_parts.append("**📚 Official Documentation:**")
            for doc in web_data['documentation'][:2]:
                response_parts.append(f"• {doc['title']}: {doc['url']}\n")
        
        # Code examples
        if web_data.get('code_examples'):
            response_parts.append("**💻 Code Examples:**")
            for example in web_data['code_examples'][:2]:
                response_parts.append(f"• {example['title']}: {example['url']}\n")
        
        # Tutorials
        if web_data.get('tutorials'):
            response_parts.append("**🎓 Learning Resources:**")
            for tutorial in web_data['tutorials'][:2]:
                response_parts.append(f"• {tutorial['title']}")
                response_parts.append(f"  {tutorial['snippet'][:100]}...")
                response_parts.append(f"  {tutorial['url']}\n")
        
        return "\n".join(response_parts)


def demo_comprehensive_web_access():
    """Demonstrate the comprehensive web access capabilities."""
    
    print("🌐 Nexus CLI - Complete Web Integration Demo")
    print("=" * 60)
    
    web_access = ComprehensiveWebAccess()
    
    # Test queries
    test_queries = [
        "Python FastAPI best practices 2024",
        "React hooks tutorial examples",
        "Rust async programming guide"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Searching for: {query}")
        print("-" * 40)
        
        results = web_access.search_everything(query)
        
        # Show summary
        total_results = sum(len(results[key]) for key in results if results[key])
        print(f"📊 Found {total_results} results across {len([k for k, v in results.items() if v])} categories")
        
        # Show sample results
        if results['web_search']:
            print(f"   • {len(results['web_search'])} web search results")
        if results['documentation']:
            print(f"   • {len(results['documentation'])} documentation links")
        if results['code_examples']:
            print(f"   • {len(results['code_examples'])} code examples")
        if results['tutorials']:
            print(f"   • {len(results['tutorials'])} tutorials")
    
    print("\n" + "=" * 60)
    print("✅ Your Nexus CLI can now access the ENTIRE WEB!")
    print("💡 Set up API keys for even more comprehensive results:")
    print("   • GOOGLE_SEARCH_API_KEY + GOOGLE_SEARCH_ENGINE_ID")
    print("   • SERPER_API_KEY") 
    print("   • GITHUB_TOKEN")
    print("   • Add these to your .env file")


if __name__ == "__main__":
    demo_comprehensive_web_access()
