#!/usr/bin/env python3
"""
Nexus CLI - Enhanced Web Access Setup
Enable complete web integration for data retrieval and code generation
"""

import os
import requests
from pathlib import Path

def setup_web_search_apis():
    """Setup additional web search APIs for comprehensive data access."""
    
    print("🌐 Setting up Enhanced Web Access for Nexus CLI")
    print("=" * 60)
    
    # 1. Google Custom Search (Free tier: 100 queries/day)
    print("\n1. 🔍 Google Custom Search API")
    print("   • Free: 100 searches/day")
    print("   • Get API key: https://developers.google.com/custom-search/v1/introduction")
    print("   • Set: GOOGLE_SEARCH_API_KEY=your_key")
    
    # 2. Serper API (Google Search alternative)
    print("\n2. 🔎 Serper API (Google Search)")
    print("   • Free: 2,500 searches/month")
    print("   • Get API key: https://serper.dev")
    print("   • Set: SERPER_API_KEY=your_key")
    
    # 3. DuckDuckGo Instant Answer API (Free, no key needed)
    print("\n3. 🦆 DuckDuckGo Instant Answer")
    print("   • Completely free, no API key needed")
    print("   • Already integrated!")
    
    # 4. GitHub API (Better rate limits with token)
    print("\n4. 🐙 GitHub API")
    print("   • Free: 60 requests/hour (unauth) or 5,000/hour (auth)")
    print("   • Get token: https://github.com/settings/tokens")
    print("   • Set: GITHUB_TOKEN=your_token")
    
    # 5. Real-time web scraping
    print("\n5. 🌐 Real-time Web Scraping")
    print("   • Uses requests + BeautifulSoup")
    print("   • Can scrape any public website")
    print("   • Already integrated!")
    
    print("\n" + "=" * 60)
    print("💡 To enable these, add API keys to your .env file:")
    print("   GOOGLE_SEARCH_API_KEY=your_key")
    print("   SERPER_API_KEY=your_key") 
    print("   GITHUB_TOKEN=your_token")

def test_current_web_access():
    """Test what web access is currently working."""
    
    print("\n🧪 Testing Current Web Access Capabilities")
    print("=" * 50)
    
    # Test Stack Overflow
    try:
        response = requests.get(
            "https://api.stackexchange.com/2.3/search/excerpts?order=desc&sort=relevance&q=python%20async&site=stackoverflow",
            timeout=5
        )
        if response.status_code == 200:
            print("✅ Stack Overflow API: Working")
        else:
            print("❌ Stack Overflow API: Failed")
    except:
        print("❌ Stack Overflow API: Connection failed")
    
    # Test GitHub API
    try:
        response = requests.get(
            "https://api.github.com/search/code?q=fastapi+example&sort=indexed&per_page=1",
            timeout=5
        )
        if response.status_code == 200:
            print("✅ GitHub Code Search: Working")
        else:
            print("❌ GitHub Code Search: Rate limited (get token for better access)")
    except:
        print("❌ GitHub Code Search: Connection failed")
    
    # Test NPM Registry
    try:
        response = requests.get("https://registry.npmjs.org/-/v1/search?text=react&size=1", timeout=5)
        if response.status_code == 200:
            print("✅ NPM Registry: Working")
        else:
            print("❌ NPM Registry: Failed")
    except:
        print("❌ NPM Registry: Connection failed")
    
    # Test PyPI
    try:
        response = requests.get("https://pypi.org/pypi/fastapi/json", timeout=5)
        if response.status_code == 200:
            print("✅ PyPI Registry: Working")
        else:
            print("❌ PyPI Registry: Failed")
    except:
        print("❌ PyPI Registry: Connection failed")
    
    # Test Wikipedia
    try:
        response = requests.get("https://en.wikipedia.org/api/rest_v1/page/summary/Machine_learning", timeout=5)
        if response.status_code == 200:
            print("✅ Wikipedia API: Working")
        else:
            print("❌ Wikipedia API: Failed")
    except:
        print("❌ Wikipedia API: Connection failed")

def create_enhanced_web_api():
    """Create enhanced web API integration."""
    
    return '''
class EnhancedWebAccess:
    """Complete web access for any data source."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Nexus-CLI/1.0) Enhanced Intelligence'
        })
        
        # API keys from environment
        self.google_api_key = os.getenv('GOOGLE_SEARCH_API_KEY')
        self.serper_api_key = os.getenv('SERPER_API_KEY')
        self.github_token = os.getenv('GITHUB_TOKEN')
    
    def search_web(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search the entire web for information."""
        
        results = []
        
        # Try Google Custom Search first (best quality)
        if self.google_api_key:
            google_results = self._search_google(query, max_results)
            results.extend(google_results)
        
        # Fallback to Serper API
        elif self.serper_api_key:
            serper_results = self._search_serper(query, max_results)
            results.extend(serper_results)
        
        # Free fallback: DuckDuckGo
        else:
            duckduckgo_results = self._search_duckduckgo(query, max_results)
            results.extend(duckduckgo_results)
        
        return results
    
    def scrape_webpage(self, url: str) -> str:
        """Scrape and extract content from any webpage."""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
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
            
        except Exception as e:
            return f"Failed to scrape {url}: {str(e)}"
    
    def get_latest_tech_news(self, topic: str) -> List[Dict]:
        """Get latest tech news about a specific topic."""
        query = f"{topic} programming development latest news"
        return self.search_web(query, max_results=3)
    
    def get_documentation(self, library: str, topic: str = "") -> str:
        """Get latest documentation for any library/framework."""
        query = f"{library} documentation {topic} official docs"
        results = self.search_web(query, max_results=2)
        
        if results:
            # Try to scrape the official docs
            for result in results:
                if 'docs' in result['url'] or 'documentation' in result['url']:
                    content = self.scrape_webpage(result['url'])
                    return content
        
        return "Documentation not found"
    
    def _search_google(self, query: str, max_results: int) -> List[Dict]:
        """Search using Google Custom Search API."""
        # Implementation with Google Custom Search
        pass
    
    def _search_serper(self, query: str, max_results: int) -> List[Dict]:
        """Search using Serper API."""
        # Implementation with Serper
        pass
    
    def _search_duckduckgo(self, query: str, max_results: int) -> List[Dict]:
        """Search using DuckDuckGo Instant Answer API."""
        # Free alternative - no API key needed
        pass
'''

if __name__ == "__main__":
    setup_web_search_apis()
    test_current_web_access()
    
    print("\n" + "=" * 60)
    print("🚀 Your Nexus CLI already has extensive web access!")
    print("💡 Add API keys above for even more comprehensive data access")
