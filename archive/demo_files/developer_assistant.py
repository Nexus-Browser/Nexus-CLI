#!/usr/bin/env python3
"""
Nexus CLI - Natural Language Developer Assistant
Real-time web-powered developer question answering system
"""

import requests
import json
import os
import time
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any, Union
from urllib.parse import quote, urljoin
import logging
from dataclasses import dataclass
from enum import Enum
import re
from bs4 import BeautifulSoup
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APISource(Enum):
    """Available API sources for developer information"""
    MDN = "mdn"
    DEVDOCS = "devdocs"
    WIKIPEDIA = "wikipedia"
    NPM = "npm"
    PYPI = "pypi"
    GITHUB = "github"
    STACKOVERFLOW = "stackoverflow"
    DUCKDUCKGO = "duckduckgo"

@dataclass
class DeveloperQuery:
    """Structured representation of a developer question"""
    original_question: str
    normalized_query: str
    detected_technologies: List[str]
    query_type: str  # 'api', 'concept', 'tutorial', 'troubleshooting'
    suggested_sources: List[APISource]

@dataclass
class APIResponse:
    """Structured API response data"""
    source: APISource
    title: str
    content: str
    url: str
    code_examples: List[str]
    metadata: Dict[str, Any]
    relevance_score: float

class NaturalLanguageDeveloperAssistant:
    """
    Advanced developer assistant that answers questions using real-time web APIs
    Integrates with iLLuMinator for enhanced content synthesis
    """
    
    def __init__(self, illuminator_model_path: Optional[str] = None):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Nexus-CLI/2.0) Developer Assistant'
        })
        
        # API credentials
        self.github_token = os.getenv('GITHUB_TOKEN')
        self.wikipedia_base = "https://en.wikipedia.org/api/rest_v1"
        self.devdocs_base = "https://devdocs.io"
        
        # Technology patterns for query analysis
        self.tech_patterns = {
            'javascript': ['javascript', 'js', 'node', 'nodejs', 'react', 'vue', 'angular', 'fetch', 'promise', 'async'],
            'python': ['python', 'django', 'flask', 'fastapi', 'requests', 'asyncio', 'pandas', 'numpy'],
            'css': ['css', 'styling', 'flexbox', 'grid', 'animation', 'responsive'],
            'html': ['html', 'dom', 'elements', 'attributes', 'semantic'],
            'web apis': ['fetch', 'xmlhttprequest', 'websocket', 'geolocation', 'notification'],
            'frameworks': ['react', 'vue', 'angular', 'svelte', 'express', 'fastapi', 'django'],
            'tools': ['webpack', 'vite', 'rollup', 'babel', 'typescript', 'eslint']
        }
        
        # Query type patterns
        self.query_type_patterns = {
            'api': ['how does', 'what is', 'api', 'method', 'function', 'parameter'],
            'concept': ['explain', 'concept', 'theory', 'principle', 'pattern'],
            'tutorial': ['tutorial', 'example', 'how to', 'step by step', 'guide'],
            'troubleshooting': ['error', 'bug', 'fix', 'problem', 'not working', 'issue']
        }
        
        # Initialize iLLuMinator model if available
        self.illuminator = None
        if illuminator_model_path and os.path.exists(illuminator_model_path):
            try:
                self._load_illuminator_model(illuminator_model_path)
            except Exception as e:
                logger.warning(f"Could not load iLLuMinator model: {e}")
    
    def _load_illuminator_model(self, model_path: str):
        """Load the iLLuMinator model for content synthesis"""
        try:
            import sys
            sys.path.append('/Users/anishpaleja/Nexus-CLI/model')
            from nexus_model import NexusModel
            
            self.illuminator = NexusModel()
            logger.info("iLLuMinator model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load iLLuMinator: {e}")
    
    async def answer_developer_question(self, question: str) -> str:
        """
        Main entry point: Answer a developer question using real-time web APIs
        """
        try:
            # 1. Analyze and structure the query
            parsed_query = self._parse_developer_query(question)
            logger.info(f"Parsed query: {parsed_query.query_type} about {parsed_query.detected_technologies}")
            
            # 2. Search multiple APIs concurrently
            api_responses = await self._search_multiple_apis(parsed_query)
            
            # 3. Synthesize the response
            if self.illuminator:
                response = await self._synthesize_with_illuminator(parsed_query, api_responses)
            else:
                response = self._synthesize_basic_response(parsed_query, api_responses)
            
            return response
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return f"I encountered an error while researching your question: {e}"
    
    def _parse_developer_query(self, question: str) -> DeveloperQuery:
        """Parse and analyze the developer question to determine search strategy"""
        question_lower = question.lower()
        
        # Detect technologies mentioned
        detected_techs = []
        for tech_category, keywords in self.tech_patterns.items():
            if any(keyword in question_lower for keyword in keywords):
                detected_techs.append(tech_category)
        
        # Determine query type
        query_type = 'concept'  # default
        for qtype, patterns in self.query_type_patterns.items():
            if any(pattern in question_lower for pattern in patterns):
                query_type = qtype
                break
        
        # Suggest best API sources based on detected technologies
        suggested_sources = self._suggest_api_sources(detected_techs, query_type)
        
        # Normalize query for API searches
        normalized_query = self._normalize_query(question)
        
        return DeveloperQuery(
            original_question=question,
            normalized_query=normalized_query,
            detected_technologies=detected_techs,
            query_type=query_type,
            suggested_sources=suggested_sources
        )
    
    def _suggest_api_sources(self, technologies: List[str], query_type: str) -> List[APISource]:
        """Suggest the best API sources based on detected technologies and query type"""
        sources = []
        
        # Technology-specific sources
        if any(tech in ['javascript', 'css', 'html', 'web apis'] for tech in technologies):
            sources.extend([APISource.MDN, APISource.DEVDOCS])
        
        if 'python' in technologies:
            sources.extend([APISource.PYPI, APISource.DEVDOCS])
        
        if any(tech in ['frameworks', 'tools'] for tech in technologies):
            sources.extend([APISource.GITHUB, APISource.NPM])
        
        # Query-type specific sources
        if query_type == 'troubleshooting':
            sources.insert(0, APISource.STACKOVERFLOW)
        elif query_type == 'concept':
            sources.insert(0, APISource.WIKIPEDIA)
        
        # Always include general search
        if APISource.DUCKDUCKGO not in sources:
            sources.append(APISource.DUCKDUCKGO)
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(sources))
    
    def _normalize_query(self, question: str) -> str:
        """Normalize the question for API searches"""
        # Remove question words that don't help with search
        stop_words = ['how', 'what', 'why', 'when', 'where', 'does', 'is', 'can', 'should']
        
        words = question.lower().split()
        filtered_words = [word for word in words if word not in stop_words]
        
        return ' '.join(filtered_words)
    
    async def _search_multiple_apis(self, query: DeveloperQuery) -> List[APIResponse]:
        """Search multiple APIs concurrently for comprehensive information"""
        tasks = []
        
        async with aiohttp.ClientSession() as session:
            for source in query.suggested_sources[:4]:  # Limit to top 4 sources
                if source == APISource.MDN:
                    tasks.append(self._search_mdn(session, query))
                elif source == APISource.DEVDOCS:
                    tasks.append(self._search_devdocs(session, query))
                elif source == APISource.WIKIPEDIA:
                    tasks.append(self._search_wikipedia(session, query))
                elif source == APISource.NPM:
                    tasks.append(self._search_npm(session, query))
                elif source == APISource.PYPI:
                    tasks.append(self._search_pypi(session, query))
                elif source == APISource.STACKOVERFLOW:
                    tasks.append(self._search_stackoverflow(session, query))
                elif source == APISource.DUCKDUCKGO:
                    tasks.append(self._search_duckduckgo(session, query))
            
            # Execute all searches concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and None results
            valid_responses = []
            for result in results:
                if isinstance(result, APIResponse):
                    valid_responses.append(result)
                elif isinstance(result, Exception):
                    logger.warning(f"API search failed: {result}")
            
            return sorted(valid_responses, key=lambda x: x.relevance_score, reverse=True)
    
    async def _search_mdn(self, session: aiohttp.ClientSession, query: DeveloperQuery) -> Optional[APIResponse]:
        """Search MDN Web Docs for JavaScript/Web API documentation"""
        try:
            # Use DuckDuckGo to find relevant MDN pages
            search_query = f"site:developer.mozilla.org {query.normalized_query}"
            
            async with session.get(
                f"https://api.duckduckgo.com/?q={quote(search_query)}&format=json&no_html=1"
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Look for MDN results in related topics
                    for topic in data.get('RelatedTopics', []):
                        if isinstance(topic, dict) and topic.get('FirstURL'):
                            url = topic.get('FirstURL', '')
                            if 'developer.mozilla.org' in url:
                                # Fetch the actual MDN page content
                                content = await self._fetch_page_content(session, url)
                                code_examples = self._extract_code_examples(content, 'javascript')
                                
                                return APIResponse(
                                    source=APISource.MDN,
                                    title=f"MDN: {topic.get('Text', '').split(' - ')[0]}",
                                    content=topic.get('Text', ''),
                                    url=url,
                                    code_examples=code_examples,
                                    metadata={'source': 'MDN Web Docs'},
                                    relevance_score=0.9
                                )
        except Exception as e:
            logger.warning(f"MDN search failed: {e}")
        
        return None
    
    async def _search_devdocs(self, session: aiohttp.ClientSession, query: DeveloperQuery) -> Optional[APIResponse]:
        """Search DevDocs for comprehensive API documentation"""
        try:
            # DevDocs has a search API
            search_url = f"https://devdocs.io/search?q={quote(query.normalized_query)}"
            
            async with session.get(search_url) as response:
                if response.status == 200:
                    # DevDocs returns HTML, we'll parse for relevant links
                    html_content = await response.text()
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    # Look for the first relevant result
                    result_links = soup.find_all('a', href=True)[:3]
                    
                    for link in result_links:
                        href = link.get('href')
                        if href and href.startswith('/'):
                            full_url = f"https://devdocs.io{href}"
                            title = link.text.strip()
                            
                            if title and len(title) > 5:  # Valid title
                                return APIResponse(
                                    source=APISource.DEVDOCS,
                                    title=f"DevDocs: {title}",
                                    content=f"Documentation for {title}",
                                    url=full_url,
                                    code_examples=[],
                                    metadata={'source': 'DevDocs'},
                                    relevance_score=0.8
                                )
        except Exception as e:
            logger.warning(f"DevDocs search failed: {e}")
        
        return None
    
    async def _search_wikipedia(self, session: aiohttp.ClientSession, query: DeveloperQuery) -> Optional[APIResponse]:
        """Search Wikipedia for programming concepts and explanations"""
        try:
            # First, search for relevant articles
            search_url = f"https://en.wikipedia.org/api/rest_v1/page/search/{quote(query.normalized_query)}"
            
            async with session.get(search_url) as response:
                if response.status == 200:
                    data = await response.json()
                    pages = data.get('pages', [])
                    
                    if pages:
                        # Get the first relevant page
                        page = pages[0]
                        title = page.get('title')
                        
                        # Fetch page content
                        content_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title)}"
                        async with session.get(content_url) as content_response:
                            if content_response.status == 200:
                                content_data = await content_response.json()
                                
                                return APIResponse(
                                    source=APISource.WIKIPEDIA,
                                    title=f"Wikipedia: {title}",
                                    content=content_data.get('extract', ''),
                                    url=content_data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                                    code_examples=[],
                                    metadata={'source': 'Wikipedia'},
                                    relevance_score=0.7
                                )
        except Exception as e:
            logger.warning(f"Wikipedia search failed: {e}")
        
        return None
    
    async def _search_npm(self, session: aiohttp.ClientSession, query: DeveloperQuery) -> Optional[APIResponse]:
        """Search NPM registry for JavaScript packages"""
        try:
            if not any(tech in ['javascript', 'frameworks', 'tools'] for tech in query.detected_technologies):
                return None
            
            search_url = f"https://registry.npmjs.org/-/v1/search?text={quote(query.normalized_query)}&size=1"
            
            async with session.get(search_url) as response:
                if response.status == 200:
                    data = await response.json()
                    objects = data.get('objects', [])
                    
                    if objects:
                        obj = objects[0]
                        package = obj.get('package', {})
                        
                        return APIResponse(
                            source=APISource.NPM,
                            title=f"NPM: {package.get('name')}",
                            content=package.get('description', ''),
                            url=f"https://www.npmjs.com/package/{package.get('name')}",
                            code_examples=[],
                            metadata={
                                'version': package.get('version'),
                                'keywords': package.get('keywords', [])
                            },
                            relevance_score=obj.get('score', {}).get('final', 0.5)
                        )
        except Exception as e:
            logger.warning(f"NPM search failed: {e}")
        
        return None
    
    async def _search_pypi(self, session: aiohttp.ClientSession, query: DeveloperQuery) -> Optional[APIResponse]:
        """Search PyPI for Python packages"""
        try:
            if 'python' not in query.detected_technologies:
                return None
            
            # Try exact package name first
            package_url = f"https://pypi.org/pypi/{query.normalized_query.replace(' ', '-')}/json"
            
            async with session.get(package_url) as response:
                if response.status == 200:
                    data = await response.json()
                    info = data.get('info', {})
                    
                    return APIResponse(
                        source=APISource.PYPI,
                        title=f"PyPI: {info.get('name')}",
                        content=info.get('summary', ''),
                        url=info.get('project_url', ''),
                        code_examples=[],
                        metadata={
                            'version': info.get('version'),
                            'author': info.get('author')
                        },
                        relevance_score=0.8
                    )
        except Exception as e:
            logger.warning(f"PyPI search failed: {e}")
        
        return None
    
    async def _search_stackoverflow(self, session: aiohttp.ClientSession, query: DeveloperQuery) -> Optional[APIResponse]:
        """Search Stack Overflow for programming questions and answers"""
        try:
            search_url = f"https://api.stackexchange.com/2.3/search/advanced"
            params = {
                'order': 'desc',
                'sort': 'relevance',
                'q': query.normalized_query,
                'site': 'stackoverflow',
                'pagesize': 1,
                'filter': 'withbody'
            }
            
            async with session.get(search_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    items = data.get('items', [])
                    
                    if items:
                        item = items[0]
                        
                        return APIResponse(
                            source=APISource.STACKOVERFLOW,
                            title=f"Stack Overflow: {item.get('title')}",
                            content=f"Score: {item.get('score')} | Views: {item.get('view_count')}",
                            url=item.get('link', ''),
                            code_examples=[],
                            metadata={
                                'score': item.get('score'),
                                'answer_count': item.get('answer_count'),
                                'tags': item.get('tags', [])
                            },
                            relevance_score=min(item.get('score', 0) / 50, 1.0)
                        )
        except Exception as e:
            logger.warning(f"Stack Overflow search failed: {e}")
        
        return None
    
    async def _search_duckduckgo(self, session: aiohttp.ClientSession, query: DeveloperQuery) -> Optional[APIResponse]:
        """Search DuckDuckGo for general programming information"""
        try:
            search_url = f"https://api.duckduckgo.com/?q={quote(query.normalized_query)}&format=json&no_html=1"
            
            async with session.get(search_url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Check for instant answer first
                    if data.get('Answer'):
                        return APIResponse(
                            source=APISource.DUCKDUCKGO,
                            title="DuckDuckGo Instant Answer",
                            content=data.get('Answer'),
                            url='',
                            code_examples=[],
                            metadata={'type': 'instant_answer'},
                            relevance_score=0.95
                        )
                    
                    # Check abstract
                    if data.get('Abstract'):
                        return APIResponse(
                            source=APISource.DUCKDUCKGO,
                            title=f"DuckDuckGo: {data.get('AbstractSource', 'General Info')}",
                            content=data.get('Abstract'),
                            url=data.get('AbstractURL', ''),
                            code_examples=[],
                            metadata={'source': data.get('AbstractSource')},
                            relevance_score=0.8
                        )
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}")
        
        return None
    
    async def _fetch_page_content(self, session: aiohttp.ClientSession, url: str) -> str:
        """Fetch and return clean text content from a web page"""
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()
                    
                    # Get text content
                    text = soup.get_text()
                    
                    # Clean up whitespace
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    text = ' '.join(chunk for chunk in chunks if chunk)
                    
                    return text[:2000]  # Limit content length
        except Exception as e:
            logger.warning(f"Failed to fetch page content: {e}")
        
        return ""
    
    def _extract_code_examples(self, content: str, language: str = None) -> List[str]:
        """Extract code examples from content"""
        code_examples = []
        
        # Common code block patterns
        patterns = [
            r'```(?:' + (language or r'\w*') + r')?\n(.*?)\n```',
            r'<code>(.*?)</code>',
            r'`([^`]+)`'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            code_examples.extend(matches)
        
        # Clean and filter examples
        cleaned_examples = []
        for example in code_examples:
            example = example.strip()
            if len(example) > 10 and len(example) < 500:  # Reasonable code length
                cleaned_examples.append(example)
        
        return cleaned_examples[:3]  # Limit to 3 examples
    
    async def _synthesize_with_illuminator(self, query: DeveloperQuery, responses: List[APIResponse]) -> str:
        """Use iLLuMinator model to synthesize a comprehensive answer"""
        try:
            # Prepare context for the model
            context_parts = [f"Question: {query.original_question}\n"]
            
            for response in responses[:3]:  # Use top 3 responses
                context_parts.append(f"Source: {response.source.value}")
                context_parts.append(f"Title: {response.title}")
                context_parts.append(f"Content: {response.content}")
                if response.code_examples:
                    context_parts.append(f"Code Examples: {response.code_examples[0]}")
                context_parts.append("---")
            
            context = "\n".join(context_parts)
            
            # Create prompt for iLLuMinator
            prompt = f"""Based on the following information from multiple developer resources, provide a comprehensive answer to the developer's question. Include relevant code examples if available.

{context}

Please synthesize this information into a clear, helpful response for the developer."""
            
            # Generate response using iLLuMinator
            illuminator_response = self.illuminator.generate_response(prompt)
            
            # Add source links
            source_links = [f"• {resp.title}: {resp.url}" for resp in responses if resp.url]
            if source_links:
                illuminator_response += "\n\n**Sources:**\n" + "\n".join(source_links)
            
            return illuminator_response
            
        except Exception as e:
            logger.warning(f"iLLuMinator synthesis failed: {e}")
            return self._synthesize_basic_response(query, responses)
    
    def _synthesize_basic_response(self, query: DeveloperQuery, responses: List[APIResponse]) -> str:
        """Basic response synthesis without iLLuMinator"""
        if not responses:
            return f"I couldn't find specific information about '{query.original_question}'. You might want to try rephrasing your question or checking the official documentation directly."
        
        response_parts = [f"## Answer: {query.original_question}\n"]
        
        # Add best response
        best_response = responses[0]
        response_parts.append(f"**{best_response.title}**")
        response_parts.append(best_response.content)
        
        # Add code examples if available
        if best_response.code_examples:
            response_parts.append("\n**Code Example:**")
            response_parts.append(f"```\n{best_response.code_examples[0]}\n```")
        
        # Add additional sources
        if len(responses) > 1:
            response_parts.append("\n**Additional Resources:**")
            for resp in responses[1:3]:
                if resp.url:
                    response_parts.append(f"• [{resp.title}]({resp.url})")
        
        # Add main source link
        if best_response.url:
            response_parts.append(f"\n**Learn More:** {best_response.url}")
        
        return "\n".join(response_parts)


def main():
    """Command-line interface for the developer assistant"""
    parser = argparse.ArgumentParser(description="Natural Language Developer Assistant")
    parser.add_argument("question", nargs="*", help="Your developer question")
    parser.add_argument("--model-path", help="Path to iLLuMinator model")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    # Initialize assistant
    assistant = NaturalLanguageDeveloperAssistant(args.model_path)
    
    async def process_question(question: str) -> str:
        """Process a single question"""
        print(f"🔍 Researching: {question}")
        print("⏳ Searching developer resources...")
        
        start_time = time.time()
        answer = await assistant.answer_developer_question(question)
        search_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(answer)
        print(f"\n⏱️  Research completed in {search_time:.2f} seconds")
        print('='*60)
        
        return answer
    
    async def interactive_mode():
        """Interactive question-answering mode"""
        print("🚀 Nexus CLI - Developer Assistant (Interactive Mode)")
        print("Ask any developer question. Type 'exit' to quit.\n")
        
        while True:
            try:
                question = input("💬 Your question: ").strip()
                
                if question.lower() in ['exit', 'quit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not question:
                    continue
                
                await process_question(question)
                print()
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    async def single_question_mode():
        """Process a single question from command line"""
        question = " ".join(args.question)
        if not question:
            print("Please provide a question or use --interactive mode")
            return
        
        await process_question(question)
    
    # Run the appropriate mode
    if args.interactive:
        asyncio.run(interactive_mode())
    else:
        asyncio.run(single_question_mode())


if __name__ == "__main__":
    main()
