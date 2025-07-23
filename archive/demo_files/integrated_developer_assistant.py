#!/usr/bin/env python3
"""
Nexus CLI - Integrated Developer Assistant
Natural language developer question answering integrated with existing iLLuMinator system
"""

import requests
import json
import os
import time
import sys
from typing import Dict, List, Optional, Any, Union
from urllib.parse import quote, urljoin
import logging
from dataclasses import dataclass
from enum import Enum
import re
import argparse

# Add model path for iLLuMinator
sys.path.append('/Users/anishpaleja/Nexus-CLI/model')

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

class IntegratedDeveloperAssistant:
    """
    Integrated developer assistant that answers questions using real-time web APIs
    Works with existing Nexus CLI and iLLuMinator model
    """
    
    def __init__(self, use_illuminator: bool = True):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Nexus-CLI/2.0) Developer Assistant'
        })
        
        # API credentials
        self.github_token = os.getenv('GITHUB_TOKEN')
        self.wikipedia_base = "https://en.wikipedia.org/api/rest_v1"
        
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
        if use_illuminator:
            try:
                self._load_illuminator_model()
            except Exception as e:
                logger.warning(f"Could not load iLLuMinator model: {e}")
    
    def _load_illuminator_model(self):
        """Load the iLLuMinator model for content synthesis"""
        try:
            # Try to import the existing model
            from model import IlluminatorAPI
            self.illuminator = IlluminatorAPI()
            logger.info("iLLuMinator model loaded successfully")
        except ImportError:
            try:
                # Try alternative import
                from nexus_model import NexusModel
                self.illuminator = NexusModel()
                logger.info("Nexus model loaded successfully")
            except ImportError:
                logger.warning("Could not load iLLuMinator model - using basic synthesis")
    
    def answer_developer_question(self, question: str) -> str:
        """
        Main entry point: Answer a developer question using real-time web APIs
        """
        try:
            # 1. Analyze and structure the query
            parsed_query = self._parse_developer_query(question)
            logger.info(f"Parsed query: {parsed_query.query_type} about {parsed_query.detected_technologies}")
            
            # 2. Search multiple APIs
            api_responses = self._search_multiple_apis(parsed_query)
            
            # 3. Synthesize the response
            if self.illuminator:
                response = self._synthesize_with_illuminator(parsed_query, api_responses)
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
    
    def _search_multiple_apis(self, query: DeveloperQuery) -> List[APIResponse]:
        """Search multiple APIs for comprehensive information"""
        responses = []
        
        for source in query.suggested_sources[:4]:  # Limit to top 4 sources
            try:
                if source == APISource.MDN:
                    result = self._search_mdn(query)
                elif source == APISource.WIKIPEDIA:
                    result = self._search_wikipedia(query)
                elif source == APISource.NPM:
                    result = self._search_npm(query)
                elif source == APISource.PYPI:
                    result = self._search_pypi(query)
                elif source == APISource.STACKOVERFLOW:
                    result = self._search_stackoverflow(query)
                elif source == APISource.DUCKDUCKGO:
                    result = self._search_duckduckgo(query)
                else:
                    continue
                
                if result:
                    responses.append(result)
                    
            except Exception as e:
                logger.warning(f"Search failed for {source}: {e}")
        
        return sorted(responses, key=lambda x: x.relevance_score, reverse=True)
    
    def _search_mdn(self, query: DeveloperQuery) -> Optional[APIResponse]:
        """Search MDN Web Docs using DuckDuckGo site search"""
        try:
            search_query = f"site:developer.mozilla.org {query.normalized_query}"
            response = self.session.get(
                f"https://api.duckduckgo.com/?q={quote(search_query)}&format=json&no_html=1",
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Look for MDN results in related topics
                for topic in data.get('RelatedTopics', []):
                    if isinstance(topic, dict) and topic.get('FirstURL'):
                        url = topic.get('FirstURL', '')
                        if 'developer.mozilla.org' in url:
                            title_text = topic.get('Text', '')
                            
                            return APIResponse(
                                source=APISource.MDN,
                                title=f"MDN: {title_text.split(' - ')[0] if ' - ' in title_text else title_text[:50]}",
                                content=title_text,
                                url=url,
                                code_examples=self._extract_basic_code_examples(title_text),
                                metadata={'source': 'MDN Web Docs'},
                                relevance_score=0.9
                            )
        except Exception as e:
            logger.warning(f"MDN search failed: {e}")
        
        return None
    
    def _search_wikipedia(self, query: DeveloperQuery) -> Optional[APIResponse]:
        """Search Wikipedia for programming concepts"""
        try:
            # Search for relevant articles
            search_url = f"https://en.wikipedia.org/api/rest_v1/page/search/{quote(query.normalized_query)}"
            response = self.session.get(search_url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                pages = data.get('pages', [])
                
                if pages:
                    page = pages[0]
                    title = page.get('title')
                    
                    # Get page summary
                    summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title)}"
                    summary_response = self.session.get(summary_url, timeout=5)
                    
                    if summary_response.status_code == 200:
                        summary_data = summary_response.json()
                        
                        return APIResponse(
                            source=APISource.WIKIPEDIA,
                            title=f"Wikipedia: {title}",
                            content=summary_data.get('extract', '')[:500],  # Limit content
                            url=summary_data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                            code_examples=[],
                            metadata={'source': 'Wikipedia'},
                            relevance_score=0.7
                        )
        except Exception as e:
            logger.warning(f"Wikipedia search failed: {e}")
        
        return None
    
    def _search_npm(self, query: DeveloperQuery) -> Optional[APIResponse]:
        """Search NPM registry for JavaScript packages"""
        try:
            if not any(tech in ['javascript', 'frameworks', 'tools'] for tech in query.detected_technologies):
                return None
            
            search_url = f"https://registry.npmjs.org/-/v1/search?text={quote(query.normalized_query)}&size=1"
            response = self.session.get(search_url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
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
    
    def _search_pypi(self, query: DeveloperQuery) -> Optional[APIResponse]:
        """Search PyPI for Python packages"""
        try:
            if 'python' not in query.detected_technologies:
                return None
            
            # Try exact package name
            package_name = query.normalized_query.replace(' ', '-')
            package_url = f"https://pypi.org/pypi/{package_name}/json"
            response = self.session.get(package_url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                info = data.get('info', {})
                
                return APIResponse(
                    source=APISource.PYPI,
                    title=f"PyPI: {info.get('name')}",
                    content=info.get('summary', ''),
                    url=info.get('project_url', f"https://pypi.org/project/{package_name}"),
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
    
    def _search_stackoverflow(self, query: DeveloperQuery) -> Optional[APIResponse]:
        """Search Stack Overflow for programming questions"""
        try:
            search_url = "https://api.stackexchange.com/2.3/search/advanced"
            params = {
                'order': 'desc',
                'sort': 'relevance',
                'q': query.normalized_query,
                'site': 'stackoverflow',
                'pagesize': 1,
                'filter': 'default'
            }
            
            response = self.session.get(search_url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                items = data.get('items', [])
                
                if items:
                    item = items[0]
                    
                    return APIResponse(
                        source=APISource.STACKOVERFLOW,
                        title=f"Stack Overflow: {item.get('title', '')[:50]}...",
                        content=f"Score: {item.get('score', 0)} | Views: {item.get('view_count', 0)} | Answers: {item.get('answer_count', 0)}",
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
    
    def _search_duckduckgo(self, query: DeveloperQuery) -> Optional[APIResponse]:
        """Search DuckDuckGo for general programming information"""
        try:
            search_url = f"https://api.duckduckgo.com/?q={quote(query.normalized_query + ' programming')}&format=json&no_html=1"
            response = self.session.get(search_url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check for instant answer
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
                        title=f"Overview: {data.get('AbstractSource', 'Programming Info')}",
                        content=data.get('Abstract'),
                        url=data.get('AbstractURL', ''),
                        code_examples=[],
                        metadata={'source': data.get('AbstractSource')},
                        relevance_score=0.8
                    )
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}")
        
        return None
    
    def _extract_basic_code_examples(self, content: str) -> List[str]:
        """Extract basic code examples from content"""
        code_examples = []
        
        # Look for common code patterns
        patterns = [
            r'`([^`]+)`',  # Inline code
            r'(\w+\([^)]*\))',  # Function calls
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if len(match) > 5 and len(match) < 100:
                    code_examples.append(match)
        
        return code_examples[:2]  # Limit to 2 examples
    
    def _synthesize_with_illuminator(self, query: DeveloperQuery, responses: List[APIResponse]) -> str:
        """Use iLLuMinator model to synthesize a comprehensive answer"""
        try:
            # Prepare context for the model
            context_parts = [f"Developer Question: {query.original_question}\n"]
            context_parts.append(f"Query Type: {query.query_type}")
            context_parts.append(f"Technologies: {', '.join(query.detected_technologies)}\n")
            
            for i, response in enumerate(responses[:3], 1):  # Use top 3 responses
                context_parts.append(f"Source {i}: {response.source.value}")
                context_parts.append(f"Title: {response.title}")
                context_parts.append(f"Content: {response.content}")
                if response.code_examples:
                    context_parts.append(f"Code: {response.code_examples[0]}")
                context_parts.append("---")
            
            context = "\n".join(context_parts)
            
            # Create prompt for iLLuMinator
            prompt = f"""You are a helpful developer assistant. Based on the following research from multiple sources, provide a clear and comprehensive answer to the developer's question. 

{context}

Please synthesize this information into a practical, helpful response. Include code examples when relevant."""
            
            # Generate response using iLLuMinator
            if hasattr(self.illuminator, 'generate_response'):
                illuminator_response = self.illuminator.generate_response(prompt)
            elif hasattr(self.illuminator, 'query'):
                illuminator_response = self.illuminator.query(prompt)
            else:
                # Fallback to basic synthesis
                return self._synthesize_basic_response(query, responses)
            
            # Add source links
            source_links = []
            for resp in responses:
                if resp.url:
                    source_links.append(f"• {resp.title}: {resp.url}")
            
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
        
        response_parts = [f"## {query.original_question}\n"]
        
        # Add best response
        best_response = responses[0]
        response_parts.append(f"**{best_response.title}**")
        response_parts.append(best_response.content)
        
        # Add code examples if available
        if best_response.code_examples:
            response_parts.append("\n**Code Example:**")
            response_parts.append(f"`{best_response.code_examples[0]}`")
        
        # Add additional sources
        if len(responses) > 1:
            response_parts.append("\n**Additional Information:**")
            for resp in responses[1:3]:
                response_parts.append(f"• **{resp.title}**: {resp.content[:100]}...")
        
        # Add source links
        source_links = []
        for resp in responses:
            if resp.url:
                source_links.append(f"• {resp.title}: {resp.url}")
        
        if source_links:
            response_parts.append("\n**Learn More:**")
            response_parts.extend(source_links)
        
        return "\n".join(response_parts)


# Integration with existing Nexus CLI
def integrate_with_nexus_cli():
    """Function to integrate with the main Nexus CLI system"""
    assistant = IntegratedDeveloperAssistant()
    
    def ask_developer_question(question: str) -> str:
        """Main function for Nexus CLI integration"""
        return assistant.answer_developer_question(question)
    
    return ask_developer_question


def main():
    """Command-line interface for the developer assistant"""
    parser = argparse.ArgumentParser(description="Integrated Developer Assistant")
    parser.add_argument("question", nargs="*", help="Your developer question")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--no-model", action="store_true", help="Disable iLLuMinator model")
    
    args = parser.parse_args()
    
    # Initialize assistant
    assistant = IntegratedDeveloperAssistant(use_illuminator=not args.no_model)
    
    def process_question(question: str) -> str:
        """Process a single question"""
        print(f"🔍 Researching: {question}")
        print("⏳ Searching developer resources...")
        
        start_time = time.time()
        answer = assistant.answer_developer_question(question)
        search_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(answer)
        print(f"\n⏱️  Research completed in {search_time:.2f} seconds")
        print('='*60)
        
        return answer
    
    def interactive_mode():
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
                
                process_question(question)
                print()
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def single_question_mode():
        """Process a single question from command line"""
        question = " ".join(args.question)
        if not question:
            print("Please provide a question or use --interactive mode")
            return
        
        process_question(question)
    
    # Run the appropriate mode
    if args.interactive:
        interactive_mode()
    else:
        single_question_mode()


if __name__ == "__main__":
    main()
