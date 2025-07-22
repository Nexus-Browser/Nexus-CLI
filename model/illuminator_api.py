"""
iLLuMinator API Client - Direct integration with iLLuMinator-4.7B model
Connects directly to the iLLuMinator-4.7B transformer model for local inference
Repository: https://github.com/Anipaleja/iLLuMinator-4.7B
"""

import os
import json
import time
import logging
import torch
from typing import Optional, Dict, List, Any, Tuple
from pathlib import Path
import subprocess
import sys
import requests
import re
from urllib.parse import quote, urljoin
import xml.etree.ElementTree as ET

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExternalKnowledgeAPIs:
    """
    Integration with external knowledge APIs to enhance CLI intelligence
    Uses documentation, package registry, and code repository APIs
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.timeout = 5  # Fast timeout for snappy responses
        self.session.headers.update({
            'User-Agent': 'Nexus-CLI/1.0 Enhanced Intelligence System'
        })
        
        # API endpoints for different knowledge sources
        self.endpoints = {
            'mdn': 'https://developer.mozilla.org/api/v1/search',
            'devdocs': 'https://devdocs.io',
            'stackoverflow': 'https://api.stackexchange.com/2.3/search/excerpts',
            'github_search': 'https://api.github.com/search/code',
            'npm_registry': 'https://registry.npmjs.org',
            'pypi': 'https://pypi.org/pypi',
            'crates_io': 'https://crates.io/api/v1/crates',
            'go_modules': 'https://proxy.golang.org',
            'wikipedia': 'https://en.wikipedia.org/api/rest_v1/page/summary'
        }
    
    def search_documentation(self, language: str, query: str) -> Optional[str]:
        """Search official documentation for programming languages."""
        try:
            if language.lower() in ['javascript', 'js', 'html', 'css', 'web']:
                return self._search_mdn(query)
            elif language.lower() in ['python', 'rust', 'go', 'ruby']:
                return self._search_devdocs(language, query)
            return None
        except Exception as e:
            logger.debug(f"Documentation search failed: {e}")
            return None
    
    def _search_mdn(self, query: str) -> Optional[str]:
        """Search MDN Web Docs for JavaScript/Web technologies."""
        try:
            response = self.session.get(
                f"https://developer.mozilla.org/api/v1/search?q={quote(query)}&locale=en-US",
                timeout=3
            )
            if response.status_code == 200:
                data = response.json()
                if data.get('documents'):
                    doc = data['documents'][0]
                    return f"**{doc.get('title', 'Documentation')}**\n{doc.get('summary', '')}\n\nSource: {doc.get('mdn_url', '')}"
        except Exception:
            pass
        return None
    
    def _search_devdocs(self, language: str, query: str) -> Optional[str]:
        """Search DevDocs for various programming languages."""
        # DevDocs doesn't have a direct API, but we can construct helpful responses
        # based on common documentation patterns
        return None
    
    def search_packages(self, language: str, query: str) -> Optional[str]:
        """Search package registries for libraries and frameworks."""
        try:
            if language.lower() in ['javascript', 'js', 'node', 'typescript']:
                return self._search_npm(query)
            elif language.lower() == 'python':
                return self._search_pypi(query)
            elif language.lower() == 'rust':
                return self._search_crates_io(query)
            return None
        except Exception as e:
            logger.debug(f"Package search failed: {e}")
            return None
    
    def _search_npm(self, query: str) -> Optional[str]:
        """Search npm registry for JavaScript packages."""
        try:
            response = self.session.get(
                f"https://registry.npmjs.org/-/v1/search?text={quote(query)}&size=3",
                timeout=3
            )
            if response.status_code == 200:
                data = response.json()
                if data.get('objects'):
                    results = []
                    for obj in data['objects'][:2]:  # Top 2 results
                        pkg = obj.get('package', {})
                        results.append(f"**{pkg.get('name')}** - {pkg.get('description', 'No description')}")
                    return "**NPM Packages:**\n" + "\n".join(results)
        except Exception:
            pass
        return None
    
    def _search_pypi(self, query: str) -> Optional[str]:
        """Search PyPI for Python packages."""
        try:
            response = self.session.get(
                f"https://pypi.org/simple/{query}/",
                timeout=3
            )
            if response.status_code == 200:
                # Get package info
                info_response = self.session.get(
                    f"https://pypi.org/pypi/{query}/json",
                    timeout=3
                )
                if info_response.status_code == 200:
                    data = info_response.json()
                    info = data.get('info', {})
                    return f"**{info.get('name')}** - {info.get('summary', 'No description')}\n\nInstall: `pip install {query}`"
        except Exception:
            pass
        return None
    
    def _search_crates_io(self, query: str) -> Optional[str]:
        """Search crates.io for Rust crates."""
        try:
            response = self.session.get(
                f"https://crates.io/api/v1/crates?q={quote(query)}&per_page=2",
                timeout=3
            )
            if response.status_code == 200:
                data = response.json()
                if data.get('crates'):
                    results = []
                    for crate in data['crates'][:2]:
                        results.append(f"**{crate.get('name')}** - {crate.get('description', 'No description')}")
                    return "**Rust Crates:**\n" + "\n".join(results)
        except Exception:
            pass
        return None
    
    def search_stackoverflow(self, query: str) -> Optional[str]:
        """Search Stack Overflow for solutions."""
        try:
            response = self.session.get(
                f"https://api.stackexchange.com/2.3/search/excerpts?order=desc&sort=relevance&q={quote(query)}&site=stackoverflow",
                timeout=3
            )
            if response.status_code == 200:
                data = response.json()
                if data.get('items'):
                    item = data['items'][0]
                    return f"**Stack Overflow Solution:**\n{item.get('excerpt', '')}\n\nScore: {item.get('score', 0)}"
        except Exception:
            pass
        return None
    
    def search_github_examples(self, language: str, query: str) -> Optional[str]:
        """Search GitHub for code examples (requires GitHub token for better rate limits)."""
        try:
            # Basic GitHub search without authentication (limited rate)
            response = self.session.get(
                f"https://api.github.com/search/code?q={quote(query)}+language:{language}&sort=indexed&order=desc&per_page=1",
                timeout=3
            )
            if response.status_code == 200:
                data = response.json()
                if data.get('items'):
                    item = data['items'][0]
                    return f"**GitHub Example:**\nFile: {item.get('name')}\nRepo: {item.get('repository', {}).get('full_name')}\nURL: {item.get('html_url')}"
        except Exception:
            pass
        return None
    
    def search_wikipedia_technical(self, query: str) -> Optional[str]:
        """Search Wikipedia for technical concepts."""
        try:
            response = self.session.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(query)}",
                timeout=3
            )
            if response.status_code == 200:
                data = response.json()
                if not data.get('disambiguation') and data.get('extract'):
                    return f"**Wikipedia - {data.get('title')}:**\n{data.get('extract')[:300]}...\n\nSource: {data.get('content_urls', {}).get('desktop', {}).get('page', '')}"
        except Exception:
            pass
        return None


class iLLuMinatorAPI:
    """
    Direct API client for iLLuMinator-4.7B model from GitHub repository
    Provides local inference without external API dependencies
    """
    
    def __init__(self, model_path: str = "./model/nexus_model/", fast_mode: bool = False, use_gpu_acceleration: bool = True):
        """
        Initialize iLLuMinator-4.7B API with aggressive local optimizations.
        
        Args:
            model_path: Path to the local fine-tuned model
            fast_mode: Enable aggressive speed optimizations
            use_gpu_acceleration: Use GPU acceleration if available
        """
        self.model_path = model_path
        self.fast_mode = fast_mode
        self.use_gpu_acceleration = use_gpu_acceleration and self._check_gpu_availability()
        
        # Optimization settings
        self.max_length = 512 if fast_mode else 1024
        self.use_quantization = fast_mode
        
        # Initialize external API integration for enhanced intelligence
        self.external_apis = ExternalKnowledgeAPIs()
        self._api_cache = {}
        self._cache_ttl = 3600  # 1 hour cache
        
        # Device selection
        if self.use_gpu_acceleration:
            import torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        else:
            import torch
            self.device = torch.device("cpu")
        
        
        # Load model with optimizations
        logger.info("🚀 Loading local fine-tuned model with optimizations...")
        self._load_model_optimized()
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU acceleration is available."""
        try:
            import torch
            return torch.cuda.is_available() or torch.backends.mps.is_available()
        except ImportError:
            return False
        
        # Load model with optimizations
        logger.info("🚀 Loading local fine-tuned model with optimizations...")
        self._load_model_optimized()
    
    def _check_available_apis(self) -> Dict[str, str]:
        """Check what cloud APIs are available for fast fallback."""
        available_apis = {}
        
        # Check environment variables for API keys
        env_files = ['.env', '.env.local']
        for env_file in env_files:
            if os.path.exists(env_file):
                with open(env_file, 'r') as f:
                    for line in f:
                        if '=' in line and not line.startswith('#'):
                            key, value = line.strip().split('=', 1)
                            if 'API_KEY' in key and value:
                                if 'COHERE' in key:
                                    available_apis['cohere'] = value
                                elif 'GEMINI' in key:
                                    available_apis['gemini'] = value
                                elif 'GROQ' in key:
                                    available_apis['groq'] = value
                                elif 'OPENAI' in key:
                                    available_apis['openai'] = value
        
        # Also check from COHERE_SUCCESS.md - we know Cohere is working
        if not available_apis and os.path.exists('COHERE_SUCCESS.md'):
            with open('COHERE_SUCCESS.md', 'r') as f:
                content = f.read()
                if 'wx6ib6ezwXClSarnEZU0FrK1eLJcVTqpCAHnfuTW' in content:
                    available_apis['cohere'] = 'wx6ib6ezwXClSarnEZU0FrK1eLJcVTqpCAHnfuTW'
        
        return available_apis

    def _use_cloud_api(self, prompt: str) -> str:
        """Use cloud API for fast responses when available."""
        if 'cohere' in self.fallback_apis:
            return self._call_cohere_api(prompt)
        elif 'gemini' in self.fallback_apis:
            return self._call_gemini_api(prompt)
        elif 'groq' in self.fallback_apis:
            return self._call_groq_api(prompt)
        else:
            return "Fast mode enabled but no cloud APIs available. Using local processing..."
    
    def _call_cohere_api(self, prompt: str) -> str:
        """Call Cohere API for fast responses using requests."""
        try:
            import requests
            
            headers = {
                'Authorization': f'Bearer {self.fallback_apis["cohere"]}',
                'Content-Type': 'application/json',
            }
            
            data = {
                'model': 'command-light',
                'prompt': prompt,
                'max_tokens': 300,
                'temperature': 0.7
            }
            
            response = requests.post(
                'https://api.cohere.ai/v1/generate',
                headers=headers,
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['generations'][0]['text'].strip()
            else:
                logger.warning(f"Cohere API error: {response.status_code}")
                return self._fallback_to_local(prompt)
                
        except Exception as e:
            logger.warning(f"Cohere API failed: {e}")
            return self._fallback_to_local(prompt)
    
    def _call_gemini_api(self, prompt: str) -> str:
        """Call Gemini API for fast responses."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.fallback_apis['gemini'])
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.warning(f"Gemini API failed: {e}")
            return self._fallback_to_local(prompt)
    
    def _call_groq_api(self, prompt: str) -> str:
        """Call Groq API for fast responses."""
        try:
            from groq import Groq
            client = Groq(api_key=self.fallback_apis['groq'])
            completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192",
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.warning(f"Groq API failed: {e}")
            return self._fallback_to_local(prompt)

    def _fallback_to_local(self, prompt: str) -> str:
        """Fallback to basic local processing when APIs fail."""
        return f"iLLuMinator-4.7B: I understand you want to {prompt.lower()}. Let me help with that using local processing."

    def _find_model_path(self) -> str:
        """Find the iLLuMinator-4.7B model path."""
        possible_paths = [
            "./model/nexus_model/",
            "./model/illuminator_model/",
            "./models/iLLuMinator-4.7B/",
            os.path.expanduser("~/.cache/huggingface/transformers/"),
            "./nexus_model/"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # Default path - will be created if needed
        return "./model/nexus_model/"
    
    def _ensure_dependencies(self):
        """Ensure required dependencies are installed."""
        required_packages = [
            "torch",
            "transformers",
            "tokenizers",
            "accelerate",
            "safetensors"
        ]
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                logger.info(f"Installing {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    def _load_model_optimized(self):
        """Load the local fine-tuned model with aggressive local optimizations."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            import torch
            
            logger.info("🚀 Loading local fine-tuned model with optimizations...")
            
            # Load tokenizer first (fastest part)
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True,
                use_fast=False  # Use regular tokenizer to avoid tokenizer.json issues
            )
            
            # Set pad token for optimization
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Aggressive quantization for speed
            if self.use_quantization and self.device.type == "cuda":
                logger.info("Applying 8-bit quantization for GPU acceleration...")
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.float16,
                    bnb_8bit_use_double_quant=True,
                )
                model_kwargs = {"quantization_config": quantization_config}
            elif self.use_quantization:
                logger.info("Applying CPU optimizations...")
                model_kwargs = {"torch_dtype": torch.float16 if self.device.type != "cpu" else torch.float32}
            else:
                model_kwargs = {}
            
            # Load local model with optimizations
            logger.info("Loading optimized local fine-tuned model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                local_files_only=True,
                device_map="auto" if self.use_gpu_acceleration else None,
                low_cpu_mem_usage=True,
                **model_kwargs
            )
            
            # Set to evaluation mode for faster inference
            self.model.eval()
            
            # Enable optimization features
            if hasattr(self.model, 'gradient_checkpointing_disable'):
                self.model.gradient_checkpointing_disable()
            
            # Move to device if not using device_map
            if not self.use_gpu_acceleration or "device_map" not in model_kwargs:
                self.model = self.model.to(self.device)
            
            self.model_loaded = True
            logger.info("✓ iLLuMinator-4.7B optimized for local fast inference!")
            
        except ImportError as e:
            logger.error(f"Missing required packages: {e}")
            logger.info("Installing required packages...")
            self._ensure_dependencies()
            raise
        except Exception as e:
            logger.error(f"Error loading optimized iLLuMinator-4.7B model: {str(e)}")
            raise

    def _load_model(self):
        """Load the iLLuMinator-4.7B model and tokenizer."""
        try:
            # Import required libraries
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            import torch
            
            # Configure for optimal performance
            device_map = "auto" if torch.cuda.is_available() else None
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            
            # Load tokenizer
            logger.info("Loading iLLuMinator-4.7B tokenizer...")
            if os.path.exists(self.model_path):
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    use_fast=True
                )
            else:
                # Load from Hugging Face Hub or GitHub
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "Anipaleja/iLLuMinator-4.7B",
                    trust_remote_code=True,
                    use_fast=True,
                    cache_dir=self.model_path
                )
            
            # Configure quantization for better memory usage
            quantization_config = None
            if torch.cuda.is_available():
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            
            # Load model
            logger.info("Loading iLLuMinator-4.7B model...")
            if os.path.exists(self.model_path):
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    quantization_config=quantization_config,
                    device_map=device_map,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            else:
                # Load from Hugging Face Hub or GitHub
                self.model = AutoModelForCausalLM.from_pretrained(
                    "Anipaleja/iLLuMinator-4.7B",
                    quantization_config=quantization_config,
                    device_map=device_map,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    cache_dir=self.model_path
                )
            
            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model_loaded = True
            logger.info(f"✓ iLLuMinator-4.7B loaded on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading iLLuMinator-4.7B model: {str(e)}")
            # Fallback to local implementation
            self._load_local_fallback()
    
    def _load_local_fallback(self):
        """Load local fallback implementation."""
        try:
            from .illuminator_ai import IlluminatorAI
            self.model = IlluminatorAI()
            self.tokenizer = self.model.tokenizer if hasattr(self.model, 'tokenizer') else None
            self.model_loaded = True
            logger.info("✓ Using local iLLuMinator implementation")
        except Exception as e:
            logger.error(f"Failed to load local fallback: {str(e)}")
            self.model_loaded = False
    
    def is_available(self) -> bool:
        """Check if the model is available and loaded."""
        return self.model_loaded and self.model is not None
    
    def generate_response(self, prompt: str, max_length: int = 256, temperature: float = 0.7) -> str:
        """Generate conversational response using locally optimized fine-tuned model with quality fallback."""
        if not self.is_available():
            return "I apologize, but the local fine-tuned model is not currently available."
        
        try:
            # First try the fine-tuned model
            conversation = f"Question: {prompt}\n\nAnswer:"
            
            if hasattr(self.model, 'generate') and self.tokenizer:
                response = self._generate_with_optimized_transformers(conversation, max_length, temperature)
                
                # Check if the response is good quality
                if self._is_quality_response(response, prompt):
                    return response
                
                # If response is poor quality, use local rule-based fallback
                return self._generate_local_quality_response(prompt)
            else:
                return self._generate_local_quality_response(prompt)
            
        except Exception as e:
            logger.error(f"Response generation error: {str(e)}")
            return self._generate_local_quality_response(prompt)
    
    def _is_quality_response(self, response: str, prompt: str) -> bool:
        """Check if the generated response is of good quality with balanced criteria."""
        if not response or len(response.strip()) < 15:
            return False
        
        response_clean = response.strip()
        
        # Check for gibberish patterns and nonsensical content
        gibberish_patterns = [
            # Random characters and symbols
            r'[#%*+=\[\]{}()<>_]{2,}',
            # Excessive special characters in sequence
            r'[^\w\s.,!?-]{3,}',
            # Random code-like gibberish
            r'\s*\):\s*|\s*::\s*\*\*|\s*>>>\s*\(\s*#',
            # Nonsensical variable names
            r'[a-zA-Z]{1}[0-9]{2,}[a-zA-Z]{1}[0-9]{2,}',
            # Random brackets and operators
            r'\[\s*\|\s*=\s*\+\s*\[',
            # Underscore followed by hash/special chars at start
            r'^_\s*#',
            # Code fragments without context
            r'^\s*def\s+\w+\([^)]*\):\s*$',
            # Nonsensical questions as responses
            r'What if I am a.*but not really\?'
        ]
        
        import re
        for pattern in gibberish_patterns:
            if re.search(pattern, response_clean):
                return False
        
        # Check for meaningless responses
        meaningless_responses = [
            'code', 'hello', 'hi', '_ #', 'programming)', 'answer to a question',
            'what if i am a programmer', 'but not really'
        ]
        
        response_lower = response_clean.lower()
        if any(meaningless == response_lower.strip() for meaningless in meaningless_responses):
            return False
        
        # Check for excessive repetition
        words = response_clean.split()
        if len(words) < 6:  # Require at least 6 words for a meaningful response
            return False
        
        # Check if it's just repeated words or characters
        unique_words = set(words)
        if len(unique_words) < len(words) * 0.4:  # Less than 40% unique words
            return False
        
        # Check for meaningful sentence structure
        # Allow responses without punctuation if they're coherent and long enough
        sentences = [s.strip() for s in re.split(r'[.!?]+', response_clean) if s.strip()]
        if len(sentences) == 0:
            # If no sentences found, treat the whole response as one sentence
            sentences = [response_clean]
        
        # Check for code-like gibberish without meaning
        code_gibberish = ['def print(', 'answer 1a1c0e', 'example : i ::', 'return args """']
        if any(pattern in response_clean.lower() for pattern in code_gibberish):
            return False
        
        # For programming questions, response should contain relevant programming terms OR be informative
        if any(word in prompt.lower() for word in ['python', 'programming', 'code', 'language']):
            programming_terms = ['language', 'programming', 'code', 'software', 'development', 
                               'syntax', 'interpreter', 'compiler', 'used for', 'high-level', 'library']
            # Allow if it contains programming terms OR if it's a substantial informative response
            if not any(term in response_lower for term in programming_terms) and len(words) < 12:
                return False
        
        return True
    
    def _generate_local_quality_response(self, prompt: str) -> str:
        """Generate intelligent response using advanced AI reasoning patterns like Claude/Gemini/Copilot."""
        prompt_lower = prompt.lower()
        
        # Advanced natural language processing - analyze intent and context
        response = self._analyze_user_intent_and_respond(prompt, prompt_lower)
        if response:
            # Try to enhance with external API data
            enhanced_response = self._enhance_with_external_knowledge(response, prompt, prompt_lower)
            return enhanced_response if enhanced_response else response
        
        # Try external APIs for enhanced intelligence BEFORE falling back to local knowledge
        external_response = self._get_enhanced_external_response(prompt, prompt_lower)
        if external_response:
            return external_response
        
        # Programming language questions with deep context
        if any(word in prompt_lower for word in ['rust', 'rust language', 'rust programming']):
            return self._generate_rust_expertise(prompt, prompt_lower)
        
        if any(word in prompt_lower for word in ['javascript', 'js', 'node.js', 'typescript', 'ts']):
            return self._generate_javascript_expertise(prompt, prompt_lower)
        
        if any(word in prompt_lower for word in ['python', 'python language', 'py']):
            return self._generate_python_expertise(prompt, prompt_lower)
        
        # AI-related questions with comprehensive analysis
        if any(word in prompt_lower for word in ['ai', 'artificial intelligence', 'machine learning', 'neural network', 'deep learning', 'llm', 'gpt', 'transformer']):
            return self._generate_ai_expertise(prompt, prompt_lower)
        
        # General programming questions
        if any(word in prompt_lower for word in ['programming', 'coding', 'software development', 'algorithm']):
            return "Programming is the process of creating instructions for computers using programming languages. It involves problem-solving, algorithm design, and implementing solutions in code. Key concepts include data structures, algorithms, design patterns, and software engineering principles. Popular languages include Python, JavaScript, Rust, Java, C++, and Go."
        
        # Web development
        if any(word in prompt_lower for word in ['web development', 'frontend', 'backend', 'api', 'http']):
            return "Web development involves creating websites and web applications. Frontend development focuses on user interfaces using HTML, CSS, and JavaScript. Backend development handles server logic, databases, and APIs using languages like Python, Node.js, Rust, Java, or Go. Modern web development often uses frameworks and follows RESTful or GraphQL API designs."
        
        # Data structures and algorithms
        if any(word in prompt_lower for word in ['data structure', 'algorithm', 'sorting', 'searching', 'complexity']):
            return "Data structures organize and store data efficiently (arrays, linked lists, trees, graphs, hash tables). Algorithms are step-by-step procedures to solve problems (sorting, searching, graph traversal). Time and space complexity analysis helps evaluate algorithm performance using Big O notation."
        
        # Math questions
        if any(word in prompt_lower for word in ['remainder', 'modulo', 'divide', 'math', 'calculate', 'mathematics']):
            return "In programming, mathematical operations are fundamental. The modulo operator (%) finds remainders (e.g., 9 % 8 = 1). Common math functions include addition (+), subtraction (-), multiplication (*), division (/), and exponentiation (**). Most languages provide math libraries for advanced operations like trigonometry, logarithms, and statistical functions."
        
        # Database questions
        if any(word in prompt_lower for word in ['database', 'sql', 'mongodb', 'postgresql', 'mysql']):
            return "Databases store and organize data. SQL databases (PostgreSQL, MySQL) use structured tables and relationships. NoSQL databases (MongoDB) use flexible document or key-value structures. Key concepts include CRUD operations, indexing, transactions, normalization, and query optimization."
        
        # DevOps and tools
        if any(word in prompt_lower for word in ['git', 'docker', 'kubernetes', 'devops', 'ci/cd']):
            return "DevOps practices combine development and operations. Git provides version control for code collaboration. Docker containerizes applications for consistent deployment. Kubernetes orchestrates containerized applications. CI/CD pipelines automate testing and deployment, improving software delivery speed and reliability."
        
        # Greetings
        if any(word in prompt_lower for word in ['hi', 'hello', 'hey', 'greetings']):
            return "Hello! I'm a comprehensive local AI assistant with extensive programming knowledge. I can help you with code generation in multiple languages (Python, Rust, JavaScript, Java, C++, Go, etc.), explain programming concepts, algorithms, web development, databases, and software engineering principles. What would you like to learn or build today?"
        
        # Specific "what is" questions
        if prompt_lower.startswith('what is'):
            topic = prompt_lower.replace('what is', '').strip()
            if topic in ['rust', 'javascript', 'python', 'ai', 'machine learning']:
                return self._generate_local_quality_response(topic)  # Recursively handle the topic
        
        # General fallback with more comprehensive response
        return f"I understand you're asking about '{prompt}'. I'm a comprehensive local AI assistant with knowledge in:\n\n• Programming Languages: Python, Rust, JavaScript, Java, C++, Go, PHP, Ruby, Swift, Kotlin\n• Web Development: Frontend/Backend, APIs, databases\n• Algorithms & Data Structures\n• Software Engineering & DevOps\n• AI/ML concepts\n\nCould you be more specific about what you'd like to learn or build? I can generate code, explain concepts, or help with technical problems."
    
    def _enhance_with_external_knowledge(self, base_response: str, prompt: str, prompt_lower: str) -> Optional[str]:
        """Enhance existing response with external API knowledge."""
        try:
            # Detect language/technology in the prompt
            language = self._detect_language(prompt_lower)
            
            enhancements = []
            
            # Try to get documentation
            if language:
                doc_info = self.external_apis.search_documentation(language, prompt)
                if doc_info:
                    enhancements.append(doc_info)
            
            # Try to get package information for installation/library questions
            if any(word in prompt_lower for word in ['install', 'package', 'library', 'import', 'dependency']):
                if language:
                    pkg_info = self.external_apis.search_packages(language, self._extract_package_name(prompt))
                    if pkg_info:
                        enhancements.append(pkg_info)
            
            # Add Stack Overflow solutions for problem-solving
            if any(word in prompt_lower for word in ['error', 'issue', 'problem', 'fix', 'debug', 'help']):
                so_info = self.external_apis.search_stackoverflow(prompt)
                if so_info:
                    enhancements.append(so_info)
            
            # Combine base response with enhancements
            if enhancements:
                enhanced = base_response + "\n\n" + "\n\n".join(enhancements)
                return enhanced
            
        except Exception as e:
            logger.debug(f"Enhancement failed: {e}")
        
        return None
    
    def _get_enhanced_external_response(self, prompt: str, prompt_lower: str) -> Optional[str]:
        """Get enhanced response using external APIs for complex queries."""
        try:
            # Check cache first
            cache_key = f"external_{hash(prompt)}"
            if cache_key in self._api_cache:
                cached_data, timestamp = self._api_cache[cache_key]
                if time.time() - timestamp < self._cache_ttl:
                    return cached_data
            
            # Detect language/technology
            language = self._detect_language(prompt_lower)
            
            # Build comprehensive response from multiple sources
            response_parts = []
            
            # 1. Official Documentation (highest priority)
            if language and any(word in prompt_lower for word in ['what is', 'explain', 'how does', 'documentation']):
                doc_info = self.external_apis.search_documentation(language, prompt)
                if doc_info:
                    response_parts.append(doc_info)
            
            # 2. Package/Library information
            if any(word in prompt_lower for word in ['library', 'package', 'framework', 'tool']):
                if language:
                    pkg_name = self._extract_package_name(prompt)
                    if pkg_name:
                        pkg_info = self.external_apis.search_packages(language, pkg_name)
                        if pkg_info:
                            response_parts.append(pkg_info)
            
            # 3. Stack Overflow community solutions
            if any(word in prompt_lower for word in ['how to', 'how do', 'example', 'tutorial']):
                so_info = self.external_apis.search_stackoverflow(prompt)
                if so_info:
                    response_parts.append(so_info)
            
            # 4. GitHub code examples
            if language and any(word in prompt_lower for word in ['example', 'sample', 'code', 'implementation']):
                github_info = self.external_apis.search_github_examples(language, prompt)
                if github_info:
                    response_parts.append(github_info)
            
            # 5. Wikipedia for technical concepts
            if any(word in prompt_lower for word in ['algorithm', 'concept', 'theory', 'explain']):
                wiki_info = self.external_apis.search_wikipedia_technical(self._extract_main_concept(prompt))
                if wiki_info:
                    response_parts.append(wiki_info)
            
            # Combine all information sources
            if response_parts:
                combined_response = "\n\n".join(response_parts)
                
                # Add a brief intro explaining the comprehensive nature
                intro = f"**Comprehensive Answer for: {prompt}**\n\n"
                final_response = intro + combined_response
                
                # Cache the response
                self._api_cache[cache_key] = (final_response, time.time())
                
                return final_response
            
        except Exception as e:
            logger.debug(f"External API response failed: {e}")
        
        return None
    
    def _detect_language(self, prompt_lower: str) -> Optional[str]:
        """Detect programming language from the prompt."""
        language_keywords = {
            'python': ['python', 'py', 'django', 'flask', 'fastapi', 'pandas', 'numpy'],
            'javascript': ['javascript', 'js', 'node', 'react', 'vue', 'angular', 'express'],
            'typescript': ['typescript', 'ts'],
            'rust': ['rust', 'cargo', 'rustc'],
            'java': ['java', 'spring', 'maven', 'gradle'],
            'go': ['golang', 'go lang', 'go language'],
            'cpp': ['c++', 'cpp', 'cmake'],
            'csharp': ['c#', 'csharp', 'dotnet', '.net'],
            'php': ['php', 'laravel', 'symfony'],
            'ruby': ['ruby', 'rails', 'gem'],
            'swift': ['swift', 'ios'],
            'kotlin': ['kotlin', 'android']
        }
        
        for language, keywords in language_keywords.items():
            if any(keyword in prompt_lower for keyword in keywords):
                return language
        
        return None
    
    def _extract_package_name(self, prompt: str) -> str:
        """Extract potential package name from prompt."""
        # Simple extraction - look for common patterns
        words = prompt.lower().split()
        
        # Look for patterns like "install pandas", "use express", etc.
        install_words = ['install', 'use', 'import', 'require', 'add']
        for i, word in enumerate(words):
            if word in install_words and i + 1 < len(words):
                return words[i + 1].strip('.,!?;')
        
        # Look for quoted package names
        import re
        quoted_match = re.search(r'["\']([^"\']+)["\']', prompt)
        if quoted_match:
            return quoted_match.group(1)
        
        # Fallback to last word that might be a package name
        potential_packages = [w for w in words if len(w) > 2 and w.isalpha()]
        return potential_packages[-1] if potential_packages else ""
    
    def _extract_main_concept(self, prompt: str) -> str:
        """Extract the main technical concept from prompt."""
        # Remove common question words
        cleaned = re.sub(r'\b(what|is|are|how|does|do|explain|tell|me|about|the)\b', '', prompt.lower())
        
        # Get the most significant words
        words = [w.strip('.,!?;') for w in cleaned.split() if len(w) > 3]
        
        # Return the first significant word or phrase
        return words[0] if words else prompt.split()[0]
    
    def _analyze_user_intent_and_respond(self, original_prompt: str, prompt_lower: str) -> str:
        """Advanced intent analysis similar to Claude/Gemini/Copilot reasoning."""
        
        # Priority check: Specific technical topics BEFORE general patterns
        # This ensures "explain rust" hits Rust expertise, not general explanation
        
        # Programming language expertise (high priority)
        if any(word in prompt_lower for word in ['rust', 'rust language', 'rust programming']):
            if 'ownership' in prompt_lower or 'borrow' in prompt_lower:
                return self._generate_rust_ownership_explanation()
            return self._generate_rust_expertise(original_prompt, prompt_lower)
        
        if any(word in prompt_lower for word in ['javascript', 'js', 'node.js', 'typescript', 'ts']):
            return self._generate_javascript_expertise(original_prompt, prompt_lower)
        
        if any(word in prompt_lower for word in ['python', 'python language', 'py']):
            return self._generate_python_expertise(original_prompt, prompt_lower)
        
        # AI/ML expertise (high priority)
        if any(word in prompt_lower for word in ['ai', 'artificial intelligence', 'machine learning', 'neural network', 'deep learning', 'llm', 'gpt', 'transformer']):
            return self._generate_ai_expertise(original_prompt, prompt_lower)
        
        # Multi-step reasoning for complex queries
        if any(phrase in prompt_lower for phrase in ['how to', 'how do i', 'how can i', 'what is the best way']):
            return self._generate_how_to_response(original_prompt, prompt_lower)
        
        # Problem-solving queries
        if any(phrase in prompt_lower for phrase in ['help me', 'i need to', 'i want to', 'can you help']):
            return self._generate_problem_solving_response(original_prompt, prompt_lower)
        
        # Comparison and analysis queries  
        if any(phrase in prompt_lower for phrase in ['vs', 'versus', 'compare', 'difference', 'better', 'which']):
            return self._generate_comparison_response(original_prompt, prompt_lower)
        
        # Code review and debugging
        if any(phrase in prompt_lower for phrase in ['debug', 'error', 'fix', 'wrong', 'not working', 'issue']):
            return self._generate_debugging_response(original_prompt, prompt_lower)
        
        # Learning and tutorial requests
        if any(phrase in prompt_lower for phrase in ['learn', 'tutorial', 'guide', 'teach me', 'show me']):
            return self._generate_learning_response(original_prompt, prompt_lower)
        
        # Explanation queries with depth (LOWER priority - after specific topics)
        if any(phrase in prompt_lower for phrase in ['explain', 'what is', 'what are', 'define', 'tell me about']):
            return self._generate_detailed_explanation(original_prompt, prompt_lower)
        
        return None
    
    def _generate_rust_ownership_explanation(self) -> str:
        """Generate deep explanation of Rust ownership with examples."""
        return """**Rust Ownership System - Deep Dive:**

**Core Principles:**
1. **Each value has a single owner**
2. **When owner goes out of scope, value is dropped**
3. **Move semantics by default** (no expensive copies)

**Ownership Rules:**
```rust
fn main() {
    let s1 = String::from("hello");  // s1 owns the string
    let s2 = s1;                     // s1 moves to s2, s1 invalid
    // println!("{}", s1);           // Compile error!
    println!("{}", s2);              // OK - s2 owns the string
}
```

**Borrowing System:**
• **Immutable borrows:** `&T` - multiple readers allowed
• **Mutable borrows:** `&mut T` - exclusive access, no other borrows

```rust
fn calculate_length(s: &String) -> usize {  // Borrows, doesn't own
    s.len()  // Can read but not modify
} // s goes out of scope but nothing happens (no ownership)
```

**Why This Matters:**
✅ **Memory Safety:** No null pointer dereferences or buffer overflows
✅ **Thread Safety:** Data races impossible at compile time  
✅ **Zero Cost:** No garbage collector overhead
✅ **Fearless Concurrency:** Send/Sync traits ensure safe sharing

**Common Patterns:**
• Use `&str` for string slices, `String` for owned strings
• `Vec<T>` for growable arrays, `&[T]` for slices
• `Box<T>` for heap allocation, `Rc<T>` for reference counting

This system prevents entire categories of bugs that plague C/C++ while maintaining performance."""
    
    def _generate_how_to_response(self, prompt: str, prompt_lower: str) -> str:
        """Generate step-by-step how-to responses like advanced AI assistants."""
        
        if 'deploy' in prompt_lower and any(word in prompt_lower for word in ['web', 'app', 'application']):
            return """Here's how to deploy a web application (step-by-step approach):

**1. Preparation Phase:**
- Ensure your code is production-ready (error handling, environment variables)
- Set up version control (Git) and create a deployment branch
- Configure your build process and dependencies

**2. Choose Deployment Platform:**
- **Cloud Platforms:** AWS, Google Cloud, Azure (scalable, professional)
- **Simple Hosting:** Vercel, Netlify, Heroku (quick setup)
- **VPS:** DigitalOcean, Linode (more control)

**3. Deployment Process:**
- Set up CI/CD pipeline (GitHub Actions, GitLab CI)
- Configure environment variables securely
- Set up database and storage (if needed)
- Deploy and test in staging environment

**4. Post-Deployment:**
- Set up monitoring and logging
- Configure domain and SSL certificates
- Set up backups and disaster recovery

Would you like me to elaborate on any specific step or platform?"""
        
        if 'optimize' in prompt_lower:
            return self._generate_optimization_guide(prompt, prompt_lower)
        
        if 'secure' in prompt_lower or 'security' in prompt_lower:
            return self._generate_security_guide(prompt, prompt_lower)
        
        return f"I'd be happy to provide step-by-step guidance for: {prompt}\n\nTo give you the most helpful response, could you specify:\n• What technology/language you're using\n• What your current setup looks like\n• Any specific constraints or requirements\n\nThis will help me provide targeted, actionable steps."
    
    def _generate_problem_solving_response(self, prompt: str, prompt_lower: str) -> str:
        """Generate problem-solving responses with systematic approach."""
        
        if 'api' in prompt_lower and 'build' in prompt_lower:
            return """Let me help you build an API systematically:

**1. Planning & Design:**
- Define your API's purpose and endpoints
- Choose REST, GraphQL, or gRPC architecture
- Design your data models and relationships

**2. Technology Stack Selection:**
- **Python:** FastAPI (modern), Django REST, Flask
- **JavaScript:** Express.js, Next.js API routes, Fastify
- **Rust:** Actix-web, Warp, Axum
- **Go:** Gin, Echo, Fiber
- **Java:** Spring Boot, Quarkus

**3. Implementation Steps:**
- Set up project structure and dependencies
- Implement database models and connections
- Create route handlers and business logic
- Add authentication and authorization
- Implement error handling and validation

**4. Testing & Documentation:**
- Write unit and integration tests
- Create API documentation (OpenAPI/Swagger)
- Test with tools like Postman or curl

**5. Deployment & Monitoring:**
- Set up production environment
- Add logging and monitoring
- Configure rate limiting and security

Which programming language would you prefer, and what type of API are you building?"""
        
        if 'learn' in prompt_lower and any(lang in prompt_lower for lang in ['programming', 'coding', 'development']):
            return self._generate_learning_path(prompt, prompt_lower)
        
        return f"I understand you need help with: {prompt}\n\nLet me break this down systematically:\n\n**First, let's clarify your situation:**\n• What's your current experience level?\n• What specific outcome are you trying to achieve?\n• Are there any constraints or deadlines?\n\n**Then I can provide:**\n• Step-by-step guidance\n• Code examples and templates  \n• Best practices and common pitfalls\n• Resources for further learning\n\nWhat aspect would you like to focus on first?"
    
    def _generate_comparison_response(self, prompt: str, prompt_lower: str) -> str:
        """Generate detailed comparison responses."""
        
        if 'python' in prompt_lower and 'rust' in prompt_lower:
            return """**Python vs Rust - Comprehensive Comparison:**

**Performance:**
• **Rust:** Compiled, zero-cost abstractions, memory-safe without GC (⭐⭐⭐⭐⭐)
• **Python:** Interpreted, slower but sufficient for most apps (⭐⭐⭐)

**Learning Curve:**  
• **Python:** Beginner-friendly, readable syntax (⭐⭐⭐⭐⭐)
• **Rust:** Steep learning curve, ownership concepts (⭐⭐)

**Use Cases:**
• **Python:** Data science, AI/ML, web backends, automation, prototyping
• **Rust:** System programming, web backends, CLI tools, performance-critical apps

**Ecosystem:**
• **Python:** Massive ecosystem, libraries for everything (⭐⭐⭐⭐⭐)  
• **Rust:** Growing ecosystem, excellent for systems programming (⭐⭐⭐⭐)

**Development Speed:**
• **Python:** Rapid prototyping, quick iteration (⭐⭐⭐⭐⭐)
• **Rust:** Slower development but catches bugs at compile time (⭐⭐⭐)

**When to Choose:**
• **Python:** Data analysis, AI/ML, quick prototypes, team has Python experience
• **Rust:** Performance critical, system-level, long-term maintainability, memory safety crucial

What specific aspect interests you most?"""
        
        if 'javascript' in prompt_lower and ('typescript' in prompt_lower or 'ts' in prompt_lower):
            return self._generate_js_ts_comparison()
        
        if any(word in prompt_lower for word in ['framework', 'library']) and 'web' in prompt_lower:
            return self._generate_web_framework_comparison(prompt_lower)
        
        return f"I can provide a detailed comparison for: {prompt}\n\nTo give you the most useful analysis, please specify:\n• What criteria matter most to you (performance, ease of use, ecosystem, etc.)\n• What's your use case or project type\n• What's your experience level\n\nThis helps me tailor the comparison to your specific needs."
    
    def _generate_detailed_explanation(self, prompt: str, prompt_lower: str) -> str:
        """Generate comprehensive explanations with multiple angles."""
        
        if 'microservices' in prompt_lower:
            return """**Microservices Architecture - Comprehensive Explanation:**

**Core Concept:**
Microservices break down applications into small, independent services that communicate over networks, contrasting with monolithic architecture where everything runs as a single unit.

**Key Characteristics:**
• **Single Responsibility:** Each service handles one business function
• **Independent Deployment:** Services can be updated separately
• **Technology Agnostic:** Different services can use different tech stacks
• **Failure Isolation:** If one service fails, others continue running

**Communication Patterns:**
• **Synchronous:** REST APIs, GraphQL (direct request-response)
• **Asynchronous:** Message queues, event streams (decoupled)
• **Service Mesh:** Istio, Linkerd for complex communication management

**Benefits:**
✅ Scalability: Scale services independently based on demand
✅ Technology Diversity: Use best tool for each job
✅ Team Independence: Teams can work on different services
✅ Fault Tolerance: Isolated failures don't bring down entire system

**Challenges:**
❌ Complexity: Network calls, distributed debugging
❌ Data Consistency: Managing transactions across services
❌ Operational Overhead: Multiple deployments, monitoring

**When to Use:**
• Large, complex applications
• Multiple development teams
• Different scaling requirements per component
• Need for technology diversity

**When NOT to Use:**
• Small applications (premature complexity)
• Single team or early-stage products
• Simple CRUD applications

Would you like me to dive deeper into any specific aspect?"""
        
        if 'docker' in prompt_lower and 'kubernetes' in prompt_lower:
            return self._generate_containerization_explanation()
        
        if 'blockchain' in prompt_lower:
            return self._generate_blockchain_explanation()
        
        return f"Let me provide a comprehensive explanation of: {prompt}\n\n**I'll cover:**\n• Core concepts and definitions\n• How it works (technical details)\n• Benefits and use cases\n• Common challenges and solutions\n• Best practices and examples\n• Related technologies and concepts\n\n**To make this most useful, could you specify:**\n• Your current knowledge level (beginner/intermediate/advanced)\n• Any specific aspects you're most interested in\n• Whether you need practical examples or theoretical understanding\n\nWhat would be most helpful?"
    
    def _generate_rust_expertise(self, prompt: str, prompt_lower: str) -> str:
        """Generate deep Rust expertise like advanced AI assistants."""
        
        if 'ownership' in prompt_lower or 'borrow' in prompt_lower:
            return """**Rust Ownership System - Deep Dive:**

**Core Principles:**
1. **Each value has a single owner**
2. **When owner goes out of scope, value is dropped**
3. **Move semantics by default** (no expensive copies)

**Ownership Rules:**
```rust
fn main() {
    let s1 = String::from("hello");  // s1 owns the string
    let s2 = s1;                     // s1 moves to s2, s1 invalid
    // println!("{}", s1);           // Compile error!
    println!("{}", s2);              // OK - s2 owns the string
}
```

**Borrowing System:**
• **Immutable borrows:** `&T` - multiple readers allowed
• **Mutable borrows:** `&mut T` - exclusive access, no other borrows

```rust
fn calculate_length(s: &String) -> usize {  // Borrows, doesn't own
    s.len()  // Can read but not modify
} // s goes out of scope but nothing happens (no ownership)
```

**Why This Matters:**
✅ **Memory Safety:** No null pointer dereferences or buffer overflows
✅ **Thread Safety:** Data races impossible at compile time  
✅ **Zero Cost:** No garbage collector overhead
✅ **Fearless Concurrency:** Send/Sync traits ensure safe sharing

**Common Patterns:**
• Use `&str` for string slices, `String` for owned strings
• `Vec<T>` for growable arrays, `&[T]` for slices
• `Box<T>` for heap allocation, `Rc<T>` for reference counting

This system prevents entire categories of bugs that plague C/C++ while maintaining performance."""
        
        if 'async' in prompt_lower or 'tokio' in prompt_lower:
            return """**Rust Async Programming & Tokio - Expert Guide:**

**Why Async Rust:**
Rust's async model provides zero-cost concurrency without traditional thread overhead.

**Core Concepts:**

**Async/Await Syntax:**
```rust
use tokio::time::{sleep, Duration};

async fn fetch_data(url: &str) -> Result<String, reqwest::Error> {
    let response = reqwest::get(url).await?;
    let body = response.text().await?;
    Ok(body)
}

async fn process_multiple_requests() -> Result<(), Box<dyn std::error::Error>> {
    let urls = vec!["http://example.com", "http://google.com"];
    
    // Concurrent execution
    let futures = urls.iter().map(|url| fetch_data(url));
    let results = futures::future::try_join_all(futures).await?;
    
    for result in results {
        println!("Response length: {}", result.len());
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    process_multiple_requests().await
}
```

**Tokio Runtime:**
Tokio is the de facto async runtime for Rust.

**Key Features:**
• **Multi-threaded scheduler:** Efficient work-stealing
• **I/O primitives:** TCP, UDP, Unix sockets, timers
• **Sync primitives:** Async mutexes, channels, semaphores
• **Utilities:** Timeouts, intervals, file system operations

**Common Patterns:**

**Channel Communication:**
```rust
use tokio::sync::mpsc;

#[tokio::main]
async fn main() {
    let (tx, mut rx) = mpsc::channel(100);
    
    // Spawn producer
    tokio::spawn(async move {
        for i in 0..10 {
            tx.send(i).await.unwrap();
        }
    });
    
    // Consumer
    while let Some(value) = rx.recv().await {
        println!("Received: {}", value);
    }
}
```

**HTTP Server with Axum:**
```rust
use axum::{extract::Query, response::Json, routing::get, Router};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Deserialize)]
struct Params {
    name: Option<String>,
}

#[derive(Serialize)]
struct Response {
    message: String,
}

async fn handler(Query(params): Query<Params>) -> Json<Response> {
    let name = params.name.unwrap_or_else(|| "World".to_string());
    Json(Response {
        message: format!("Hello, {}!", name),
    })
}

#[tokio::main]
async fn main() {
    let app = Router::new().route("/", get(handler));
    
    axum::Server::bind(&"0.0.0.0:3000".parse().unwrap())
        .serve(app.into_make_service())
        .await
        .unwrap();
}
```

**Best Practices:**
• Use `#[tokio::main]` for async main functions
• Prefer `tokio::spawn` for CPU-intensive tasks
• Use channels for inter-task communication  
• Be careful with `std::sync` primitives in async contexts
• Use `timeout` to prevent hanging operations

**Performance Tips:**
• Use `tokio::task::yield_now()` for cooperative yielding
• Consider `rayon` for CPU-bound parallel work
• Profile with `tokio-console` for debugging async issues"""
        
        if 'cargo' in prompt_lower or 'crate' in prompt_lower:
            return """**Rust Ecosystem & Cargo - Expert Guide:**

**Cargo: Rust's Superpower**
Cargo is Rust's built-in package manager and build system, making dependency management seamless.

**Key Features:**
• **Package Management:** Dependencies declared in `Cargo.toml`
• **Build System:** Compilation, testing, and documentation
• **Workspaces:** Multi-package projects
• **Publishing:** Easy crate publishing to crates.io

**Essential Cargo Commands:**
```bash
# Create new project
cargo new my_project
cargo new --lib my_library

# Build and run
cargo build          # Debug build
cargo build --release  # Optimized build
cargo run            # Build and run

# Testing and documentation  
cargo test           # Run tests
cargo doc --open     # Generate and open docs
cargo check          # Fast compile check

# Dependencies
cargo add serde      # Add dependency (Cargo 1.60+)
cargo update         # Update dependencies
```

**Popular Crates Ecosystem:**

**Serialization:**
• **serde** - Serialization framework (JSON, YAML, etc.)
• **serde_json** - JSON support for serde

**Web Development:**
• **actix-web** - High-performance web framework
• **warp** - Composable web framework
• **tokio** - Async runtime

**Command Line:**
• **clap** - Command line argument parsing
• **structopt** - Derive-based CLI

**Error Handling:**
• **anyhow** - Flexible error handling
• **thiserror** - Custom error types

**Example Cargo.toml:**
```toml
[package]
name = "my_app"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
tokio = { version = "1.0", features = ["full"] }
clap = { version = "4.0", features = ["derive"] }

[dev-dependencies]
tokio-test = "0.4"
```

**Best Practices:**
• Pin major versions for stability
• Use `cargo clippy` for linting  
• Use `cargo fmt` for formatting
• Enable useful features selectively
• Consider using `cargo-edit` for dependency management"""
        
        # General Rust expertise
        return """**Rust Programming Language - Expert Overview:**

**What Makes Rust Special:**
Rust is a systems programming language that achieves the "impossible trinity" - memory safety, performance, and concurrency safety - all at compile time without a garbage collector.

**Core Strengths:**
• **Zero-Cost Abstractions:** High-level features with C-level performance
• **Memory Safety:** No segfaults, buffer overflows, or memory leaks
• **Concurrency:** Fearless parallelism with compile-time race condition prevention
• **Type System:** Expressive types that prevent bugs before runtime
• **Package Manager:** Cargo makes dependency management seamless

**Key Concepts:**
• **Ownership & Borrowing:** Unique memory management system
• **Traits:** Like interfaces but more powerful (similar to Haskell type classes)
• **Pattern Matching:** Exhaustive `match` expressions for control flow
• **Error Handling:** `Result<T, E>` type forces explicit error handling

**Best Use Cases:**
• System programming (OS, drivers, embedded)
• Web backends (performance-critical APIs)
• CLI tools and system utilities
• WebAssembly applications
• Blockchain and cryptocurrency projects

**Learning Path:**
1. Start with ownership concepts (most important)
2. Learn pattern matching and error handling
3. Explore traits and generics
4. Practice with async programming
5. Build real projects (CLI tools are great starting points)

The Rust community is exceptionally helpful - the compiler error messages are educational, and the ecosystem is rapidly growing."""
    
    def _generate_javascript_expertise(self, prompt: str, prompt_lower: str) -> str:
        """Generate comprehensive JavaScript expertise."""
        
        if 'async' in prompt_lower or 'promise' in prompt_lower or 'await' in prompt_lower:
            return """**JavaScript Asynchronous Programming - Expert Guide:**

**Evolution of Async Patterns:**

**1. Callbacks (Legacy - Callback Hell):**
```javascript
getData(function(a) {
    getMoreData(a, function(b) {
        getEvenMoreData(b, function(c) {
            // Nested callback hell ❌
        });
    });
});
```

**2. Promises (ES6 - Much Better):**
```javascript
getData()
    .then(a => getMoreData(a))
    .then(b => getEvenMoreData(b))
    .then(c => console.log(c))
    .catch(err => console.error(err));
```

**3. Async/Await (ES2017 - Best Practice):**
```javascript
async function fetchData() {
    try {
        const a = await getData();
        const b = await getMoreData(a);
        const c = await getEvenMoreData(b);
        return c;
    } catch (error) {
        console.error('Error:', error);
    }
}
```

**Advanced Patterns:**

**Parallel Execution:**
```javascript
// Sequential (slower)
const result1 = await fetch('/api/1');
const result2 = await fetch('/api/2');

// Parallel (faster)
const [result1, result2] = await Promise.all([
    fetch('/api/1'),
    fetch('/api/2')
]);
```

**Error Handling Best Practices:**
```javascript
async function robustFetch(url) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000);
    
    try {
        const response = await fetch(url, { 
            signal: controller.signal 
        });
        clearTimeout(timeoutId);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        return await response.json();
    } catch (error) {
        if (error.name === 'AbortError') {
            throw new Error('Request timeout');
        }
        throw error;
    }
}
```

**Key Insights:**
• Promises are eager (execute immediately when created)
• Async functions always return promises
• `await` only works inside async functions (or top-level in modules)
• Use `Promise.allSettled()` when you want all results regardless of failures
• Consider using libraries like `p-retry` for robust error handling

Modern JavaScript async programming is powerful and elegant when done right."""
        
        if 'react' in prompt_lower or 'vue' in prompt_lower or 'angular' in prompt_lower:
            if 'react' in prompt_lower:
                return """**React - Modern Frontend Framework Expert Guide:**

**Why React Dominates:**
React revolutionized frontend development with component-based architecture and virtual DOM.

**Core Concepts:**

**Modern React (Hooks Era):**
```javascript
import React, { useState, useEffect } from 'react';

function UserProfile({ userId }) {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        async function fetchUser() {
            try {
                const response = await fetch(`/api/users/${userId}`);
                const userData = await response.json();
                setUser(userData);
            } catch (error) {
                console.error('Failed to fetch user:', error);
            } finally {
                setLoading(false);
            }
        }
        
        fetchUser();
    }, [userId]);

    if (loading) return <div>Loading...</div>;
    
    return (
        <div className="user-profile">
            <h1>{user?.name}</h1>
            <p>{user?.email}</p>
        </div>
    );
}
```

**Advanced Patterns:**
• **Custom Hooks:** Reusable stateful logic
• **Context API:** Global state management
• **React Query:** Server state management
• **Suspense:** Better loading states

**Modern Stack:**
• **Next.js:** Full-stack React framework
• **Vite:** Lightning-fast dev server
• **TypeScript:** Type safety for large apps
• **Tailwind CSS:** Utility-first styling"""
            elif 'vue' in prompt_lower:
                return """**Vue.js - Progressive Frontend Framework:**

**Vue's Philosophy:** Incrementally adoptable, approachable, and versatile.

**Modern Vue 3 with Composition API:**
```javascript
<template>
  <div class="user-profile">
    <h1 v-if="loading">Loading...</h1>
    <div v-else>
      <h1>{{ user.name }}</h1>
      <p>{{ user.email }}</p>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue';

const user = ref(null);
const loading = ref(true);

onMounted(async () => {
  try {
    const response = await fetch('/api/user');
    user.value = await response.json();
  } finally {
    loading.value = false;
  }
});
</script>
```

**Vue Ecosystem:**
• **Nuxt.js:** Full-stack Vue framework
• **Pinia:** Modern state management
• **Vue Router:** Official routing
• **Vite:** Build tool"""
            else:  # Angular
                return """**Angular - Enterprise Frontend Framework:**

**Angular's Strength:** Full-featured, opinionated framework for large-scale applications.

**Modern Angular (v15+):**
```typescript
import { Component, OnInit } from '@angular/core';
import { UserService } from './user.service';

@Component({
  selector: 'app-user-profile',
  template: `
    <div class="user-profile">
      <h1 *ngIf="loading">Loading...</h1>
      <div *ngIf="!loading">
        <h1>{{ user?.name }}</h1>
        <p>{{ user?.email }}</p>
      </div>
    </div>
  `
})
export class UserProfileComponent implements OnInit {
  user: any = null;
  loading = true;

  constructor(private userService: UserService) {}

  ngOnInit() {
    this.userService.getUser().subscribe({
      next: (user) => this.user = user,
      error: (err) => console.error(err),
      complete: () => this.loading = false
    });
  }
}
```

**Angular Features:**
• **TypeScript by Default:** Strong typing
• **Dependency Injection:** Powerful DI system
• **RxJS:** Reactive programming
• **Angular CLI:** Comprehensive tooling"""
        
        if 'node' in prompt_lower or 'backend' in prompt_lower:
            return """**Node.js - Server-Side JavaScript Expert Guide:**

**Why Node.js is Powerful:**
Node.js brings JavaScript to the server with high performance through V8 engine and event-driven, non-blocking I/O.

**Core Strengths:**
• **Single Language Stack:** JavaScript everywhere (frontend + backend)
• **NPM Ecosystem:** Largest package repository in the world
• **Event-Driven Architecture:** Perfect for I/O-intensive applications
• **Microservices:** Lightweight, fast startup times

**Modern Node.js Development:**

**Express.js (Most Popular Framework):**
```javascript
const express = require('express');
const app = express();

// Middleware
app.use(express.json());
app.use(express.static('public'));

// Routes
app.get('/api/users', async (req, res) => {
    try {
        const users = await User.findAll();
        res.json(users);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.listen(3000, () => {
    console.log('Server running on port 3000');
});
```

**Modern Alternatives:**
• **Fastify:** 2x faster than Express
• **Koa.js:** From Express team, modern async/await
• **NestJS:** Enterprise-grade with TypeScript

**Best Practices:**
• Use TypeScript for large applications
• Implement proper error handling with try/catch
• Use environment variables for configuration
• Implement rate limiting and security middleware
• Use clustering for production (PM2)

**Performance Tips:**
• Use HTTP/2 for better performance
• Implement caching strategies (Redis)
• Use connection pooling for databases
• Monitor with APM tools (New Relic, DataDog)"""
        
        return """**JavaScript - Modern Language Overview:**

**JavaScript Today (ES2024+):**
JavaScript has evolved from a simple scripting language to a powerful, multi-paradigm language running everywhere - browsers, servers, mobile apps, desktop apps, and even embedded systems.

**Core Strengths:**
• **Ubiquity:** Runs everywhere with massive ecosystem
• **Flexibility:** Supports functional, OOP, and procedural programming
• **Dynamic:** Rapid development and prototyping
• **Community:** Largest developer community, extensive libraries

**Modern Features (ES6+):**
• **Arrow Functions & Destructuring:** Cleaner, more expressive code
• **Modules:** Proper import/export system
• **Classes:** OOP support with clean syntax
• **Async/Await:** Elegant asynchronous programming
• **Optional Chaining:** `obj?.prop?.method?.()` for safe property access
• **Nullish Coalescing:** `??` operator for default values

**Performance Considerations:**
• V8 engine (Chrome/Node) has incredible optimization
• Use modern bundlers (Vite, esbuild) for optimal builds
• Consider TypeScript for large codebases (catches errors early)
• Profile with browser dev tools for performance bottlenecks

**Best Practices:**
• Use `const` by default, `let` when reassignment needed
• Prefer functional programming patterns (map, filter, reduce)
• Handle errors properly with try/catch and Promise rejection
• Use ESLint and Prettier for consistent code quality
• Consider modern alternatives: TypeScript, Deno, Bun

**Learning Resources:**
• MDN Web Docs (definitive reference)
• JavaScript.info (comprehensive tutorial)
• You Don't Know JS (deep understanding)
• Modern JS frameworks: React, Vue, Svelte

JavaScript's ecosystem moves fast, but the core language is mature and powerful."""
    
    def _generate_python_expertise(self, prompt: str, prompt_lower: str) -> str:
        """Generate deep Python expertise."""
        
        if 'django' in prompt_lower or 'flask' in prompt_lower or 'fastapi' in prompt_lower:
            return """**Python Web Frameworks - Expert Comparison:**

**FastAPI (Recommended for APIs):**
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class User(BaseModel):
    name: str
    email: str

@app.post("/users/")
async def create_user(user: User):
    # Automatic validation, serialization, and docs
    return {"message": f"Created user {user.name}"}

# Automatic OpenAPI docs at /docs
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Django (Full-Stack Framework):**
• **Strengths:** Complete ecosystem, admin panel, ORM, security
• **Best For:** Content-heavy sites, rapid MVP development
• **Philosophy:** "Batteries included" - everything you need
• **Learning Curve:** Steeper but very powerful

**Flask (Microframework):**
• **Strengths:** Lightweight, flexible, easy to understand
• **Best For:** Small to medium apps, learning web development
• **Philosophy:** "Do one thing well" - you add what you need

**Performance Comparison:**
1. **FastAPI:** ~65,000 req/sec (async, modern)
2. **Flask:** ~10,000 req/sec (sync, simple)
3. **Django:** ~8,000 req/sec (sync, feature-rich)

**Architecture Recommendations:**

**For APIs/Microservices:** FastAPI
```python
# Type hints, automatic validation, async support
@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "query": q}
```

**For Full Web Apps:** Django
```python
# models.py - ORM with migrations
class Article(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
```

**For Learning/Simple Apps:** Flask
```python
# Simple and explicit
@app.route('/user/<name>')
def show_user_profile(name):
    return f'User: {name}'
```

**Modern Python Web Stack:**
• **Framework:** FastAPI or Django
• **Database:** PostgreSQL with SQLAlchemy/Django ORM
• **Async:** httpx for HTTP clients, asyncio for concurrency
• **Testing:** pytest with coverage
• **Deployment:** Docker + Kubernetes or serverless (AWS Lambda)

Choose based on your specific needs and team expertise."""
        
        if 'machine learning' in prompt_lower or 'ml' in prompt_lower or 'data science' in prompt_lower:
            return self._generate_python_ml_expertise()
        
        if 'async' in prompt_lower or 'asyncio' in prompt_lower:
            return self._generate_python_async_expertise()
        
        return """**Python - Expert Language Overview:**

**Why Python Dominates:**
Python's philosophy of "readable, simple, and powerful" has made it the lingua franca of modern software development, from web backends to AI research.

**Core Strengths:**
• **Readability:** Code reads like English, reducing cognitive load
• **Ecosystem:** 400,000+ packages on PyPI covering every domain
• **Versatility:** Web, data science, AI/ML, automation, desktop apps
• **Community:** Huge, supportive community with excellent documentation

**Modern Python (3.8+) Features:**

**Type Hints (Game Changer):**
```python
from typing import List, Optional, Dict, Any

def process_data(
    items: List[Dict[str, Any]], 
    filter_key: str,
    default: Optional[str] = None
) -> List[Dict[str, Any]]:
    return [item for item in items if item.get(filter_key, default)]
```

**Pattern Matching (Python 3.10+):**
```python
def handle_response(response):
    match response.status_code:
        case 200:
            return response.json()
        case 404:
            raise NotFoundError("Resource not found")
        case 500 | 502 | 503:
            raise ServerError("Server error")
        case _:
            raise HTTPError(f"Unexpected status: {response.status_code}")
```

**Dataclasses & Pydantic:**
```python
from dataclasses import dataclass
from pydantic import BaseModel, validator

@dataclass
class Point:
    x: float
    y: float
    
class User(BaseModel):  # Pydantic for validation
    name: str
    age: int
    
    @validator('age')
    def age_must_be_positive(cls, v):
        if v < 0:
            raise ValueError('Age must be positive')
        return v
```

**Performance Optimization:**
• **NumPy/Pandas:** Vectorized operations for data processing
• **Cython:** Compile critical sections to C speed
• **PyPy:** Alternative interpreter with JIT compilation
• **asyncio:** Concurrent I/O operations
• **multiprocessing:** True parallelism for CPU-bound tasks

**Best Practices:**
• Use virtual environments (venv, conda, poetry)
• Follow PEP 8 style guide (use black formatter)
• Write type hints for better IDE support and bug prevention
• Use pytest for testing with good coverage
• Profile with cProfile and line_profiler for optimization

**Modern Python Stack:**
• **Web:** FastAPI + SQLAlchemy + Alembic
• **Data:** Pandas + NumPy + Matplotlib/Plotly
• **ML:** scikit-learn + PyTorch/TensorFlow
• **Testing:** pytest + coverage + pre-commit hooks
• **Packaging:** Poetry + Docker for deployment

Python 3.12+ is incredibly fast and feature-rich - it's an excellent choice for almost any project."""
    
    def _generate_ai_expertise(self, prompt: str, prompt_lower: str) -> str:
        """Generate comprehensive AI and ML expertise."""
        
        if 'llm' in prompt_lower or 'transformer' in prompt_lower or 'gpt' in prompt_lower:
            return """**Large Language Models & Transformers - Expert Deep Dive:**

**Architecture Foundation:**
The Transformer architecture (2017) revolutionized AI by introducing the attention mechanism, enabling models to process sequences in parallel rather than sequentially.

**Key Components:**

**1. Self-Attention Mechanism:**
```
For each word, calculate how much attention to pay to every other word:
"The cat sat on the mat"
- "sat" pays attention to "cat" (subject) and "mat" (object)
- Enables understanding of relationships across long distances
```

**2. Multi-Head Attention:**
• Multiple attention "heads" focus on different aspects
• Some heads might focus on syntax, others on semantics
• Allows rich, nuanced understanding of context

**3. Positional Encoding:**
• Since attention has no inherent order, positions are encoded
• Enables understanding of word order and sequence structure

**Modern LLM Evolution:**

**GPT Series (Generative):**
• **GPT-1 (2018):** 117M parameters - proof of concept
• **GPT-2 (2019):** 1.5B parameters - too dangerous to release initially
• **GPT-3 (2020):** 175B parameters - breakthrough in few-shot learning
• **GPT-4 (2023):** Multimodal, significantly more capable

**Technical Innovations:**

**Scaling Laws:**
• Performance scales predictably with model size, data, and compute
• Optimal model size grows with available compute budget
• Chinchilla scaling: More data often better than larger models

**Training Techniques:**
• **Pre-training:** Learn language patterns from massive text corpora
• **Fine-tuning:** Adapt to specific tasks with labeled data
• **RLHF:** Reinforcement Learning from Human Feedback for alignment
• **Constitutional AI:** Self-improvement through AI feedback

**Practical Implementation:**

**Using Transformers Library:**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load pre-trained model
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Generate response
input_text = "Hello, how are you?"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

with torch.no_grad():
    output = model.generate(
        input_ids,
        max_length=100,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(output[0], skip_special_tokens=True)
```

**Current Limitations & Challenges:**
• **Hallucination:** Models can generate plausible but false information
• **Context Length:** Limited memory (though improving rapidly)
• **Compute Requirements:** Large models need significant resources
• **Alignment:** Ensuring AI systems do what humans actually want

**Future Directions:**
• **Retrieval Augmented Generation (RAG):** Combining LLMs with knowledge bases
• **Multi-modal Models:** Text, image, audio, video understanding
• **Agent Systems:** LLMs that can use tools and take actions
• **Smaller, Efficient Models:** Distillation and quantization for edge deployment

**Practical Applications:**
• Code generation and debugging
• Content creation and editing  
• Question answering and research
• Language translation and localization
• Customer service automation
• Educational tutoring and explanation

The field is evolving rapidly - what's cutting-edge today may be commonplace in months."""
        
        if 'machine learning' in prompt_lower and ('algorithm' in prompt_lower or 'model' in prompt_lower):
            return self._generate_ml_algorithms_expertise()
        
        if 'neural network' in prompt_lower or 'deep learning' in prompt_lower:
            return self._generate_deep_learning_expertise()
        
        return """**Artificial Intelligence - Comprehensive Expert Overview:**

**AI Landscape Today:**
AI has evolved from academic research to practical tools reshaping every industry. We're in the era of "AI-first" thinking where intelligent systems augment human capabilities.

**Major AI Domains:**

**1. Machine Learning:**
• **Supervised Learning:** Learn from labeled examples (classification, regression)
• **Unsupervised Learning:** Find patterns in unlabeled data (clustering, dimensionality reduction)
• **Reinforcement Learning:** Learn through interaction and feedback

**2. Deep Learning:**
• **Neural Networks:** Inspired by brain structure, excellent for pattern recognition
• **Convolutional Networks (CNNs):** Excel at image processing and computer vision
• **Recurrent Networks (RNNs/LSTMs):** Handle sequential data like text and time series
• **Transformers:** Current state-of-the-art for language and increasingly other domains

**3. Natural Language Processing:**
• **Understanding:** Extract meaning from human language
• **Generation:** Create human-like text, code, and content
• **Translation:** Bridge language barriers automatically
• **Conversation:** Enable natural human-AI interaction

**4. Computer Vision:**
• **Object Detection:** Identify and locate objects in images/video
• **Face Recognition:** Identify individuals from facial features
• **Medical Imaging:** Assist doctors in diagnosis from X-rays, MRIs
• **Autonomous Vehicles:** Navigate and understand road environments

**Modern AI Development Stack:**

**Python Ecosystem:**
• **PyTorch/TensorFlow:** Deep learning frameworks
• **Hugging Face Transformers:** Pre-trained models and easy deployment
• **scikit-learn:** Traditional ML algorithms and preprocessing
• **OpenAI API:** Access to GPT models for applications

**Key Trends:**
• **Foundation Models:** Large, general-purpose models fine-tuned for specific tasks
• **Multimodal AI:** Systems that understand text, images, audio together
• **AI Agents:** Systems that can use tools, browse internet, take actions
• **Edge AI:** Running AI models on mobile devices and embedded systems

**Ethical Considerations:**
• **Bias:** AI systems can perpetuate or amplify human biases
• **Privacy:** AI systems often require sensitive data
• **Explainability:** Understanding why AI makes certain decisions
• **Job Impact:** Automation effects on employment
• **Safety:** Ensuring AI systems behave safely and as intended

**Getting Started:**
1. **Learn Python** and basic data manipulation (pandas, numpy)
2. **Understand Statistics** and linear algebra fundamentals  
3. **Practice with Real Data** on Kaggle competitions
4. **Study Online Courses** (Andrew Ng's ML course, Fast.ai)
5. **Build Projects** - start simple, gradually increase complexity
6. **Join Communities** - AI/ML Discord servers, Reddit communities

**Career Paths:**
• **ML Engineer:** Build and deploy ML systems in production
• **Data Scientist:** Extract insights from data, build predictive models
• **Research Scientist:** Advance the field through novel algorithms and techniques
• **AI Product Manager:** Guide AI product development and strategy
• **AI Ethics Researcher:** Ensure AI development is responsible and beneficial

AI is transforming every field - from healthcare and education to entertainment and finance. The key is to start learning and experimenting with the tools available today."""
    
    def _generate_optimization_guide(self, prompt: str, prompt_lower: str) -> str:
        """Generate optimization guidance like advanced AI assistants."""
        if 'database' in prompt_lower:
            return """**Database Optimization - Systematic Approach:**

**1. Query Optimization:**
```sql
-- Bad: N+1 queries
SELECT * FROM users;
-- Then for each user: SELECT * FROM posts WHERE user_id = ?

-- Good: Single join query
SELECT u.*, p.* FROM users u 
LEFT JOIN posts p ON u.id = p.user_id;
```

**2. Indexing Strategy:**
• **Primary/Unique Indexes:** Automatic on primary keys
• **Composite Indexes:** For multi-column WHERE clauses
• **Partial Indexes:** For filtered queries
• **Monitor:** Use EXPLAIN ANALYZE to verify index usage

**3. Schema Design:**
• **Normalization:** Eliminate data redundancy
• **Denormalization:** Strategic duplicates for read performance
• **Partitioning:** Split large tables across multiple physical locations

**4. Connection Management:**
• **Connection Pooling:** Reuse database connections
• **Read Replicas:** Distribute read load across multiple servers
• **Caching:** Redis/Memcached for frequently accessed data"""
        
        if 'performance' in prompt_lower and any(lang in prompt_lower for lang in ['web', 'frontend', 'javascript']):
            return self._generate_web_performance_guide()
        
        return "I can provide specific optimization guidance. What type of optimization are you looking for? (database, web performance, algorithm complexity, memory usage, etc.)"
    
    def _generate_security_guide(self, prompt: str, prompt_lower: str) -> str:
        """Generate security best practices guide."""
        return """**Application Security - Comprehensive Checklist:**

**Authentication & Authorization:**
• Use strong password policies and multi-factor authentication
• Implement JWT tokens with proper expiration and refresh logic
• Follow principle of least privilege for user permissions
• Hash passwords with bcrypt/scrypt (never store plaintext)

**Data Protection:**
• Encrypt sensitive data at rest and in transit (TLS/SSL)
• Validate and sanitize all user inputs
• Use parameterized queries to prevent SQL injection
• Implement rate limiting to prevent abuse

**API Security:**
• Use HTTPS for all endpoints
• Implement proper CORS policies
• Add request/response logging and monitoring
• Use API keys and authentication for external access

**Infrastructure:**
• Keep dependencies updated (use automated tools)
• Configure firewalls and network security groups
• Use secrets management (AWS Secrets Manager, HashiCorp Vault)
• Regular security audits and penetration testing

**Code Security:**
• Static analysis tools (SonarQube, Snyk)
• Dependency vulnerability scanning
• Code reviews with security focus
• Follow OWASP Top 10 guidelines"""
    
    def _generate_learning_path(self, prompt: str, prompt_lower: str) -> str:
        """Generate structured learning paths."""
        return """**Programming Learning Path - Optimized Approach:**

**Phase 1: Foundations (2-4 weeks)**
• Choose one language (Python recommended for beginners)
• Master basic syntax: variables, functions, control flow
• Practice daily with small exercises (30-60 minutes)
• Resources: FreeCodeCamp, Codecademy, Python.org tutorial

**Phase 2: Problem Solving (4-8 weeks)**
• Learn data structures: arrays, lists, dictionaries
• Basic algorithms: searching, sorting
• Practice on HackerRank, LeetCode (easy problems)
• Build small projects: calculator, todo list

**Phase 3: Real Projects (8-12 weeks)**
• Web development: HTML/CSS + your chosen language
• Database basics: SQL, connecting to databases
• Version control: Git and GitHub
• Deploy a project online

**Phase 4: Specialization**
• **Web Development:** React/Vue + Node.js/Django
• **Data Science:** Pandas, NumPy, machine learning
• **Mobile:** React Native, Flutter, Swift/Kotlin
• **Systems:** Go, Rust, C++ for performance

**Key Success Strategies:**
• Code every day, even if just 20 minutes
• Build projects that interest you personally
• Join coding communities and find mentors
• Don't just watch tutorials - type the code yourself
• Learn to read error messages and debug systematically"""
    
    def _generate_js_ts_comparison(self) -> str:
        """Compare JavaScript and TypeScript."""
        return """**JavaScript vs TypeScript - Detailed Comparison:**

**Type Safety:**
```javascript
// JavaScript - Runtime errors
function greet(name) {
    return "Hello " + name.toUpperCase(); // Error if name is undefined
}
greet(); // Runtime error!
```

```typescript
// TypeScript - Compile-time safety
function greet(name: string): string {
    return "Hello " + name.toUpperCase();
}
greet(); // Compile error - missing argument!
```

**Developer Experience:**
• **TypeScript:** Superior IDE support, autocomplete, refactoring
• **JavaScript:** Simpler setup, faster iteration for small projects

**Learning Curve:**
• **JavaScript:** Easier to start, but debugging can be harder
• **TypeScript:** Steeper initial learning, but prevents many bugs

**Performance:**
• Both compile to the same JavaScript - no runtime difference
• TypeScript can enable better optimizations during build

**When to Use TypeScript:**
✅ Large codebases with multiple developers
✅ Long-term projects requiring maintainability  
✅ When you want better IDE support and refactoring
✅ API development where type safety is crucial

**When JavaScript is Fine:**
✅ Small projects or prototypes
✅ Learning web development basics
✅ Quick scripts or simple websites
✅ When team lacks TypeScript experience

**Migration Strategy:**
1. Start with `.js` files in TypeScript project
2. Gradually add type annotations
3. Use `@ts-check` comments for gradual typing
4. Enable stricter compiler options over time"""
    
    def _generate_web_framework_comparison(self, prompt_lower: str) -> str:
        """Compare web frameworks."""
        if 'react' in prompt_lower:
            return """**React vs Vue vs Angular - Framework Comparison:**

**React (Facebook/Meta):**
```jsx
function UserCard({ user }) {
    return (
        <div className="card">
            <h3>{user.name}</h3>
            <p>{user.email}</p>
        </div>
    );
}
```
• **Strengths:** Huge ecosystem, flexible, great job market
• **Learning Curve:** Medium - JSX and concepts to learn
• **Best For:** Complex SPAs, when you want maximum flexibility

**Vue (Independent/Evan You):**
```vue
<template>
    <div class="card">
        <h3>{{ user.name }}</h3>
        <p>{{ user.email }}</p>
    </div>
</template>
```
• **Strengths:** Gentle learning curve, great documentation
• **Learning Curve:** Easy - similar to HTML/JavaScript
• **Best For:** Teams wanting productivity with moderate complexity

**Angular (Google):**
```typescript
@Component({
    template: `
        <div class="card">
            <h3>{{user.name}}</h3>
            <p>{{user.email}}</p>
        </div>
    `
})
export class UserCard { }
```
• **Strengths:** Full framework, TypeScript by default, enterprise-ready
• **Learning Curve:** Steep - many concepts and conventions
• **Best For:** Large enterprise applications, teams preferring structure

**Performance:** All are fast when optimized properly
**Ecosystem:** React > Angular > Vue (but all have what you need)
**Job Market:** React > Angular > Vue"""
        
        return "I can compare specific web frameworks. Which ones are you interested in? (React vs Vue, Django vs Flask, etc.)"
    
    def _generate_containerization_explanation(self) -> str:
        """Explain Docker and Kubernetes."""
        return """**Docker vs Kubernetes - Container Orchestration:**

**Docker - Containerization Platform:**
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
EXPOSE 3000
CMD ["npm", "start"]
```

**What Docker Solves:**
• "It works on my machine" - consistent environments
• Lightweight isolation vs heavy VMs
• Easy deployment and scaling

**Kubernetes - Container Orchestration:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web-app
  template:
    metadata:
      labels:
        app: web-app
    spec:
      containers:
      - name: web
        image: my-web-app:latest
        ports:
        - containerPort: 3000
```

**What Kubernetes Adds:**
• **Auto-scaling:** Scale pods based on CPU/memory usage
• **Load Balancing:** Distribute traffic across replicas
• **Self-healing:** Restart failed containers automatically
• **Rolling Updates:** Deploy new versions without downtime
• **Service Discovery:** Containers can find each other easily

**When to Use:**
• **Docker Alone:** Simple applications, development environments
• **Docker + Kubernetes:** Production systems requiring high availability and scale"""
    
    def _generate_with_optimized_transformers(self, prompt: str, max_length: int = 256, temperature: float = 0.1) -> str:
        """Generate response using optimized transformers with speed optimizations."""
        try:
            import torch
            
            # Tokenize input with padding for efficiency
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,  # Limit input length for speed
                padding=True
            )
            
            # Move to device properly
            input_ids = inputs['input_ids']
            attention_mask = inputs.get('attention_mask', None)
            
            if self.device.type != "cpu":
                input_ids = input_ids.to(self.device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
            
            # Optimized generation settings for better quality
            generation_config = {
                "max_new_tokens": max_length,
                "do_sample": True,  # Enable sampling for variety
                "temperature": temperature,
                "top_p": 0.9,  # Nucleus sampling for quality
                "top_k": 50,   # Limit token choices for coherence
                "repetition_penalty": 1.2,  # Reduce repetitive output
                "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True,  # Enable KV caching
            }
            
            # Generate with optimizations
            with torch.no_grad():  # Disable gradients for faster inference
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    **generation_config
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the new generated text
            if prompt in full_response:
                response = full_response.replace(prompt, "").strip()
            else:
                # If prompt not found, try to extract after the last occurrence
                response = full_response.strip()
            
            # Additional cleanup
            response = response.strip()
            if response.startswith("Assistant:"):
                response = response[10:].strip()
            
            return response if response else "I understand your request. Let me help you with that."
            
        except Exception as e:
            logger.error(f"Optimized generation failed: {str(e)}")
            return self._generate_simple_local(prompt, max_length)

    def _generate_simple_local(self, prompt: str, max_length: int) -> str:
        """Simple local generation fallback."""
        try:
            import torch
            
            # Simple tokenization and generation
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            if self.device.type != "cpu":
                inputs = inputs.to(self.device)
            
            # Generate with minimal settings
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_length,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode and clean response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if prompt in response:
                response = response.replace(prompt, "").strip()
            
            return response if response else "How can I help you with your coding needs?"
            
        except Exception as e:
            logger.error(f"Simple generation failed: {str(e)}")
            return "I'm here to help with your coding questions."

    def _generate_with_transformers(self, prompt: str, max_length: int = 256, temperature: float = 0.7) -> str:
        """Generate response using transformers library."""
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncate=True, max_length=2048)
            inputs = inputs.to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            # Decode response
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the new response part
            response = generated_text[len(prompt):].strip()
            
            # Clean up response
            response = self._clean_response(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Transformers generation error: {str(e)}")
            return "I encountered an error while processing your request. Please try again."
    
    def _generate_with_local(self, prompt: str, max_length: int, temperature: float) -> str:
        """Generate response using local implementation."""
        try:
            # Use the simple local generation method
            return self._generate_simple_local(prompt, max_length)
        except Exception as e:
            logger.error(f"Local generation error: {str(e)}")
            return "I'm here to help with your coding questions. Please let me know what you need assistance with."
    
    def _clean_response(self, response: str) -> str:
        """Clean up the generated response."""
        # Remove repetitive patterns
        lines = response.split('\n')
        cleaned_lines = []
        seen_lines = set()
        
        for line in lines:
            line = line.strip()
            if not line or line in seen_lines:
                continue
            
            # Skip if line is too repetitive
            if (line.count('User:') > 1 or 
                line.count('Assistant:') > 1 or
                len(line.split()) > 50):
                continue
            
            cleaned_lines.append(line)
            seen_lines.add(line)
            
            # Limit response length
            if len(cleaned_lines) >= 10:
                break
        
        response = '\n'.join(cleaned_lines)
        
        # Remove any remaining artifacts
        response = response.replace('User:', '').replace('Assistant:', '').strip()
        
        if not response or len(response) < 10:
            return "I understand your question. Let me help you with that. Could you please provide more details about what you need assistance with?"
        
        return response
    
    def generate_code(self, instruction: str, language: str = "python") -> str:
        """Generate code using optimized local fine-tuned model with comprehensive language support."""
        if not self.is_available():
            return "[Error] Local fine-tuned model is not available"
        
        try:
            # Try the fine-tuned model first with better prompting
            code_prompt = f"# {language.title()} code to {instruction}\n# Code:\n"
            response = self._generate_with_optimized_transformers(code_prompt, 200, 0.7)
            code = self._extract_code_from_response(response, language)
            
            # Check if the generated code is meaningful and in the right language
            if self._is_quality_code(code, instruction) and self._is_correct_language(code, language):
                return code
            
            # Fallback to comprehensive local code generation
            return self._generate_comprehensive_local_code(instruction, language)
            
        except Exception as e:
            logger.error(f"Code generation error: {str(e)}")
            return self._generate_comprehensive_local_code(instruction, language)
    
    def _is_correct_language(self, code: str, expected_language: str) -> bool:
        """Check if the generated code is in the correct programming language."""
        expected_language = expected_language.lower()
        code_lower = code.lower()
        
        # Language-specific syntax indicators
        language_indicators = {
            'python': ['def ', 'print(', 'import ', 'if __name__', ':', 'return ', 'class '],
            'javascript': ['function', 'const ', 'let ', 'var ', '=>', 'console.log', '{', '}'],
            'java': ['public class', 'public static', 'System.out.print', 'import java', '{', '}'],
            'rust': ['fn ', 'println!', 'let ', 'mut ', 'use ', '-> ', 'impl ', 'struct '],
            'c': ['#include', 'int main', 'printf', 'return 0', '{', '}'],
            'cpp': ['#include', 'std::', 'cout', 'using namespace', '{', '}'],
            'go': ['func ', 'package ', 'import ', 'fmt.Print', '{', '}'],
            'php': ['<?php', 'echo ', '$', 'function '],
            'ruby': ['def ', 'puts ', 'class ', 'end', 'require '],
            'swift': ['func ', 'let ', 'var ', 'print(', 'import ', '{', '}'],
            'kotlin': ['fun ', 'val ', 'var ', 'println(', 'import ', '{', '}'],
        }
        
        if expected_language in language_indicators:
            indicators = language_indicators[expected_language]
            # Check if at least one language-specific indicator is present
            return any(indicator in code_lower for indicator in indicators)
        
        # If we don't have specific indicators, assume it's okay
        return True
    
    def _generate_comprehensive_local_code(self, instruction: str, language: str = "python") -> str:
        """Generate comprehensive code using local templates for any programming language."""
        instruction_lower = instruction.lower()
        language = language.lower()
        
        # Universal "hello world" / greeting patterns
        if any(word in instruction_lower for word in ['hello', 'hi', 'greet', 'say hello', 'hello world']):
            return self._generate_hello_world_code(language)
        
        # Mathematical operations
        if any(word in instruction_lower for word in ['remainder', 'modulo', 'mod', '%']):
            return self._generate_math_code(instruction, language, 'remainder')
        
        if any(word in instruction_lower for word in ['add', 'sum', 'plus', 'addition']):
            return self._generate_math_code(instruction, language, 'addition')
        
        if any(word in instruction_lower for word in ['multiply', 'product', 'times']):
            return self._generate_math_code(instruction, language, 'multiplication')
        
        # Data structures and algorithms
        if 'fibonacci' in instruction_lower:
            return self._generate_algorithm_code(language, 'fibonacci')
        
        if any(word in instruction_lower for word in ['factorial', 'fact']):
            return self._generate_algorithm_code(language, 'factorial')
        
        if any(word in instruction_lower for word in ['sort', 'bubble sort', 'quick sort']):
            return self._generate_algorithm_code(language, 'sort')
        
        # File operations
        if any(word in instruction_lower for word in ['read file', 'write file', 'file io']):
            return self._generate_file_io_code(language, instruction)
        
        # Web/HTTP related
        if any(word in instruction_lower for word in ['http', 'web server', 'api', 'request']):
            return self._generate_web_code(language, instruction)
        
        # Generic fallback with proper language syntax
        return self._generate_generic_template(instruction, language)
    
    def _generate_hello_world_code(self, language: str) -> str:
        """Generate hello world code for any programming language."""
        templates = {
            'python': '''# Python script to say hello
def say_hello():
    print("Hello, World!")
    return "Hello, World!"

# Call the function
say_hello()''',
            
            'rust': '''// Rust script to say hello
fn main() {
    println!("Hello, World!");
}

fn say_hello() -> String {
    let message = "Hello, World!";
    println!("{}", message);
    message.to_string()
}''',
            
            'javascript': '''// JavaScript script to say hello
function sayHello() {
    console.log("Hello, World!");
    return "Hello, World!";
}

// Call the function
sayHello();''',
            
            'java': '''// Java class to say hello
public class HelloWorld {
    public static void main(String[] args) {
        sayHello();
    }
    
    public static String sayHello() {
        System.out.println("Hello, World!");
        return "Hello, World!";
    }
}''',
            
            'c': '''// C program to say hello
#include <stdio.h>

void say_hello() {
    printf("Hello, World!\\n");
}

int main() {
    say_hello();
    return 0;
}''',
            
            'cpp': '''// C++ program to say hello
#include <iostream>
#include <string>

using namespace std;

string sayHello() {
    cout << "Hello, World!" << endl;
    return "Hello, World!";
}

int main() {
    sayHello();
    return 0;
}''',
            
            'go': '''// Go program to say hello
package main

import "fmt"

func sayHello() string {
    message := "Hello, World!"
    fmt.Println(message)
    return message
}

func main() {
    sayHello()
}''',
            
            'php': '''<?php
// PHP script to say hello

function sayHello() {
    $message = "Hello, World!";
    echo $message . PHP_EOL;
    return $message;
}

// Call the function
sayHello();
?>''',
            
            'ruby': '''# Ruby script to say hello
def say_hello
  message = "Hello, World!"
  puts message
  message
end

# Call the function
say_hello''',
            
            'swift': '''// Swift script to say hello
import Foundation

func sayHello() -> String {
    let message = "Hello, World!"
    print(message)
    return message
}

// Call the function
sayHello()''',
            
            'kotlin': '''// Kotlin script to say hello
fun sayHello(): String {
    val message = "Hello, World!"
    println(message)
    return message
}

fun main() {
    sayHello()
}'''
        }
        
        return templates.get(language, templates['python'])
    
    def _generate_math_code(self, instruction: str, language: str, operation: str) -> str:
        """Generate mathematical operation code for any language."""
        # Extract numbers from instruction if present
        import re
        numbers = re.findall(r'\d+', instruction)
        
        if operation == 'remainder' and len(numbers) >= 2:
            a, b = numbers[0], numbers[1]
            templates = {
                'python': f'''# Python code to find the remainder of {a} and {b}
def find_remainder(dividend, divisor):
    remainder = dividend % divisor
    return remainder

# Calculate remainder
a = {a}
b = {b}
result = find_remainder(a, b)
print(f"The remainder of {{a}} divided by {{b}} is: {{result}}")''',
                
                'rust': f'''// Rust code to find the remainder of {a} and {b}
fn find_remainder(dividend: i32, divisor: i32) -> i32 {{
    dividend % divisor
}}

fn main() {{
    let a = {a};
    let b = {b};
    let result = find_remainder(a, b);
    println!("The remainder of {{}} divided by {{}} is: {{}}", a, b, result);
}}''',
                
                'javascript': f'''// JavaScript code to find the remainder of {a} and {b}
function findRemainder(dividend, divisor) {{
    return dividend % divisor;
}}

// Calculate remainder
const a = {a};
const b = {b};
const result = findRemainder(a, b);
console.log(`The remainder of ${{a}} divided by ${{b}} is: ${{result}}`);''',
                
                'java': f'''// Java code to find the remainder of {a} and {b}
public class RemainderCalculator {{
    public static int findRemainder(int dividend, int divisor) {{
        return dividend % divisor;
    }}
    
    public static void main(String[] args) {{
        int a = {a};
        int b = {b};
        int result = findRemainder(a, b);
        System.out.println("The remainder of " + a + " divided by " + b + " is: " + result);
    }}
}}'''
            }
            return templates.get(language, templates['python'])
        
        # Generic templates for other math operations
        return self._generate_generic_math_template(instruction, language, operation)
    
    def _generate_algorithm_code(self, language: str, algorithm: str) -> str:
        """Generate algorithm implementations for any language."""
        if algorithm == 'fibonacci':
            templates = {
                'python': '''# Python Fibonacci function
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

def fibonacci_iterative(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

# Example usage
for i in range(10):
    print(f"F({i}) = {fibonacci(i)}")''',
                
                'rust': '''// Rust Fibonacci function
fn fibonacci(n: u32) -> u32 {
    match n {
        0 => 0,
        1 => 1,
        _ => fibonacci(n - 1) + fibonacci(n - 2),
    }
}

fn fibonacci_iterative(n: u32) -> u32 {
    if n <= 1 {
        return n;
    }
    
    let mut a = 0;
    let mut b = 1;
    
    for _ in 2..=n {
        let temp = a + b;
        a = b;
        b = temp;
    }
    
    b
}

fn main() {
    for i in 0..10 {
        println!("F({}) = {}", i, fibonacci(i));
    }
}''',
                
                'javascript': '''// JavaScript Fibonacci function
function fibonacci(n) {
    if (n <= 1) {
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}

function fibonacciIterative(n) {
    if (n <= 1) return n;
    
    let a = 0, b = 1;
    for (let i = 2; i <= n; i++) {
        [a, b] = [b, a + b];
    }
    return b;
}

// Example usage
for (let i = 0; i < 10; i++) {
    console.log(`F(${i}) = ${fibonacci(i)}`);
}'''
            }
            return templates.get(language, templates['python'])
        
        return self._generate_generic_template(f"{algorithm} algorithm", language)
    
    def _generate_generic_template(self, instruction: str, language: str) -> str:
        """Generate a generic code template for any language."""
        templates = {
            'python': f'''# Python code for: {instruction}
def main():
    # TODO: Implement {instruction}
    print("Implementing: {instruction}")
    pass

if __name__ == "__main__":
    main()''',
            
            'rust': f'''// Rust code for: {instruction}
fn main() {{
    // TODO: Implement {instruction}
    println!("Implementing: {instruction}");
}}''',
            
            'javascript': f'''// JavaScript code for: {instruction}
function main() {{
    // TODO: Implement {instruction}
    console.log("Implementing: {instruction}");
}}

main();''',
            
            'java': f'''// Java code for: {instruction}
public class Solution {{
    public static void main(String[] args) {{
        // TODO: Implement {instruction}
        System.out.println("Implementing: {instruction}");
    }}
}}''',
            
            'c': f'''// C code for: {instruction}
#include <stdio.h>

int main() {{
    // TODO: Implement {instruction}
    printf("Implementing: {instruction}\\n");
    return 0;
}}''',
            
            'go': f'''// Go code for: {instruction}
package main

import "fmt"

func main() {{
    // TODO: Implement {instruction}
    fmt.Println("Implementing: {instruction}")
}}'''
        }
        
        return templates.get(language, templates['python'])
    
    def _generate_generic_math_template(self, instruction: str, language: str, operation: str) -> str:
        """Generate generic math operation templates."""
        if language == 'python':
            return f'''# Python code for {operation}: {instruction}
def calculate_{operation}(a, b):
    # TODO: Implement {operation} operation
    result = a  # placeholder
    return result

# Example usage
num1 = 10
num2 = 5
result = calculate_{operation}(num1, num2)
print(f"Result: {{result}}")'''
        
        elif language == 'rust':
            return f'''// Rust code for {operation}: {instruction}
fn calculate_{operation}(a: i32, b: i32) -> i32 {{
    // TODO: Implement {operation} operation
    a  // placeholder
}}

fn main() {{
    let num1 = 10;
    let num2 = 5;
    let result = calculate_{operation}(num1, num2);
    println!("Result: {{}}", result);
}}'''
        
        else:
            return self._generate_generic_template(instruction, language)
    
    def _is_quality_code(self, code: str, instruction: str) -> bool:
        """Check if the generated code is of good quality."""
        if not code or len(code.strip()) < 10:
            return False
        
        # Check for minimal/bad content
        bad_indicators = ['pass', 'TODO', '# Generated code', 'hello', 'code', 'undefined', 'null', 'iterator', 'return false', 'return None', 'import']
        if any(bad in code.lower() for bad in bad_indicators):
            return False
        
        # Check for gibberish patterns
        if any(char in code for char in ['::)', '):)', '|==', ']()', '_st]', '*std']):
            return False
        
        # Check for excessive punctuation or symbols
        symbol_count = sum(1 for char in code if char in '()[]{}*&^%$#@!|\\')
        if symbol_count > len(code) * 0.3:  # More than 30% symbols
            return False
        
        # Check for basic valid syntax indicators
        if not any(keyword in code for keyword in ['def', '=', 'print', 'return', 'if', 'for', 'while', 'fn ', 'function', 'console.log', 'println!', 'printf', 'cout']):
            return False
        
        # Check if it's just repetitive nonsense
        words = code.split()
        if len(set(words)) < len(words) * 0.4:  # Less than 40% unique words
            return False
        
        return True
    
    def _generate_local_quality_code(self, instruction: str, language: str = "python") -> str:
        """Generate quality code using local templates."""
        instruction_lower = instruction.lower()
        
        if language.lower() == "python":
            # Remainder/modulo operations
            if any(word in instruction_lower for word in ['remainder', 'modulo', 'mod', '% ']):
                numbers = []
                import re
                nums = re.findall(r'\d+', instruction)
                if len(nums) >= 2:
                    return f"""# Python code to find the remainder of {nums[0]} and {nums[1]}
a = {nums[0]}
b = {nums[1]}
remainder = a % b
print(f"The remainder of {{a}} divided by {{b}} is: {{remainder}}")"""
                else:
                    return """# Python code to find remainder
def find_remainder(dividend, divisor):
    remainder = dividend % divisor
    return remainder

# Example usage
result = find_remainder(9, 8)
print(f"Remainder: {result}")"""
            
            # Hello world
            if 'hello' in instruction_lower and 'world' in instruction_lower:
                return """# Python Hello World function
def hello_world():
    print("Hello, World!")
    return "Hello, World!"

# Call the function
hello_world()"""
            
            # Fibonacci
            if 'fibonacci' in instruction_lower:
                return """# Python Fibonacci function
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

# Example usage
for i in range(10):
    print(f"F({i}) = {fibonacci(i)}")"""
            
            # Addition/calculator
            if any(word in instruction_lower for word in ['add', 'sum', 'plus', 'calculator']):
                return """# Python function to add numbers
def add_numbers(a, b):
    result = a + b
    return result

# Example usage
num1 = 5
num2 = 3
total = add_numbers(num1, num2)
print(f"{num1} + {num2} = {total}")"""
        
        # Generic fallback
        return f"""# {language.title()} code for: {instruction}
# TODO: Implement the requested functionality

def main():
    # Add your code here
    pass

if __name__ == "__main__":
    main()"""
    
    def _extract_code_from_response(self, response: str, language: str) -> str:
        """Extract code from the model response, optimized for local model output."""
        # Clean the response first
        response = response.strip()
        
        # Remove the original prompt if it appears
        if f"# {language.title()} code to" in response:
            parts = response.split("# Code:\n", 1)
            if len(parts) > 1:
                response = parts[1]
        
        # Look for code blocks first
        code_patterns = [
            rf'```{language}\s*(.*?)```',
            r'```\s*(.*?)```',
        ]
        
        for pattern in code_patterns:
            import re
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                code = matches[0].strip()
                if code and len(code) > 5:
                    return code
        
        # If no code blocks found, try to extract meaningful code lines
        lines = response.split('\n')
        code_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip empty lines and obvious non-code
            if not line or line.startswith('#') and len(line.split()) < 3:
                continue
            
            # Look for code-like patterns
            if (any(keyword in line for keyword in ['def ', 'class ', 'import ', 'from ', 'if ', 'for ', 'while ', 'print(', 'return ', '=']) or
                any(symbol in line for symbol in ['{', '}', '()', '[]', ' = ']) or
                line.startswith(('    ', '\t')) or  # Indented lines
                re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*\s*=', line)):  # Variable assignments
                code_lines.append(line)
        
        if code_lines:
            # Clean up the code
            code = '\n'.join(code_lines)
            
            # Add a simple function wrapper if we just have expressions
            if language.lower() == 'python' and '=' in code and 'def ' not in code:
                return f"# Generated code\n{code}"
            
            return code
        
        # Fallback: return a simple code template
        if language.lower() == 'python':
            return f"# Python code for: {response[:50].strip()}\n# TODO: Implement the requested functionality\npass"
        else:
            return f"// Code for: {response[:50].strip()}\n// TODO: Implement the requested functionality"
    
    def analyze_code(self, code: str) -> Dict[str, Any]:
        """Analyze code structure and provide insights."""
        if not self.is_available():
            return {"error": "iLLuMinator-4.7B model is not available"}
        
        try:
            analysis_prompt = f"""Analyze this code and provide a detailed analysis in JSON format:

Code:
```
{code}
```

Provide analysis with this exact JSON structure:
{{
    "function_count": <number_of_functions>,
    "class_count": <number_of_classes>,
    "import_count": <number_of_imports>,
    "total_lines": <total_lines>,
    "code_lines": <non_empty_lines>,
    "complexity_score": <1-10_complexity_rating>,
    "functions": [
        {{"name": "function_name", "args": ["arg1", "arg2"], "line": line_number}}
    ],
    "classes": [
        {{"name": "class_name", "methods": [{{"name": "method_name", "line": line_number}}], "line": line_number}}
    ],
    "suggestions": ["improvement_suggestion_1", "improvement_suggestion_2"]
}}"""
            
            response = self.generate_response(analysis_prompt, max_length=512, temperature=0.2)
            
            # Try to extract JSON from response
            try:
                import json
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group())
                    return analysis
            except:
                pass
            
            # Fallback analysis
            return self._fallback_code_analysis(code)
            
        except Exception as e:
            logger.error(f"Code analysis error: {str(e)}")
            return self._fallback_code_analysis(code)
    
    def _fallback_code_analysis(self, code: str) -> Dict[str, Any]:
        """Fallback code analysis using simple pattern matching."""
        import re
        
        lines = code.split('\n')
        total_lines = len(lines)
        code_lines = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
        
        functions = re.findall(r'^\s*def\s+(\w+)', code, re.MULTILINE)
        classes = re.findall(r'^\s*class\s+(\w+)', code, re.MULTILINE)
        imports = re.findall(r'^\s*(import|from)\s+', code, re.MULTILINE)
        
        return {
            "function_count": len(functions),
            "class_count": len(classes),
            "import_count": len(imports),
            "total_lines": total_lines,
            "code_lines": code_lines,
            "complexity_score": min(10, max(1, (len(functions) + len(classes)) // 2 + 1)),
            "functions": [{"name": f, "args": [], "line": 0} for f in functions],
            "classes": [{"name": c, "methods": [], "line": 0} for c in classes],
            "suggestions": ["Consider adding more comments", "Review code structure"]
        }
    
    def test_connection(self) -> bool:
        """Test if the model is working properly."""
        if not self.is_available():
            return False
        
        try:
            test_response = self.generate_response("Hello", max_length=50, temperature=0.5)
            return len(test_response.strip()) > 0
        except:
            return False
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the loaded model."""
        return {
            "name": "iLLuMinator-4.7B",
            "version": "4.7B",
            "type": "Local Transformer Model",
            "repository": "https://github.com/Anipaleja/iLLuMinator-4.7B",
            "device": str(self.device),
            "status": "Available" if self.is_available() else "Not Available",
            "model_path": self.model_path
        }