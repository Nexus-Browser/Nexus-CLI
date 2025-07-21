"""
iLLuMinator API Client - Lightweight API-based model interface
Connects to the iLLuMinator-4.7B model via API for efficient inference
"""

import os
import json
import time
import logging
from typing import Optional, Dict, List, Any
import requests
from pathlib import Path

# Import iLLuMinator configuration
try:
    from illuminator_config import get_illuminator_config, get_model_manager
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class iLLuMinatorAPI:
    """
    Lightweight API client for iLLuMinator-4.7B model
    Provides code generation and conversation capabilities without local GPU requirements
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the iLLuMinator API client."""
        self._api_key = api_key or self._get_hidden_api_key()
        
        # Configure API endpoint based on key type
        if self._api_key.startswith('sk-ant-'):
            # Anthropic Claude
            self.base_url = "https://api.anthropic.com/v1/messages"
            self.api_type = "anthropic"
        elif self._api_key.startswith('gsk_'):
            # Groq
            self.base_url = "https://api.groq.com/openai/v1/chat/completions"
            self.api_type = "groq"
        elif self._api_key.startswith('AIza'):
            # Google Gemini
            self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
            self.api_type = "gemini"
        elif self._api_key.startswith('hf_'):
            # Hugging Face Inference API
            self.base_url = "https://api-inference.huggingface.co/models"
            self.api_type = "huggingface"
            # Default to a good open model (you can change this)
            self.hf_model = "microsoft/DialoGPT-large"
        elif self._api_key.startswith('together_'):
            # Together AI - good free tier
            self.base_url = "https://api.together.xyz/v1/chat/completions"
            self.api_type = "together"
        elif self._api_key.startswith('pplx-'):
            # Perplexity AI - free tier available
            self.base_url = "https://api.perplexity.ai/chat/completions"
            self.api_type = "perplexity"
        elif len(self._api_key) == 40 and not self._api_key.startswith(('sk-', 'hf_', 'AIza', 'gsk_', 'together_', 'pplx-', 'sk-ant-')):
            # Cohere API - generous free tier
            self.base_url = "https://api.cohere.ai/v1/chat"
            self.api_type = "cohere"
        elif self._api_key.startswith('sk-proj-') or self._api_key.startswith('sk-'):
            # OpenAI
            self.base_url = "https://api.openai.com/v1/chat/completions"
            self.api_type = "openai"
        elif self._api_key == "local":
            # Local/offline mode
            self.base_url = "local"
            self.api_type = "local"
        else:
            # Default to OpenAI format for other APIs
            self.base_url = "https://api.openai.com/v1/chat/completions" 
            self.api_type = "openai"
            
        self.headers = {
            "Content-Type": "application/json",
        }
        self.conversation_history = []
        
        logger.info("iLLuMinator API client initialized successfully")
        
    def _get_hidden_api_key(self) -> str:
        """Get the iLLuMinator API key from environment variables."""
        # Try environment variables first (recommended for security)
        api_key = (os.environ.get('ILLUMINATOR_API_KEY') or 
                  os.environ.get('OPENAI_API_KEY') or 
                  os.environ.get('ANTHROPIC_API_KEY') or
                  os.environ.get('GEMINI_API_KEY') or
                  os.environ.get('COHERE_API_KEY') or
                  os.environ.get('GROQ_API_KEY') or
                  os.environ.get('HF_API_KEY') or
                  os.environ.get('TOGETHER_API_KEY'))
        
        if api_key:
            return api_key
        
        # If no API key found in environment, use local mode
        print("\n⚠️  No API key found in environment variables.")
        print("To use iLLuMinator-4.7B with full AI capabilities, please set one of:")
        print("  export ILLUMINATOR_API_KEY='your_api_key_here'")
        print("  export OPENAI_API_KEY='your_openai_key'")
        print("  export ANTHROPIC_API_KEY='your_claude_key'")
        print("  export COHERE_API_KEY='your_cohere_key'")
        print("\nRunning in local mode for now...\n")
        
        return "local"
    
    def _make_api_request(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
        """Make API request to configured AI service with fallback handling."""
        try:
            if self.api_type == "anthropic":
                result = self._make_anthropic_request(prompt, max_tokens, temperature)
            elif self.api_type == "groq":
                result = self._make_groq_request(prompt, max_tokens, temperature)
            elif self.api_type == "gemini":
                result = self._make_gemini_request(prompt, max_tokens, temperature)
            elif self.api_type == "huggingface":
                result = self._make_huggingface_request(prompt, max_tokens, temperature)
            elif self.api_type == "together":
                result = self._make_together_request(prompt, max_tokens, temperature)
            elif self.api_type == "perplexity":
                result = self._make_perplexity_request(prompt, max_tokens, temperature)
            elif self.api_type == "cohere":
                result = self._make_cohere_request(prompt, max_tokens, temperature)
            elif self.api_type == "local":
                result = self._make_local_request(prompt, max_tokens, temperature)
            else:
                result = self._make_openai_request(prompt, max_tokens, temperature)
            
            # If we get a quota error, provide helpful information
            if "quota" in result.lower() or "429" in result:
                return self._handle_quota_exhausted()
            
            return result
            
        except Exception as e:
            logger.error(f"Error making API request: {str(e)}")
            return self._handle_api_error(str(e))
    
    def _handle_quota_exhausted(self) -> str:
        """Handle quota exhaustion with helpful suggestions."""
        return """🚫 API Quota Exhausted

Current API has reached its daily limit. Here are your options:

1. **Wait and retry**: Quotas reset daily (Gemini: 50 requests/day, Groq: varies)

2. **Get a new API key**:
   • Anthropic Claude: https://console.anthropic.com (free tier available)
   • OpenAI: https://platform.openai.com (pay-per-use)
   • Together AI: https://api.together.ai (good free tier)

3. **Use local mode**: The system can work with basic responses while APIs recover

To switch API keys, modify the `_get_hidden_api_key()` method in `model/illuminator_api.py`"""

    def _handle_api_error(self, error: str) -> str:
        """Handle general API errors with helpful information."""
        return f"""⚠️ API Connection Issue

Error: {error}

**Troubleshooting Steps:**
1. Check your internet connection
2. Verify the API key is valid and not expired
3. Check if the API service is experiencing downtime
4. Try switching to a different API provider

**Alternative APIs to try:**
• Gemini: Fast and free (50 requests/day)
• Groq: Very fast responses (limited quota)
• Anthropic: High quality responses
• OpenAI: Industry standard (requires payment)

System will continue working with basic functionality."""
    
    def _make_openai_request(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
        """Make request to OpenAI-compatible API."""
        try:
            # OpenAI Chat Completions format
            payload = {
                "model": "gpt-3.5-turbo",  # You could also try "gpt-4o-mini" for cheaper rates
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            # Add Authorization header for OpenAI
            headers = {
                **self.headers,
                "Authorization": f"Bearer {self._api_key}"
            }
            
            response = requests.post(self.base_url, json=payload, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]
                    return content.strip()
                else:
                    return "Error: No response generated"
            else:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                return f"Error: API request failed with status {response.status_code}"
                
        except Exception as e:
            logger.error(f"Error making OpenAI request: {str(e)}")
            return f"Error: {str(e)}"
    
    def _make_anthropic_request(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
        """Make request to Anthropic Claude API."""
        try:
            payload = {
                "model": "claude-3-haiku-20240307",  # Fast and cheap model
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
            
            headers = {
                **self.headers,
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01"
            }
            
            response = requests.post(self.base_url, json=payload, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                if "content" in result and len(result["content"]) > 0:
                    return result["content"][0]["text"].strip()
                else:
                    return "Error: No response generated"
            else:
                logger.error(f"Anthropic request failed: {response.status_code} - {response.text}")
                return f"Error: API request failed with status {response.status_code}"
                
        except Exception as e:
            logger.error(f"Error making Anthropic request: {str(e)}")
            return f"Error: {str(e)}"
    
    def _make_groq_request(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
        """Make request to Groq API."""
        try:
            # Groq uses OpenAI-compatible format
            payload = {
                "model": "llama3-8b-8192",  # Current Llama 3 model - fast and good quality
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            headers = {
                **self.headers,
                "Authorization": f"Bearer {self._api_key}"
            }
            
            response = requests.post(self.base_url, json=payload, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]
                    return content.strip()
                else:
                    return "Error: No response generated"
            else:
                logger.error(f"Groq request failed: {response.status_code} - {response.text}")
                return f"Error: API request failed with status {response.status_code}"
                
        except Exception as e:
            logger.error(f"Error making Groq request: {str(e)}")
            return f"Error: {str(e)}"
    
    def _make_gemini_request(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
        """Make request to Google Gemini API."""
        try:
            # Gemini API format
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt}
                        ]
                    }
                ],
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature": temperature
                }
            }
            
            # Gemini uses API key as query parameter
            url = f"{self.base_url}?key={self._api_key}"
            
            response = requests.post(url, json=payload, headers=self.headers)
            
            if response.status_code == 200:
                result = response.json()
                if "candidates" in result and len(result["candidates"]) > 0:
                    content = result["candidates"][0]["content"]["parts"][0]["text"]
                    return content.strip()
                else:
                    return "Error: No response generated"
            else:
                logger.error(f"Gemini request failed: {response.status_code} - {response.text}")
                return f"Error: API request failed with status {response.status_code}"
                
        except Exception as e:
            logger.error(f"Error making Gemini request: {str(e)}")
            return f"Error: {str(e)}"
    
    def _make_huggingface_request(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
        """Make request to Hugging Face Inference API."""
        try:
            # Hugging Face Inference API format
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "return_full_text": False
                }
            }
            
            # Use the configured model endpoint
            url = f"{self.base_url}/{self.hf_model}"
            
            headers = {
                **self.headers,
                "Authorization": f"Bearer {self._api_key}"
            }
            
            response = requests.post(url, json=payload, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    if "generated_text" in result[0]:
                        return result[0]["generated_text"].strip()
                    elif "text" in result[0]:
                        return result[0]["text"].strip()
                return "Error: No response generated"
            else:
                logger.error(f"Hugging Face request failed: {response.status_code} - {response.text}")
                if response.status_code == 503:
                    return "⏳ Model is loading on Hugging Face. Please try again in a few seconds."
                return f"Error: API request failed with status {response.status_code}"
                
        except Exception as e:
            logger.error(f"Error making Hugging Face request: {str(e)}")
            return f"Error: {str(e)}"
    
    def _make_together_request(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
        """Make request to Together AI API."""
        try:
            # Together AI uses OpenAI-compatible format
            payload = {
                "model": "meta-llama/Llama-2-7b-chat-hf",  # Free model
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            headers = {
                **self.headers,
                "Authorization": f"Bearer {self._api_key}"
            }
            
            response = requests.post(self.base_url, json=payload, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]
                    return content.strip()
                else:
                    return "Error: No response generated"
            else:
                logger.error(f"Together AI request failed: {response.status_code} - {response.text}")
                return f"Error: API request failed with status {response.status_code}"
                
        except Exception as e:
            logger.error(f"Error making Together AI request: {str(e)}")
            return f"Error: {str(e)}"
    
    def _make_perplexity_request(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
        """Make request to Perplexity AI API."""
        try:
            # Perplexity AI uses OpenAI-compatible format
            payload = {
                "model": "llama-3.1-sonar-small-128k-online",  # Free model with web access
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            headers = {
                **self.headers,
                "Authorization": f"Bearer {self._api_key}"
            }
            
            response = requests.post(self.base_url, json=payload, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]
                    return content.strip()
                else:
                    return "Error: No response generated"
            else:
                logger.error(f"Perplexity request failed: {response.status_code} - {response.text}")
                return f"Error: API request failed with status {response.status_code}"
                
        except Exception as e:
            logger.error(f"Error making Perplexity request: {str(e)}")
            return f"Error: {str(e)}"
    
    def _make_cohere_request(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
        """Make request to Cohere API."""
        try:
            # Cohere Chat API format
            payload = {
                "model": "command-light",  # Free model, great for most tasks
                "message": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False
            }
            
            headers = {
                **self.headers,
                "Authorization": f"Bearer {self._api_key}"
            }
            
            response = requests.post(self.base_url, json=payload, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                if "text" in result:
                    return result["text"].strip()
                else:
                    return "Error: No response generated"
            else:
                logger.error(f"Cohere request failed: {response.status_code} - {response.text}")
                return f"Error: API request failed with status {response.status_code}"
                
        except Exception as e:
            logger.error(f"Error making Cohere request: {str(e)}")
            return f"Error: {str(e)}"
    
    def _make_local_request(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
        """Make local/offline request with basic AI-like responses."""
        try:
            # Simple pattern matching for local mode
            prompt_lower = prompt.lower()
            
            if any(word in prompt_lower for word in ['hello', 'hi', 'hey']):
                return "Hello! I'm running in local mode. To get better AI responses, please set up an API key from Together AI, Hugging Face, or Perplexity AI."
            
            elif any(word in prompt_lower for word in ['code', 'function', 'python', 'javascript']):
                return """I'd love to help with coding! However, I'm currently running in local mode with limited capabilities. For advanced code generation, please set up one of these free API keys:

🔥 **Together AI** (Recommended): https://api.together.ai
   - $25 free credit + ongoing free tier
   - Access to Llama 2, Code Llama, and more

🤗 **Hugging Face**: https://huggingface.co/settings/tokens  
   - Completely free
   - No credit card required

🔍 **Perplexity AI**: https://www.perplexity.ai/settings/api
   - Free tier available
   - Web-connected models

Then update the API key in `model/illuminator_api.py`"""

            elif any(word in prompt_lower for word in ['help', 'what', 'how']):
                return """I'm running in local mode with basic responses. Here's what you can do:

1. **Get a free API key** from one of these providers:
   - Together AI: https://api.together.ai (Best option - $25 free credit)
   - Hugging Face: https://huggingface.co/settings/tokens (Completely free)
   - Perplexity AI: https://www.perplexity.ai/settings/api (Free tier)

2. **Update the code**: Replace 'local' with your new API key in the `_get_hidden_api_key()` method

3. **Restart the system** and enjoy full AI capabilities!

Current local capabilities are very limited - get an API key for the full experience."""

            else:
                return f"""I received your message: "{prompt[:100]}..."

I'm currently running in **local mode** with very limited AI capabilities. This is because all the API keys have been exhausted.

**Quick Setup for Better AI:**
1. Visit https://api.together.ai (recommended)
2. Sign up for free ($25 credit included)
3. Get your API key (starts with 'together_')
4. Replace 'local' with your key in the code
5. Restart and enjoy full AI features!

Alternatively, try Hugging Face (completely free): https://huggingface.co/settings/tokens"""
                
        except Exception as e:
            return f"Local mode error: {str(e)}"
    
    def generate_code(self, instruction: str, language: str = "python") -> str:
        """Generate code using iLLuMinator-4.7B model."""
        # Craft a specialized prompt for code generation
        code_prompt = f"""You are iLLuMinator-4.7B, an advanced code generation AI. Generate clean, efficient {language} code for the following task:

Task: {instruction}

Requirements:
- Write clean, readable code
- Include appropriate comments
- Follow best practices for {language}
- Return only the code, no explanations

{language.title()} Code:"""

        try:
            response = self._make_api_request(code_prompt, max_tokens=512, temperature=0.3)
            
            # Clean up the response to extract just the code
            if response.startswith("Error:"):
                return response
            
            # Remove any markdown formatting if present
            if "```" in response:
                # Extract code from markdown blocks
                code_blocks = response.split("```")
                for block in code_blocks:
                    if block.strip() and not block.strip().startswith(language):
                        if language.lower() in block.lower()[:20] or "def " in block or "function" in block or "class " in block:
                            return block.strip()
                # If no suitable block found, return the first code block
                for i, block in enumerate(code_blocks):
                    if i % 2 == 1:  # Odd indices are usually code blocks
                        return block.strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Code generation error: {str(e)}")
            return f"[Error] Code generation failed: {str(e)}"
    
    def generate_response(self, prompt: str, max_length: int = 256, temperature: float = 0.7) -> str:
        """Generate conversational response using iLLuMinator-4.7B model."""
        # Add context about being iLLuMinator
        system_context = """You are iLLuMinator-4.7B, an advanced AI coding assistant created by Anish Paleja. You are knowledgeable, helpful, and specialize in programming and software development. You provide clear, concise answers and can help with coding problems, explanations, and technical questions.

User: """
        
        full_prompt = system_context + prompt
        
        try:
            response = self._make_api_request(full_prompt, max_tokens=max_length, temperature=temperature)
            
            # Add to conversation history
            self.conversation_history.append({
                "role": "user", 
                "content": prompt
            })
            self.conversation_history.append({
                "role": "assistant", 
                "content": response
            })
            
            # Keep conversation history manageable
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-10:]
            
            return response
            
        except Exception as e:
            logger.error(f"Response generation error: {str(e)}")
            return f"I apologize, but I'm having trouble generating a response right now. Error: {str(e)}"
    
    def analyze_code(self, code: str) -> Dict[str, Any]:
        """Analyze code structure and provide insights."""
        analysis_prompt = f"""Analyze this code and provide a JSON response with the following structure:
{{
    "function_count": number_of_functions,
    "class_count": number_of_classes, 
    "import_count": number_of_imports,
    "total_lines": total_lines,
    "code_lines": non_empty_lines,
    "functions": [
        {{"name": "function_name", "args": ["arg1", "arg2"], "line": line_number}}
    ],
    "classes": [
        {{"name": "class_name", "methods": [{{"name": "method_name", "line": line_number}}], "line": line_number}}
    ],
    "complexity": "low|medium|high",
    "suggestions": ["suggestion1", "suggestion2"]
}}

Code to analyze:
```python
{code}
```

Return only the JSON response:"""

        try:
            response = self._make_api_request(analysis_prompt, max_tokens=512, temperature=0.2)
            
            # Try to parse as JSON
            try:
                analysis = json.loads(response)
                return analysis
            except json.JSONDecodeError:
                # Fallback to basic analysis if JSON parsing fails
                return self._basic_code_analysis(code)
                
        except Exception as e:
            logger.error(f"Code analysis error: {str(e)}")
            return self._basic_code_analysis(code)
    
    def _basic_code_analysis(self, code: str) -> Dict[str, Any]:
        """Fallback basic code analysis."""
        lines = code.split('\n')
        total_lines = len(lines)
        code_lines = len([line for line in lines if line.strip()])
        
        # Basic pattern matching
        import re
        functions = len(re.findall(r'^\s*def\s+(\w+)', code, re.MULTILINE))
        classes = len(re.findall(r'^\s*class\s+(\w+)', code, re.MULTILINE))
        imports = len(re.findall(r'^\s*(import|from)\s+', code, re.MULTILINE))
        
        return {
            "function_count": functions,
            "class_count": classes,
            "import_count": imports,
            "total_lines": total_lines,
            "code_lines": code_lines,
            "functions": [],
            "classes": [],
            "complexity": "medium",
            "suggestions": ["Consider adding more documentation", "Review function complexity"]
        }
    
    def is_available(self) -> bool:
        """Check if the iLLuMinator API is available."""
        try:
            test_response = self._make_api_request("Hello", max_tokens=10, temperature=0.1)
            return not test_response.startswith("Error:")
        except:
            return False
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the iLLuMinator model."""
        return {
            "name": "iLLuMinator-4.7B",
            "version": "1.0",
            "author": "Anish Paleja", 
            "repository": "https://github.com/Anipaleja/iLLuMinator-4.7B",
            "description": "Advanced 4.7B parameter model for code generation and assistance",
            "capabilities": "Code generation, conversation, code analysis",
            "api_status": "connected" if self.is_available() else "disconnected"
        }
