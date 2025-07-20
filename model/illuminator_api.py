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
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest"
        self.headers = {
            "Content-Type": "application/json",
        }
        self.conversation_history = []
        
        logger.info("iLLuMinator API client initialized successfully")
        
    def _get_hidden_api_key(self) -> str:
        """Get the hidden iLLuMinator API key."""
        # Get a key from: https://makersuite.google.com/app/apikey
        
        # Try environment variable first (recommended)
        api_key = os.environ.get('ILLUMINATOR_API_KEY') or os.environ.get('GEMINI_API_KEY')
        if api_key:
            return api_key
        
        # Fallback to hardcoded key (your actual Gemini API key)
        illuminator_key = "AIzaSyA_4XfgtZM7PjrhlzxrMyGvDGA6_so76Vs"  # Your actual Gemini API key
        
        return illuminator_key
    
    def _make_api_request(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
        """Make API request to iLLuMinator model."""
        try:
            url = f"{self.base_url}:generateContent?key={self._api_key}"
            
            payload = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": temperature,
                    "topK": 40,
                    "topP": 0.95,
                    "maxOutputTokens": max_tokens,
                }
            }
            
            response = requests.post(url, json=payload, headers=self.headers)
            
            if response.status_code == 200:
                result = response.json()
                if "candidates" in result and len(result["candidates"]) > 0:
                    content = result["candidates"][0]["content"]["parts"][0]["text"]
                    return content.strip()
                else:
                    return "Error: No response generated"
            else:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                return f"Error: API request failed with status {response.status_code}"
                
        except Exception as e:
            logger.error(f"Error making API request: {str(e)}")
            return f"Error: {str(e)}"
    
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
