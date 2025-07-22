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
from typing import Optional, Dict, List, Any
from pathlib import Path
import subprocess
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class iLLuMinatorAPI:
    """
    Direct API client for iLLuMinator-4.7B model from GitHub repository
    Provides local inference without external API dependencies
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the iLLuMinator-4.7B model."""
        self.model_path = model_path or self._find_model_path()
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conversation_history = []
        self.model_loaded = False
        
        try:
            self._ensure_dependencies()
            self._load_model()
            logger.info("✓ iLLuMinator-4.7B model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load iLLuMinator-4.7B model: {str(e)}")
            self.model_loaded = False
    
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
        """Generate conversational response using iLLuMinator-4.7B model."""
        if not self.is_available():
            return "I apologize, but the iLLuMinator-4.7B model is not currently available."
        
        try:
            # Create system context for iLLuMinator
            system_context = """You are iLLuMinator-4.7B, an advanced AI coding assistant created by Anish Paleja. You are knowledgeable, helpful, and specialize in programming and software development. You provide clear, concise answers and can help with coding problems, explanations, and technical questions.

Key capabilities:
- Code generation in multiple programming languages
- Code analysis and debugging
- Technical explanations and tutorials
- Best practices and optimization suggestions
- Project structure and architecture advice

Always provide helpful, accurate, and practical responses."""
            
            # Format the conversation
            conversation = f"{system_context}\n\nUser: {prompt}\n\nAssistant:"
            
            # Generate response with transformers
            if hasattr(self.model, 'generate') and self.tokenizer:
                return self._generate_with_transformers(conversation, max_length, temperature)
            else:
                # Use local implementation
                return self._generate_with_local(conversation, max_length, temperature)
            
        except Exception as e:
            logger.error(f"Response generation error: {str(e)}")
            return f"I apologize, but I encountered an error while generating a response: {str(e)}"
    
    def _generate_with_transformers(self, prompt: str, max_length: int, temperature: float) -> str:
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
            if hasattr(self.model, 'generate_response'):
                return self.model.generate_response(prompt, max_length, temperature)
            elif hasattr(self.model, 'generate'):
                return self.model.generate(prompt, max_length, temperature)
            else:
                return "I understand your request. How can I help you with your coding needs?"
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
        """Generate code using iLLuMinator-4.7B model."""
        if not self.is_available():
            return "[Error] iLLuMinator-4.7B model is not available"
        
        try:
            # Create code generation prompt
            code_prompt = f"""You are an expert {language} programmer. Generate clean, professional, and well-commented code based on the following instruction:

Instruction: {instruction}
Language: {language}

Requirements:
- Write clean, readable code
- Include appropriate comments
- Follow best practices for {language}
- Make the code production-ready
- Include error handling where appropriate

Code:
```{language}"""
            
            # Generate code
            response = self.generate_response(code_prompt, max_length=512, temperature=0.3)
            
            # Extract code from response
            code = self._extract_code_from_response(response, language)
            
            return code
            
        except Exception as e:
            logger.error(f"Code generation error: {str(e)}")
            return f"[Error] Failed to generate code: {str(e)}"
    
    def _extract_code_from_response(self, response: str, language: str) -> str:
        """Extract code from the model response."""
        # Look for code blocks
        code_patterns = [
            rf'```{language}\s*(.*?)```',
            r'```\s*(.*?)```',
            rf'{language}\s*(.*?)(?=\n\n|$)',
        ]
        
        for pattern in code_patterns:
            import re
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                code = matches[0].strip()
                if code and len(code) > 10:
                    return code
        
        # If no code block found, try to extract meaningful code-like content
        lines = response.split('\n')
        code_lines = []
        
        for line in lines:
            # Skip commentary lines that don't look like code
            if (any(keyword in line.lower() for keyword in ['def ', 'class ', 'import ', 'function', 'var ', 'let ', 'const ']) or
                any(symbol in line for symbol in ['{', '}', ';', '()', '[]', '=']) or
                line.strip().startswith('#') or
                line.strip().startswith('//')):
                code_lines.append(line)
        
        if code_lines:
            return '\n'.join(code_lines)
        
        # Fallback: return cleaned response
        return response.strip()
    
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