"""
Nexus Model - Direct integration with iLLuMinator-4.7B
Provides advanced code generation and analysis using the actual iLLuMinator-4.7B model
Repository: https://github.com/Anipaleja/iLLuMinator-4.7B
"""

import logging
import re
import ast
import json
import os
import time
from typing import Optional, Dict, List, Any
from pathlib import Path

# Import the actual iLLuMinator-4.7B model
from .illuminator_api import iLLuMinatorAPI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NexusModel:
    """
    Nexus model using actual iLLuMinator-4.7B transformer model
    Provides advanced code generation and conversation using local processing
    """
    
    def __init__(self, model_path: Optional[str] = None, fast_mode: bool = True):
        """Initialize the Nexus model with local fine-tuned model in fast mode."""
        logger.info("Initializing Nexus model with local fine-tuned model...")
        if fast_mode:
            logger.info("🚀 Fast mode enabled - using optimized local inference")
        
        try:
            # Initialize the local fine-tuned model with fast mode
            logger.info("Loading local fine-tuned transformer model...")
            self.api = iLLuMinatorAPI(model_path or "./model/nexus_model/", fast_mode=fast_mode)
            self.fast_mode = fast_mode
            
            if self.api.is_available():
                self.model_available = True
                logger.info("✓ Local fine-tuned model loaded successfully!")
            else:
                self.model_available = False
                logger.warning("Local fine-tuned model failed to load, using basic mode")
            
            # Initialize code analyzer
            self.code_analyzer = CodeAnalyzer()
            
        except Exception as e:
            logger.error(f"Error initializing local fine-tuned model: {str(e)}")
            logger.warning("Falling back to basic mode without AI capabilities")
            self.api = None
            self.model_available = False
            self.code_analyzer = CodeAnalyzer()
            self.fast_mode = False
    
    def generate_code(self, instruction: str, language: str = "python") -> str:
        """Generate code using iLLuMinator-4.7B model."""
        if not self.model_available or not self.api:
            return "[Error] iLLuMinator model not available"
        
        try:
            logger.info(f"Generating {language} code for: {instruction[:50]}...")
            response = self.api.generate_code(instruction, language)
            logger.info("✓ Code generation completed")
            return response
            
        except Exception as e:
            logger.error(f"Code generation error: {str(e)}")
            return f"[Error] Code generation failed: {str(e)}"
    
    def generate_response(self, prompt: str, max_length: int = 256, temperature: float = 0.7) -> str:
        """Generate conversational response using iLLuMinator-4.7B model."""
        if not self.model_available or not self.api:
            return "I apologize, but the iLLuMinator model is not currently available. Please check your connection."
        
        try:
            logger.info(f"Generating response for: {prompt[:30]}...")
            response = self.api.generate_response(prompt, max_length, temperature)
            logger.info("✓ Response generation completed")
            return response
            
        except Exception as e:
            logger.error(f"Response generation error: {str(e)}")
            return f"I'm having trouble generating a response right now. Error: {str(e)}"
    
    def analyze_code(self, code: str) -> Dict[str, Any]:
        """Analyze code using iLLuMinator-4.7B model or fallback analyzer."""
        if self.model_available and self.api:
            try:
                logger.info("Analyzing code with iLLuMinator...")
                analysis = self.api.analyze_code(code)
                logger.info("✓ Code analysis completed")
                return analysis
            except Exception as e:
                logger.warning(f"API analysis failed, using fallback: {str(e)}")
        
        # Fallback to basic analysis
        return self.code_analyzer.analyze(code)
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the current model."""
        if self.api:
            return self.api.get_model_info()
        else:
            return {
                "name": "Nexus (Offline)",
                "status": "disconnected",
                "description": "Model not available"
            }
    
    def is_available(self) -> bool:
        """Check if the model is available and working."""
        return self.model_available and self.api is not None
    
    def test_connection(self) -> bool:
        """Test the connection to iLLuMinator API."""
        if self.api:
            return self.api.is_available()
        return False


class CodeAnalyzer:
    """Intelligent code analyzer using AST parsing as fallback."""
    
    def analyze(self, code: str) -> Dict[str, Any]:
        """Analyze code using AST parsing."""
        try:
            tree = ast.parse(code)
            visitor = CodeASTVisitor()
            visitor.visit(tree)
            return visitor.get_results()
        except SyntaxError as e:
            return {"error": f"Syntax error: {e}"}
        except Exception as e:
            return {"error": f"Analysis error: {e}"}


class CodeASTVisitor(ast.NodeVisitor):
    """AST visitor for code analysis."""
    
    def __init__(self):
        self.functions = []
        self.classes = []
        self.imports = []
        self.variables = []
        self.total_lines = 0
    
    def visit_FunctionDef(self, node):
        """Visit function definitions."""
        self.functions.append({
            "name": node.name,
            "args": [arg.arg for arg in node.args.args],
            "line": node.lineno,
            "docstring": ast.get_docstring(node) or ""
        })
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        """Visit class definitions."""
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                methods.append({
                    "name": item.name,
                    "args": [arg.arg for arg in item.args.args],
                    "line": item.lineno
                })
        
        self.classes.append({
            "name": node.name,
            "methods": methods,
            "line": node.lineno,
            "docstring": ast.get_docstring(node) or ""
        })
        self.generic_visit(node)
    
    def visit_Import(self, node):
        """Visit import statements."""
        for alias in node.names:
            self.imports.append({
                "name": alias.name,
                "alias": alias.asname,
                "line": node.lineno
            })
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        """Visit from-import statements."""
        for alias in node.names:
            self.imports.append({
                "name": f"{node.module}.{alias.name}" if node.module else alias.name,
                "alias": alias.asname,
                "line": node.lineno
            })
        self.generic_visit(node)
    
    def visit_Assign(self, node):
        """Visit variable assignments."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.variables.append({
                    "name": target.id,
                    "line": node.lineno
                })
        self.generic_visit(node)
    
    def get_results(self) -> Dict[str, Any]:
        """Get analysis results."""
        return {
            "function_count": len(self.functions),
            "class_count": len(self.classes),
            "import_count": len(self.imports),
            "variable_count": len(self.variables),
            "functions": self.functions,
            "classes": self.classes,
            "imports": self.imports,
            "variables": self.variables[:10],  # Limit to first 10 variables
            "complexity": self._estimate_complexity(),
            "suggestions": self._generate_suggestions()
        }
    
    def _estimate_complexity(self) -> str:
        """Estimate code complexity."""
        total_items = len(self.functions) + len(self.classes)
        
        if total_items <= 5:
            return "low"
        elif total_items <= 15:
            return "medium"
        else:
            return "high"
    
    def _generate_suggestions(self) -> List[str]:
        """Generate code improvement suggestions."""
        suggestions = []
        
        # Check for functions without docstrings
        functions_without_docs = [f for f in self.functions if not f["docstring"]]
        if functions_without_docs:
            suggestions.append(f"Consider adding docstrings to {len(functions_without_docs)} functions")
        
        # Check for classes without docstrings
        classes_without_docs = [c for c in self.classes if not c["docstring"]]
        if classes_without_docs:
            suggestions.append(f"Consider adding docstrings to {len(classes_without_docs)} classes")
        
        # Check for complex functions (many arguments)
        complex_functions = [f for f in self.functions if len(f["args"]) > 5]
        if complex_functions:
            suggestions.append(f"Consider simplifying functions with many parameters: {', '.join([f['name'] for f in complex_functions])}")
        
        if not suggestions:
            suggestions.append("Code structure looks good!")
        
        return suggestions
