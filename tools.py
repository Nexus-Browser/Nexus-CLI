

import os
import ast
import re
import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from pygments import highlight
from pygments.lexers import get_lexer_by_name, TextLexer
from pygments.formatters import TerminalFormatter
import logging

class FileTools:
    """Tools for file operations and code analysis."""
    
    @staticmethod
    def read_file(file_path: str) -> str:
        """Read a file and return its contents."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logging.error(f"Error reading file {file_path}: {e}")
            return f"Error reading file: {str(e)}"
    
    @staticmethod
    def write_file(file_path: str, content: str) -> bool:
        """Write content to a file."""
        try:
            dir_name = os.path.dirname(file_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            logging.error(f"Error writing file {file_path}: {e}")
            return False
    
    @staticmethod
    def list_files(directory: str = ".", pattern: str = "*") -> List[str]:
        """List files in a directory matching a pattern."""
        try:
            path = Path(directory)
            files = list(path.glob(pattern))
            return [str(f) for f in files if f.is_file()]
        except Exception as e:
            logging.error(f"Error listing files in {directory}: {e}")
            return []
    
    @staticmethod
    def file_exists(file_path: str) -> bool:
        """Check if a file exists."""
        return os.path.exists(file_path)
    
    @staticmethod
    def get_file_info(file_path: str) -> Dict[str, Any]:
        """Get information about a file."""
        try:
            stat = os.stat(file_path)
            return {
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "created": stat.st_ctime,
                "is_file": os.path.isfile(file_path),
                "is_dir": os.path.isdir(file_path)
            }
        except Exception as e:
            logging.error(f"Error getting file info for {file_path}: {e}")
            return {}

class CodeTools:
    """Tools for code analysis and manipulation."""
    
    @staticmethod
    def syntax_highlight(code: str, language: str = "python") -> str:
        """Apply syntax highlighting to code."""
        try:
            lexer = get_lexer_by_name(language)
        except:
            lexer = TextLexer()
        
        formatter = TerminalFormatter()
        return highlight(code, lexer, formatter)
    
    @staticmethod
    def validate_python_syntax(code: str) -> Tuple[bool, str]:
        """Validate Python syntax."""
        try:
            ast.parse(code)
            return True, "Valid Python syntax"
        except SyntaxError as e:
            return False, f"Syntax error: {str(e)}"
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    @staticmethod
    def extract_functions(code: str) -> List[Dict[str, str]]:
        """Extract function definitions from Python code."""
        try:
            tree = ast.parse(code)
            functions = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append({
                        "name": node.name,
                        "line": node.lineno,
                        "args": [arg.arg for arg in node.args.args],
                        "docstring": ast.get_docstring(node) or ""
                    })
            
            return functions
        except Exception as e:
            logging.error(f"Error extracting functions: {e}")
            return []
    
    @staticmethod
    def extract_classes(code: str) -> List[Dict[str, Any]]:
        """Extract class definitions from Python code."""
        try:
            tree = ast.parse(code)
            classes = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = []
                    for child in node.body:
                        if isinstance(child, ast.FunctionDef):
                            methods.append({
                                "name": child.name,
                                "line": child.lineno,
                                "args": [arg.arg for arg in child.args.args],
                                "docstring": ast.get_docstring(child) or ""
                            })
                    
                    classes.append({
                        "name": node.name,
                        "line": node.lineno,
                        "methods": methods,
                        "docstring": ast.get_docstring(node) or ""
                    })
            
            return classes
        except Exception as e:
            logging.error(f"Error extracting classes: {e}")
            return []
    
    @staticmethod
    def format_code(code: str, language: str = "python") -> str:
        """Format code using appropriate formatters."""
        if language == "python":
            try:
                import black
                mode = black.FileMode()
                return black.format_str(code, mode=mode)
            except ImportError:
                logging.warning("Black not installed, returning unformatted code")
                return code
        else:
            return code

class ProjectTools:
    """Tools for project management and analysis."""
    
    @staticmethod
    def find_project_root(start_path: str = ".") -> Optional[str]:
        """Find the root of a project by looking for common project files."""
        current = Path(start_path).resolve()
        
        while current != current.parent:
            # Check for common project indicators
            if any((current / indicator).exists() for indicator in [
                ".git", "pyproject.toml", "setup.py", "requirements.txt", 
                "package.json", "Cargo.toml", "go.mod"
            ]):
                return str(current)
            current = current.parent
        
        return None
    
    @staticmethod
    def get_project_structure(directory: str = ".", max_depth: int = 3) -> Dict[str, Any]:
        """Get a tree structure of the project."""
        def build_tree(path: Path, depth: int = 0) -> Dict[str, Any]:
            if depth > max_depth:
                return {"type": "truncated"}
            
            if path.is_file():
                return {
                    "type": "file",
                    "name": path.name,
                    "size": path.stat().st_size
                }
            elif path.is_dir():
                children = {}
                try:
                    for child in sorted(path.iterdir()):
                        if not child.name.startswith('.') and child.name not in ['__pycache__', 'node_modules']:
                            children[child.name] = build_tree(child, depth + 1)
                except PermissionError:
                    children = {"error": "Permission denied"}
                
                return {
                    "type": "directory",
                    "children": children
                }
            else:
                return {"type": "unknown"}
        
        return build_tree(Path(directory))
    
    @staticmethod
    def run_command(command: str, cwd: str = ".") -> Tuple[int, str, str]:
        """Run a shell command and return the result."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"
        except Exception as e:
            return -1, "", str(e)

class MemoryTools:
    """Tools for managing conversation and project memory."""
    
    def __init__(self, memory_file: str = "memory/conversation_memory.json"):
        self.memory_file = memory_file
        self.memory = self._load_memory()
    
    def _load_memory(self) -> Dict[str, Any]:
        """Load memory from file."""
        try:
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logging.error(f"Error loading memory: {e}")
        
        return {
            "conversations": [],
            "project_context": {},
            "file_history": []
        }
    
    def save_memory(self):
        """Save memory to file."""
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(self.memory, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving memory: {e}")
    
    def add_conversation(self, user_input: str, ai_response: str, context: str = ""):
        """Add a conversation turn to memory."""
        self.memory["conversations"].append({
            "user_input": user_input,
            "ai_response": ai_response,
            "context": context,
            "timestamp": str(Path().cwd())
        })
        
        # Keep only last 100 conversations
        if len(self.memory["conversations"]) > 100:
            self.memory["conversations"] = self.memory["conversations"][-100:]
        
        self.save_memory()
    
    def get_recent_context(self, limit: int = 5) -> str:
        """Get recent conversation context."""
        recent = self.memory["conversations"][-limit:] if self.memory["conversations"] else []
        context = []
        
        for conv in recent:
            context.append(f"User: {conv['user_input']}")
            context.append(f"Assistant: {conv['ai_response']}")
        
        return "\n".join(context)
    
    def update_project_context(self, key: str, value: Any):
        """Update project context."""
        self.memory["project_context"][key] = value
        self.save_memory()
    
    def get_project_context(self, key: str = None) -> Any:
        """Get project context."""
        if key:
            return self.memory["project_context"].get(key)
        return self.memory["project_context"]

