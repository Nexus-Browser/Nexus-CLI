

import os
import subprocess
import json
import re
import ast
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from rich.syntax import Syntax
from rich.console import Console
import logging

console = Console()

class FileTools:
    """Enhanced file operations with intelligent features."""
    
    def __init__(self):
        self.supported_extensions = {
            '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
            '.java': 'java', '.cpp': 'cpp', '.c': 'c', '.go': 'go',
            '.rs': 'rust', '.html': 'html', '.css': 'css', '.json': 'json',
            '.md': 'markdown', '.txt': 'text', '.sh': 'bash', '.yml': 'yaml',
            '.yaml': 'yaml', '.xml': 'xml', '.sql': 'sql', '.php': 'php',
            '.rb': 'ruby', '.swift': 'swift', '.kt': 'kotlin', '.scala': 'scala'
        }
    
    def file_exists(self, file_path: str) -> bool:
        """Check if file exists with intelligent path resolution."""
        # Try relative path first
        if os.path.exists(file_path):
            return True
        
        # Try in current directory
        current_dir_path = os.path.join(os.getcwd(), file_path)
        if os.path.exists(current_dir_path):
            return True
        
        # Try common directories
        common_dirs = ['src', 'lib', 'app', 'main', 'source']
        for dir_name in common_dirs:
            if os.path.exists(os.path.join(dir_name, file_path)):
                return True
        
        return False
    
    def read_file(self, file_path: str) -> str:
        """Read file with intelligent encoding detection and error handling."""
        try:
            # Try to find the actual file path
            actual_path = self._find_file_path(file_path)
            if not actual_path:
                return f"Error reading file: File not found - {file_path}"
            
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            for encoding in encodings:
                try:
                    with open(actual_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    return content
                except UnicodeDecodeError:
                    continue
            
            return f"Error reading file: Unable to decode {file_path}"
            
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    def write_file(self, file_path: str, content: str) -> bool:
        """Write file with intelligent directory creation."""
        try:
            # Create directory if it doesn't exist
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            console.print(f"[red]Error writing file: {str(e)}[/red]")
            return False
    
    def _find_file_path(self, file_path: str) -> Optional[str]:
        """Intelligently find the actual file path."""
        # Direct path
        if os.path.exists(file_path):
            return file_path
        
        # Current directory
        current_dir_path = os.path.join(os.getcwd(), file_path)
        if os.path.exists(current_dir_path):
            return current_dir_path
        
        # Common directories
        common_dirs = ['src', 'lib', 'app', 'main', 'source', 'tests', 'test']
        for dir_name in common_dirs:
            potential_path = os.path.join(dir_name, file_path)
            if os.path.exists(potential_path):
                return potential_path
        
        # Search recursively in current directory
        for root, dirs, files in os.walk('.'):
            if file_path in files:
                return os.path.join(root, file_path)
        
        return None
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get comprehensive file information."""
        try:
            actual_path = self._find_file_path(file_path)
            if not actual_path:
                return {"error": f"File not found: {file_path}"}
            
            stat = os.stat(actual_path)
            ext = Path(actual_path).suffix.lower()
            
            return {
                "path": actual_path,
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "extension": ext,
                "language": self.supported_extensions.get(ext, "unknown"),
                "readable": os.access(actual_path, os.R_OK),
                "writable": os.access(actual_path, os.W_OK)
            }
        except Exception as e:
            return {"error": str(e)}


class CodeTools:
    """Enhanced code analysis and manipulation tools."""
    
    def __init__(self):
        self.language_keywords = {
            'python': ['def', 'class', 'import', 'from', 'if', 'else', 'for', 'while', 'try', 'except'],
            'javascript': ['function', 'class', 'const', 'let', 'var', 'if', 'else', 'for', 'while', 'try', 'catch'],
            'typescript': ['function', 'class', 'const', 'let', 'var', 'interface', 'type', 'if', 'else', 'for', 'while'],
            'java': ['public', 'private', 'class', 'interface', 'static', 'void', 'int', 'String', 'if', 'else', 'for', 'while'],
            'cpp': ['#include', 'using', 'namespace', 'class', 'public', 'private', 'int', 'void', 'if', 'else', 'for', 'while'],
            'go': ['package', 'import', 'func', 'type', 'struct', 'interface', 'var', 'const', 'if', 'else', 'for', 'range']
        }
    
    def syntax_highlight(self, code: str, language: str = "python") -> Syntax:
        """Create syntax-highlighted code display."""
        return Syntax(code, language, theme="monokai", line_numbers=True, word_wrap=True)
    
    def analyze_code_complexity(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Analyze code complexity metrics."""
        try:
            if language == "python":
                return self._analyze_python_complexity(code)
            else:
                return self._analyze_generic_complexity(code, language)
        except SyntaxError as e:
            return {"error": f"Syntax error: {e}"}
        except Exception as e:
            return {"error": f"Analysis error: {e}"}
    
    def _analyze_python_complexity(self, code: str) -> Dict[str, Any]:
        """Analyze Python code complexity using AST."""
        try:
            tree = ast.parse(code)
            analyzer = ComplexityAnalyzer()
            analyzer.visit(tree)
            return analyzer.get_metrics()
        except SyntaxError as e:
            return {"error": f"Syntax error: {e}"}
        except Exception as e:
            return {"error": f"Analysis error: {e}"}
    
    def _analyze_generic_complexity(self, code: str, language: str) -> Dict[str, Any]:
        """Analyze code complexity for non-Python languages."""
        lines = code.split('\n')
        total_lines = len(lines)
        code_lines = sum(1 for line in lines if line.strip())
        comment_lines = sum(1 for line in lines if line.strip().startswith(('//', '/*', '*', '#')))
        
        # Count control structures
        keywords = self.language_keywords.get(language, [])
        control_structures = sum(
            sum(1 for keyword in keywords if keyword in line)
            for line in lines
        )
        
        return {
            "total_lines": total_lines,
            "code_lines": code_lines,
            "comment_lines": comment_lines,
            "control_structures": control_structures,
            "complexity_score": control_structures / max(code_lines, 1)
        }
    
    def extract_functions(self, code: str, language: str = "python") -> List[Dict[str, Any]]:
        """Extract function information from code."""
        if language == "python":
            return self._extract_python_functions(code)
        else:
            return self._extract_generic_functions(code, language)
    
    def _extract_python_functions(self, code: str) -> List[Dict[str, Any]]:
        """Extract Python functions using AST."""
        try:
            tree = ast.parse(code)
            extractor = FunctionExtractor()
            extractor.visit(tree)
            return extractor.functions
        except SyntaxError:
            return []
    
    def _extract_generic_functions(self, code: str, language: str) -> List[Dict[str, Any]]:
        """Extract functions from non-Python code using regex."""
        functions = []
        
        if language in ['javascript', 'typescript']:
            # Match function declarations
            pattern = r'function\s+(\w+)\s*\([^)]*\)\s*\{'
            matches = re.finditer(pattern, code)
            for match in matches:
                functions.append({
                    "name": match.group(1),
                    "line": code[:match.start()].count('\n') + 1,
                    "type": "function"
                })
            
            # Match arrow functions
            pattern = r'const\s+(\w+)\s*=\s*\([^)]*\)\s*=>'
            matches = re.finditer(pattern, code)
            for match in matches:
                functions.append({
                    "name": match.group(1),
                    "line": code[:match.start()].count('\n') + 1,
                    "type": "arrow_function"
                })
        
        elif language == 'java':
            pattern = r'(public|private|protected)?\s*(static)?\s*\w+\s+(\w+)\s*\([^)]*\)\s*\{'
            matches = re.finditer(pattern, code)
            for match in matches:
                functions.append({
                    "name": match.group(3),
                    "line": code[:match.start()].count('\n') + 1,
                    "type": "method"
                })
        
        return functions
    
    def format_code(self, code: str, language: str = "python") -> str:
        """Format code using appropriate formatters."""
        try:
            if language == "python":
                return self._format_python_code(code)
            elif language in ["javascript", "typescript"]:
                return self._format_js_code(code)
            else:
                return code  # Return as-is for unsupported languages
        except Exception as e:
            console.print(f"[yellow]Code formatting failed: {e}[/yellow]")
            return code
    
    def _format_python_code(self, code: str) -> str:
        """Format Python code using black-like formatting."""
        # Simple Python formatting
        lines = code.split('\n')
        formatted_lines = []
        indent_level = 0
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                formatted_lines.append('')
                continue
            
            # Adjust indent level
            if stripped.endswith(':'):
                formatted_lines.append('    ' * indent_level + stripped)
                indent_level += 1
            elif stripped in ['pass', 'break', 'continue', 'return']:
                indent_level = max(0, indent_level - 1)
                formatted_lines.append('    ' * indent_level + stripped)
            else:
                formatted_lines.append('    ' * indent_level + stripped)
        
        return '\n'.join(formatted_lines)
    
    def _format_js_code(self, code: str) -> str:
        """Format JavaScript/TypeScript code."""
        # Simple JS formatting
        lines = code.split('\n')
        formatted_lines = []
        indent_level = 0
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                formatted_lines.append('')
                continue
            
            # Adjust indent level
            if stripped.endswith('{'):
                formatted_lines.append('  ' * indent_level + stripped)
                indent_level += 1
            elif stripped.startswith('}'):
                indent_level = max(0, indent_level - 1)
                formatted_lines.append('  ' * indent_level + stripped)
            else:
                formatted_lines.append('  ' * indent_level + stripped)
        
        return '\n'.join(formatted_lines)


class ProjectTools:
    """Enhanced project management and analysis tools."""
    
    def __init__(self):
        self.project_types = {
            'python': ['requirements.txt', 'pyproject.toml', 'setup.py', 'Pipfile'],
            'nodejs': ['package.json', 'yarn.lock', 'package-lock.json'],
            'rust': ['Cargo.toml', 'Cargo.lock'],
            'go': ['go.mod', 'go.sum'],
            'java': ['pom.xml', 'build.gradle', 'gradle.properties'],
            'cpp': ['CMakeLists.txt', 'Makefile', 'build.sh']
        }
    
    def find_project_root(self) -> Optional[str]:
        """Find the project root directory intelligently."""
        current_dir = os.getcwd()
        
        # Check current directory first
        if self._is_project_root(current_dir):
            return current_dir
        
        # Walk up the directory tree
        for parent in Path(current_dir).parents:
            if self._is_project_root(str(parent)):
                return str(parent)
        
        return None
    
    def _is_project_root(self, directory: str) -> bool:
        """Check if directory is a project root."""
        try:
            files = os.listdir(directory)
            
            # Check for project files
            for project_type, indicators in self.project_types.items():
                if any(indicator in files for indicator in indicators):
                    return True
            
            # Check for common project directories
            project_dirs = ['src', 'lib', 'app', 'main', 'tests', 'docs', '.git']
            if any(dir_name in files for dir_name in project_dirs):
                return True
            
            return False
        except PermissionError:
            return False
    
    def detect_project_type(self, directory: str = None) -> str:
        """Detect the type of project in the directory."""
        if directory is None:
            directory = os.getcwd()
        
        try:
            files = os.listdir(directory)
            
            for project_type, indicators in self.project_types.items():
                if any(indicator in files for indicator in indicators):
                    return project_type
            
            return "unknown"
        except PermissionError:
            return "unknown"
    
    def run_command(self, command: str, cwd: str = None) -> Tuple[int, str, str]:
        """Run a shell command with enhanced error handling."""
        if cwd is None:
            cwd = os.getcwd()
        
        try:
            # Split command for better handling
            if isinstance(command, str):
                cmd_parts = command.split()
            else:
                cmd_parts = command
            
            # Run command
            result = subprocess.run(
                cmd_parts,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            return result.returncode, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out after 5 minutes"
        except FileNotFoundError:
            return -1, "", f"Command not found: {cmd_parts[0]}"
        except Exception as e:
            return -1, "", f"Error running command: {str(e)}"
    
    def get_project_info(self, directory: str = None) -> Dict[str, Any]:
        """Get comprehensive project information."""
        if directory is None:
            directory = os.getcwd()
        
        project_type = self.detect_project_type(directory)
        info = {
            "type": project_type,
            "root": directory,
            "size": self._get_directory_size(directory),
            "files": self._count_files(directory),
            "dependencies": self._get_dependencies(directory, project_type)
        }
        
        return info
    
    def _get_directory_size(self, directory: str) -> int:
        """Calculate directory size in bytes."""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        total_size += os.path.getsize(filepath)
                    except (OSError, FileNotFoundError):
                        continue
        except PermissionError:
            pass
        
        return total_size
    
    def _count_files(self, directory: str) -> Dict[str, int]:
        """Count files by type in directory."""
        counts = {"total": 0, "code": 0, "config": 0, "docs": 0}
        
        code_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs', '.php', '.rb'}
        config_extensions = {'.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf'}
        doc_extensions = {'.md', '.txt', '.rst', '.adoc'}
        
        try:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    counts["total"] += 1
                    ext = Path(file).suffix.lower()
                    
                    if ext in code_extensions:
                        counts["code"] += 1
                    elif ext in config_extensions:
                        counts["config"] += 1
                    elif ext in doc_extensions:
                        counts["docs"] += 1
        except PermissionError:
            pass
        
        return counts
    
    def _get_dependencies(self, directory: str, project_type: str) -> Dict[str, Any]:
        """Get project dependencies."""
        deps = {}
        
        try:
            if project_type == "python":
                requirements_file = os.path.join(directory, "requirements.txt")
                if os.path.exists(requirements_file):
                    with open(requirements_file, 'r') as f:
                        deps["requirements"] = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            elif project_type == "nodejs":
                package_file = os.path.join(directory, "package.json")
                if os.path.exists(package_file):
                    with open(package_file, 'r') as f:
                        package_data = json.load(f)
                        deps["dependencies"] = package_data.get("dependencies", {})
                        deps["devDependencies"] = package_data.get("devDependencies", {})
            
            elif project_type == "rust":
                cargo_file = os.path.join(directory, "Cargo.toml")
                if os.path.exists(cargo_file):
                    # Simple parsing of Cargo.toml
                    with open(cargo_file, 'r') as f:
                        content = f.read()
                        deps["dependencies"] = self._parse_cargo_dependencies(content)
        
        except Exception as e:
            deps["error"] = str(e)
        
        return deps
    
    def _parse_cargo_dependencies(self, content: str) -> Dict[str, str]:
        """Parse Cargo.toml dependencies."""
        deps = {}
        in_dependencies = False
        
        for line in content.split('\n'):
            line = line.strip()
            if line == '[dependencies]':
                in_dependencies = True
            elif line.startswith('[') and line.endswith(']'):
                in_dependencies = False
            elif in_dependencies and '=' in line:
                parts = line.split('=', 1)
                if len(parts) == 2:
                    name = parts[0].strip()
                    version = parts[1].strip().strip('"')
                    deps[name] = version
        
        return deps


class MemoryTools:
    """Enhanced memory and context management."""
    
    def __init__(self, memory_file: str = "memory/conversation_memory.json"):
        self.memory_file = memory_file
        self.conversations = []
        self.context_cache = {}
        self.max_memory_size = 1000
        
        # Ensure memory directory exists
        os.makedirs(os.path.dirname(memory_file), exist_ok=True)
        
        # Load existing memory
        self._load_memory()
    
    def add_conversation(self, user_input: str, assistant_response: str):
        """Add a conversation to memory."""
        conversation = {
            "timestamp": self._get_timestamp(),
            "user_input": user_input,
            "assistant_response": assistant_response,
            "context": self._extract_context(user_input)
        }
        
        self.conversations.append(conversation)
        
        # Limit memory size
        if len(self.conversations) > self.max_memory_size:
            self.conversations = self.conversations[-self.max_memory_size:]
        
        # Save to file
        self._save_memory()
    
    def get_recent_context(self, num_conversations: int = 5) -> str:
        """Get recent conversation context."""
        recent = self.conversations[-num_conversations:] if self.conversations else []
        
        context_parts = []
        for conv in recent:
            context_parts.append(f"User: {conv['user_input']}")
            context_parts.append(f"Assistant: {conv['assistant_response']}")
        
        return "\n".join(context_parts)
    
    def search_memory(self, query: str) -> List[Dict[str, Any]]:
        """Search memory for relevant conversations."""
        query_lower = query.lower()
        results = []
        
        for conv in self.conversations:
            if (query_lower in conv['user_input'].lower() or 
                query_lower in conv['assistant_response'].lower()):
                results.append(conv)
        
        return results
    
    def _extract_context(self, user_input: str) -> Dict[str, Any]:
        """Extract context from user input."""
        context = {
            "topics": [],
            "intent": "unknown",
            "entities": []
        }
        
        # Extract topics
        topics = ['code', 'function', 'class', 'file', 'project', 'test', 'debug', 'help']
        for topic in topics:
            if topic in user_input.lower():
                context["topics"].append(topic)
        
        # Extract intent
        if any(word in user_input.lower() for word in ['create', 'make', 'generate', 'build']):
            context["intent"] = "create"
        elif any(word in user_input.lower() for word in ['read', 'show', 'display', 'list']):
            context["intent"] = "read"
        elif any(word in user_input.lower() for word in ['analyze', 'examine', 'check']):
            context["intent"] = "analyze"
        elif any(word in user_input.lower() for word in ['run', 'execute', 'test']):
            context["intent"] = "execute"
        
        return context
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _load_memory(self):
        """Load memory from file."""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    self.conversations = data.get('conversations', [])
        except Exception as e:
            print(f"Warning: Could not load memory: {e}")
    
    def _save_memory(self):
        """Save memory to file."""
        try:
            with open(self.memory_file, 'w') as f:
                json.dump({
                    'conversations': self.conversations,
                    'last_updated': self._get_timestamp()
                }, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save memory: {e}")


class ComplexityAnalyzer(ast.NodeVisitor):
    """AST visitor for analyzing code complexity."""
    
    def __init__(self):
        self.functions = 0
        self.classes = 0
        self.imports = 0
        self.control_structures = 0
        self.lines = 0
        self.comments = 0
    
    def visit_FunctionDef(self, node):
        self.functions += 1
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        self.classes += 1
        self.generic_visit(node)
    
    def visit_Import(self, node):
        self.imports += 1
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        self.imports += 1
        self.generic_visit(node)
    
    def visit_If(self, node):
        self.control_structures += 1
        self.generic_visit(node)
    
    def visit_For(self, node):
        self.control_structures += 1
        self.generic_visit(node)
    
    def visit_While(self, node):
        self.control_structures += 1
        self.generic_visit(node)
    
    def visit_Try(self, node):
        self.control_structures += 1
        self.generic_visit(node)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get complexity metrics."""
        return {
            "functions": self.functions,
            "classes": self.classes,
            "imports": self.imports,
            "control_structures": self.control_structures,
            "complexity_score": self.control_structures / max(self.functions, 1)
        }


class FunctionExtractor(ast.NodeVisitor):
    """AST visitor for extracting function information."""
    
    def __init__(self):
        self.functions = []
    
    def visit_FunctionDef(self, node):
        func_info = {
            "name": node.name,
            "args": [arg.arg for arg in node.args.args],
            "line": node.lineno,
            "docstring": ast.get_docstring(node) or "",
            "decorators": [self._get_decorator_name(d) for d in node.decorator_list]
        }
        self.functions.append(func_info)
        self.generic_visit(node)
    
    def _get_decorator_name(self, decorator) -> str:
        """Get decorator name."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return f"{decorator.value.id}.{decorator.attr}"
        else:
            return "unknown"

