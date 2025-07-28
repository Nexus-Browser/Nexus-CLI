#!/usr/bin/env python3
"""
Nexus CLI - A custom AI-powered coding assistant
Similar to Gemini CLI but using a custom trained model
"""

import argparse
import logging
import os
import sys
import subprocess
import re
from pathlib import Path
from typing import List, Dict, Optional, Any
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.syntax import Syntax
from rich.tree import Tree
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich.align import Align
import json
import time
from datetime import datetime

# Import our modules
from model.illuminator_nexus import NexusModel  # iLLuMinator-4.7B backend
from tools import FileTools, CodeTools, ProjectTools, MemoryTools

# Rich for beautiful CLI output
BLUE = "\033[94m"
RESET = "\033[0m"

class IntelligentNexusCLI:
    """Intelligent Nexus CLI with modern features and context awareness."""
    
    def __init__(self):
        self.console = Console()
        self.model_available = False
        try:
            self.model = NexusModel()
            # Check if the iLLuMinator API is available
            if self.model.is_available():
                self.model_available = True
                model_info = self.model.get_model_info()
                self.console.print(f"[green]✓ {model_info['name']} connected successfully![/green]")
            else:
                self.console.print("[yellow]WARNING: iLLuMinator API not available, using basic mode...[/yellow]")
                self.model_available = False
        except Exception as e:
            self.console.print(f"[red]Error loading iLLuMinator model: {e}\nFalling back to basic mode...[/red]")
            self.model = None
            self.model_available = False
        self.memory = MemoryTools()
        self.current_project = None
        self.conversation_history = []
        self.command_history = []
        self.context = {}
        self.read_files_context = {}  # Store files read by user for AI context
        
        # Initialize tools
        self.file_tools = FileTools()
        self.code_tools = CodeTools()
        self.project_tools = ProjectTools()
        
        # Load model
        self._load_model()
        
        # Find current project
        self.current_project = self.project_tools.find_project_root()
        
        # Load context
        self._load_context()
    
    def _load_model(self):
        """Load the iLLuMinator-4.7B model with progress indication."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Loading iLLuMinator-4.7B model...", total=None)
            try:
                self.model = NexusModel()
                if self.model.is_available():
                    progress.update(task, description="✓ iLLuMinator-4.7B model loaded successfully!")
                    self.model_available = True
                else:
                    progress.update(task, description="WARNING: iLLuMinator-4.7B failed to load, using basic mode")
                    self.model_available = False
                time.sleep(0.5)
            except Exception as e:
                self.console.print(f"[red]Error loading iLLuMinator-4.7B model: {e}[/red]")
                self.console.print("[yellow]Falling back to basic mode...[/yellow]")
                self.model = None
                self.model_available = False
    
    def _load_context(self):
        """Load context from project and environment."""
        self.context = {
            "project_root": self.current_project,
            "current_dir": os.getcwd(),
            "python_version": sys.version_info,
            "platform": sys.platform,
            "timestamp": datetime.now().isoformat()
        }
        
        # Load project-specific context
        if self.current_project:
            self._load_project_context()
    
    def _load_project_context(self):
        """Load project-specific context."""
        project_files = [
            "requirements.txt", "pyproject.toml", "package.json", 
            "Cargo.toml", "go.mod", "README.md", ".gitignore"
        ]
        
        for file in project_files:
            file_path = os.path.join(self.current_project, file)
            if os.path.exists(file_path):
                self.context[f"has_{file}"] = True
                if file in ["requirements.txt", "package.json"]:
                    self.context[f"{file}_content"] = self.file_tools.read_file(file_path)
    
    def get_nexus_logo(self) -> str:
        """Get the colored Nexus logo."""
        nexus_ascii = """
  
 ███             ██████   █████ ██████████  ░███   ░███ ░███     ░███    █████████
  ░░░███        ░░███░███ ░███  ░███░░░░░█  ░░██   ░███ ░███     ░███ ░███░░░░░███
    ░░░███       ░███░░███░███  ░███  █ ░    ░███ ███   ░███     ░███ ░███     ░░░
      ░░░███     ░███░░███░███  ░██████       ░░████░   ░███     ░███ ░░█████████ 
       ███░      ░███ ░░██████  ░███░░█       ███░░███  ░███     ░███  ░░░░░░░░███
     ███░        ░███  ░░█████  ░███ ░   █   ███  ░░███ ░███     ░███ ░░░    ░░███
   ███░          █████  ░░█████  ██████████ ░███   ░███  ░░█████████  ░░█████████ 
  ░░░            ░░░░░    ░░░░░ ░░░░░░░░░░  ░░░     ░░░    ░░░░░░░░    ░░░░░░░░░

"""
        return f"{BLUE}{nexus_ascii}{RESET}"
    
    def show_welcome(self):
        """Display intelligent welcome message."""
        self.console.print(self.get_nexus_logo())
        
        # Create layout for welcome
        layout = Layout()
        layout.split_column(
            Layout(Panel.fit(
                "[bold blue]Nexus CLI - Intelligent AI Coding Assistant[/bold blue]\n"
                "[dim]Powered by iLLuMinator-4.7B - Local AI model from GitHub[/dim]",
                border_style="blue"
            ), name="header"),
            Layout(name="body")
        )
        
        # Add project info
        if self.current_project:
            project_info = f"[green]Project: {self.current_project}[/green]"
            if self.context.get("has_requirements.txt"):
                project_info += " [yellow]Python[/yellow]"
            if self.context.get("has_package.json"):
                project_info += " [yellow]Node.js[/yellow]"
            if self.context.get("has_Cargo.toml"):
                project_info += " [yellow]Rust[/yellow]"
            if self.context.get("has_go.mod"):
                project_info += " [yellow]Go[/yellow]"
            
            layout["body"].update(Panel(project_info, border_style="green"))
        
        self.console.print(layout)
        self.show_help()
    
    def show_help(self):
        """Display intelligent help information."""
        help_text = """
[bold]Available Commands:[/bold]

[cyan] AI & Web Intelligence:[/cyan]
  • [code]ask <question>[/code] - Ask any question with web-enhanced intelligence
  • [code]search <query>[/code] - Direct web search with intelligent synthesis
  • Just type any question naturally - I'll understand!

[cyan]Code Generation:[/cyan]
  • [code]code <instruction>[/code] - Generate intelligent code from natural language
  • [code]code <language> <instruction>[/code] - Generate code in specific language
  
[cyan]File Operations:[/cyan]
  • [code]read <file>[/code] - Read file contents and add to AI context
  • [code]write <file> <content>[/code] - Write content to file
  • [code]list [directory][/code] - List files with intelligent formatting
  • [code]tree [directory][/code] - Show project structure tree
  
[cyan]Context Management:[/cyan]
  • [code]context[/code] - Show files currently in AI context
  • [code]clearcontext[/code] - Clear all files from AI context
  
[cyan]Code Analysis:[/cyan]
  • [code]analyze <file>[/code] - Intelligent code analysis with iLLuMinator-4.7B
  • [code]functions <file>[/code] - Extract and analyze functions
  • [code]classes <file>[/code] - Extract and analyze classes
  
[cyan]Project Management:[/cyan]
  • [code]run <command>[/code] - Run shell command with output capture
  • [code]test[/code] - Intelligent test detection and execution
  • [code]install[/code] - Smart dependency installation
  
[cyan]Conversation:[/cyan]
  • [code]chat[/code] - Start intelligent conversation with iLLuMinator-4.7B
  • [code]history[/code] - Show conversation and command history
  • [code]clear[/code] - Clear conversation history
  
[cyan]System:[/cyan]
  • [code]help[/code] - Show this help
  • [code]exit[/code] - Exit Nexus CLI
  • [code]status[/code] - Check iLLuMinator API status

[dim]Powered by iLLuMinator-4.7B with Web Intelligence![/dim]
[dim]Now with comprehensive web search across Stack Overflow, GitHub, NPM, PyPI, Documentation, and more![/dim]
[dim]Repository: https://github.com/Anipaleja/iLLuMinator-4.7B[/dim]
[dim]Examples: "How to use React hooks?", "Python asyncio tutorial", "Best practices for Docker"[/dim]
[dim]Smart Features: Web search integration, context awareness, command suggestions, intelligent error handling[/dim]
"""
        self.console.print(Panel(help_text, title="[bold]Nexus CLI Help[/bold]", border_style="cyan"))
    
    def get_command_suggestions(self, partial_input: str) -> List[str]:
        """Get intelligent command suggestions based on partial input."""
        commands = [
            "ask", "search", "code", "read", "write", "list", "tree", "analyze", 
            "functions", "classes", "run", "test", "install", 
            "chat", "history", "clear", "context", "clearcontext",
            "help", "exit", "status"
        ]
        
        suggestions = []
        partial_lower = partial_input.lower()
        
        for cmd in commands:
            if cmd.startswith(partial_lower):
                suggestions.append(cmd)
        
        # Add context-aware suggestions
        if "func" in partial_lower or "def" in partial_lower:
            suggestions.extend(["code function", "functions"])
        
        if "web" in partial_lower or "server" in partial_lower:
            suggestions.extend(["code web server", "code flask"])
        
        if "file" in partial_lower:
            suggestions.extend(["read", "write", "list"])
        
        return list(set(suggestions))[:5]  # Limit to 5 suggestions
    
    def process_command(self, user_input: str) -> str:
        """Process user input with intelligent command handling."""
        if not user_input.strip():
            return "Please provide a command or question."
        
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        self.command_history.append(user_input)
        
        # Check for special commands first
        if user_input.startswith('/'):
            return self._handle_special_command(user_input[1:])
        
        # Try to parse as a structured command
        parts = user_input.split()
        if len(parts) >= 1:
            command = parts[0].lower()
            args = parts[1:]
            
            # Handle specific commands with intelligent routing
            command_handlers = {
                "ask": self._handle_ask_command,  # New web-enhanced question command
                "search": self._handle_web_search,  # New web search command
                "code": self._handle_code_generation,
                "read": self._handle_read_file,
                "write": self._handle_write_file,
                "list": self._handle_list_files,
                "tree": self._handle_project_tree,
                "analyze": self._handle_analyze_code,
                "functions": self._handle_extract_functions,
                "classes": self._handle_extract_classes,
                "run": self._handle_run_command,
                "test": self._handle_run_tests,
                "install": self._handle_install_deps,
                "chat": self._handle_chat_mode,
                "history": self._handle_show_history,
                "clear": self._handle_clear_history,
                "context": self._handle_show_context,
                "clearcontext": self._handle_clear_context,
                "help": lambda args: (self.show_help(), "")[1],
                "exit": lambda args: "exit",
                "status": self._handle_model_status
            }
            
            if command in command_handlers:
                return command_handlers[command](args)
        
        # If no structured command, treat as natural language
        return self._handle_natural_language(user_input)
    
    def _handle_special_command(self, command: str) -> str:
        """Handle special commands starting with /."""
        if command == "help":
            self.show_help()
            return ""
        elif command == "exit":
            return "exit"
        elif command == "suggest":
            return "Command suggestions: " + ", ".join(self.get_command_suggestions(""))
        else:
            return f"Unknown special command: /{command}"
    
    def _handle_code_generation(self, args: List[str]) -> str:
        """Handle intelligent code generation requests."""
        if not self.model_available:
            return "[red]iLLuMinator-4.7B model not available. Only basic commands are supported.[/red]"
        if not args:
            return "[red]Please provide an instruction for code generation.[/red]"
        
        # Detect language from instruction or args
        language = "python"  # default
        instruction = " ".join(args)
        
        # Check if first argument is a language
        if len(args) > 1:
            first_arg = args[0].lower()
            supported_languages = {
                "python", "py", "javascript", "js", "typescript", "ts", 
                "java", "cpp", "c++", "c", "go", "rust", "rs", "php", 
                "ruby", "rb", "swift", "kotlin", "scala", "r", "matlab",
                "html", "css", "sql", "bash", "sh", "powershell", "ps1",
                "markdown", "md", "json", "xml", "yaml", "yml"
            }
            
            if first_arg in supported_languages:
                language = first_arg
                instruction = " ".join(args[1:])
                
                # Normalize language names
                language_map = {
                    "py": "python", "js": "javascript", "ts": "typescript",
                    "c++": "cpp", "rs": "rust", "rb": "ruby", "sh": "bash",
                    "ps1": "powershell", "md": "markdown", "yml": "yaml"
                }
                language = language_map.get(language, language)
        
        if not self.model:
            return "[red]Model not available. Please ensure the model is loaded.[/red]"
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                task = progress.add_task(f"Generating {language} code with iLLuMinator-4.7B...", total=None)
                
                # Generate code
                code = self.model.generate_code(instruction, language)
                
                if code.startswith("[Error]"):
                    self.console.print(f"[red]{code}[/red]")
                    return code
                
                progress.update(task, description="✓ Code generated successfully!")
                time.sleep(0.3)
            
            # Syntax highlight and display
            highlighted_code = self.code_tools.syntax_highlight(code, language)
            self.console.print(Panel(highlighted_code, title=f"[bold]{language.title()} Code[/bold]", border_style="green"))
            
            # Offer to save
            if Confirm.ask("[cyan]Would you like to save this code to a file?[/cyan]"):
                # Suggest appropriate file extension
                extensions = {
                    "python": ".py", "javascript": ".js", "typescript": ".ts",
                    "java": ".java", "cpp": ".cpp", "c": ".c", "go": ".go",
                    "rust": ".rs", "php": ".php", "ruby": ".rb", "swift": ".swift",
                    "kotlin": ".kt", "scala": ".scala", "r": ".r", "matlab": ".m",
                    "html": ".html", "css": ".css", "sql": ".sql", "bash": ".sh",
                    "powershell": ".ps1", "markdown": ".md", "json": ".json",
                    "xml": ".xml", "yaml": ".yml"
                }
                ext = extensions.get(language, ".txt")
                default_filename = f"generated_code{ext}"
                
                filename = Prompt.ask("[magenta]Enter filename[/magenta]", default=default_filename)
                if self.file_tools.write_file(filename, code):
                    return f"[green]Code saved to [bold]{filename}[/bold][/green]"
                else:
                    return f"[red]Failed to save code to {filename}. Please check the path and try again.[/red]"
            
            return "[green]Code generated successfully![/green]"
            
        except Exception as e:
            return f"[red]Error generating code: {str(e)}[/red]"
    
    def _handle_read_file(self, args: List[str]) -> str:
        """Handle intelligent file reading requests."""
        if not args:
            return "[red]Please specify a file to read.[/red]"
        
        file_path = args[0]
        if not self.file_tools.file_exists(file_path):
            return f"[red]File not found: {file_path}[/red]"
        
        try:
            content = self.file_tools.read_file(file_path)
            if content.startswith("Error reading file"):
                return f"[red]{content}[/red]"
            
            # Add to read files context for AI reference
            self.read_files_context[file_path] = {
                'content': content,
                'size': len(content),
                'lines': len(content.splitlines()),
                'read_at': datetime.now().isoformat()
            }
            
            # Determine language for syntax highlighting
            ext = Path(file_path).suffix.lower()
            language_map = {
                '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
                '.java': 'java', '.cpp': 'cpp', '.c': 'c', '.go': 'go',
                '.rs': 'rust', '.html': 'html', '.css': 'css', '.json': 'json',
                '.md': 'markdown', '.txt': 'text', '.sh': 'bash', '.yml': 'yaml',
                '.yaml': 'yaml', '.xml': 'xml', '.sql': 'sql'
            }
            language = language_map.get(ext, 'text')
            
            # Syntax highlight the content
            highlighted_content = self.code_tools.syntax_highlight(content, language)
            
            # Create file info panel
            file_info = f"Size: {self._format_file_size(len(content))} | Lines: {len(content.splitlines())} | Language: {language}"
            
            self.console.print(Panel(highlighted_content, title=f"[bold]{file_path}[/bold]", subtitle=file_info, border_style="blue"))
            
            # Show context confirmation
            context_files_count = len(self.read_files_context)
            self.console.print(f"[green]✓ Added {file_path} to context ({context_files_count} files in context)[/green]")
            
            return f"[green]File {file_path} read successfully and added to context.[/green]"
            
        except Exception as e:
            return f"[red]Error reading file: {str(e)}[/red]"
    
    def _handle_write_file(self, args: List[str]) -> str:
        """Handle file writing requests."""
        if len(args) < 2:
            return "[red]Please specify a file path and content to write.[/red]"
        
        file_path = args[0]
        content = " ".join(args[1:])
        
        if self.file_tools.write_file(file_path, content):
            return f"[green]Content written to {file_path} successfully.[/green]"
        else:
            return f"[red]Failed to write to {file_path}. Please check the path and permissions.[/red]"
    
    def _handle_list_files(self, args: List[str]) -> str:
        """Handle intelligent file listing requests."""
        directory = args[0] if args else "."
        
        if not os.path.exists(directory):
            return f"[red]Directory not found: {directory}[/red]"
        
        try:
            files = []
            directories = []
            
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                if os.path.isfile(item_path):
                    size = os.path.getsize(item_path)
                    files.append((item, size))
                elif os.path.isdir(item_path):
                    directories.append(item)
            
            # Create table for better display
            table = Table(title=f"Contents of {directory}")
            table.add_column("Name", style="cyan")
            table.add_column("Type", style="green")
            table.add_column("Size", style="yellow")
            
            # Add directories first
            for dir_name in sorted(directories):
                table.add_row(f"[DIR] {dir_name}", "Directory", "")
            
            # Add files
            for file_name, size in sorted(files):
                size_str = self._format_file_size(size)
                table.add_row(f"[FILE] {file_name}", "File", size_str)
            
            self.console.print(table)
            
            if not directories and not files:
                return f"[yellow]Directory {directory} is empty.[/yellow]"
            
            return f"[green]Listed {len(directories)} directories and {len(files)} files.[/green]"
            
        except Exception as e:
            return f"[red]Error listing files: {str(e)}[/red]"
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human readable format."""
        if size_bytes == 0:
            return "0B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.1f}{size_names[i]}"
    
    def _handle_project_tree(self, args: List[str]) -> str:
        """Handle intelligent project tree display."""
        root_dir = args[0] if args else "."
        
        if not os.path.exists(root_dir):
            return f"[red]Directory not found: {root_dir}[/red]"
        
        try:
            tree = Tree(f"[DIR] {root_dir}")
            
            # Directories to exclude
            exclude_dirs = {
                'venv', '__pycache__', '.git', 'node_modules', '.pytest_cache',
                'build', 'dist', '.tox', '.mypy_cache', '.coverage', '.DS_Store',
                '*.pyc', '*.pyo', '*.pyd', '.env', '.venv', 'env'
            }
            
            def build_tree_node(directory, parent_node, depth=0):
                if depth > 5:  # Limit depth to prevent infinite recursion
                    return
                    
                try:
                    items = []
                    for item in os.listdir(directory):
                        # Skip hidden files and excluded directories
                        if (item.startswith('.') or 
                            item in exclude_dirs or
                            item.endswith('.pyc') or
                            item.endswith('.pyo') or
                            item.endswith('.pyd')):
                            continue
                        
                        # Skip if it's a virtual environment directory
                        if item == 'venv' or item == '.venv' or item == 'env':
                            continue

                        items.append(item)
                    
                    items.sort()
                    
                    for item in items:
                        item_path = os.path.join(directory, item)
                        if os.path.isdir(item_path):
                            child = parent_node.add(f"[DIR] {item}")
                            build_tree_node(item_path, child, depth + 1)
                        else:
                            parent_node.add(f"[FILE] {item}")
                except PermissionError:
                    # Skip directories we can't access
                    pass
            
            build_tree_node(root_dir, tree)
            self.console.print(tree)
            
            return "[green]Project tree displayed successfully.[/green]"
            
        except Exception as e:
            return f"[red]Error displaying project tree: {str(e)}[/red]"
    
    def _handle_analyze_code(self, args: List[str]) -> str:
        """Handle intelligent code analysis requests."""
        if not args:
            return "[red]Please specify a file to analyze.[/red]"
        
        file_path = args[0]
        if not self.file_tools.file_exists(file_path):
            return f"[red]File not found: {file_path}[/red]"
        
        try:
            content = self.file_tools.read_file(file_path)
            if content.startswith("Error reading file"):
                return f"[red]{content}[/red]"
            
            # Use the model's code analyzer
            if self.model:
                analysis = self.model.analyze_code(content)
            else:
                # Fallback analysis
                analysis = self._fallback_analysis(content)
            
            # Display analysis results
            if "error" in analysis:
                return f"[red]Analysis error: {analysis['error']}[/red]"
            
            # Create analysis table
            table = Table(title=f"Code Analysis: {file_path}")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Total Lines", str(len(content.splitlines())))
            table.add_row("Code Lines", str(len(content.splitlines()) - content.count('\n\n')))
            table.add_row("Functions", str(analysis.get('function_count', 0)))
            table.add_row("Classes", str(analysis.get('class_count', 0)))
            table.add_row("Imports", str(analysis.get('import_count', 0)))
            table.add_row("File Size", self._format_file_size(len(content)))
            
            self.console.print(table)
            
            # Show functions and classes if any
            if analysis.get('functions'):
                func_table = Table(title="Functions")
                func_table.add_column("Name", style="cyan")
                func_table.add_column("Arguments", style="yellow")
                func_table.add_column("Line", style="green")
                
                for func in analysis['functions']:
                    args_str = ", ".join(func['args'])
                    func_table.add_row(func['name'], args_str, str(func['line']))
                
                self.console.print(func_table)
            
            if analysis.get('classes'):
                class_table = Table(title="Classes")
                class_table.add_column("Name", style="cyan")
                class_table.add_column("Methods", style="yellow")
                class_table.add_column("Line", style="green")
                
                for cls in analysis['classes']:
                    methods_str = ", ".join([m['name'] for m in cls['methods']])
                    class_table.add_row(cls['name'], methods_str, str(cls['line']))
                
                self.console.print(class_table)
            
            return "[green]Code analysis completed.[/green]"
            
        except Exception as e:
            return f"[red]Error analyzing code: {str(e)}[/red]"
    
    def _fallback_analysis(self, content: str) -> Dict[str, Any]:
        """Fallback code analysis without AST."""
        lines = content.split('\n')
        total_lines = len(lines)
        empty_lines = sum(1 for line in lines if not line.strip())
        code_lines = total_lines - empty_lines
        
        functions = len(re.findall(r'^\s*def\s+\w+', content, re.MULTILINE))
        classes = len(re.findall(r'^\s*class\s+\w+', content, re.MULTILINE))
        imports = len(re.findall(r'^\s*(import|from)\s+', content, re.MULTILINE))
        
        return {
            'function_count': functions,
            'class_count': classes,
            'import_count': imports,
            'total_lines': total_lines,
            'code_lines': code_lines
        }
    
    def _handle_extract_functions(self, args: List[str]) -> str:
        """Handle function extraction requests."""
        if not args:
            return "[red]Please specify a file to extract functions from.[/red]"
        
        file_path = args[0]
        if not self.file_tools.file_exists(file_path):
            return f"[red]File not found: {file_path}[/red]"
        
        try:
            content = self.file_tools.read_file(file_path)
            if content.startswith("Error reading file"):
                return f"[red]{content}[/red]"
            
            # Use model's analyzer if available
            if self.model:
                analysis = self.model.analyze_code(content)
                functions = analysis.get('functions', [])
            else:
                # Fallback extraction
                function_pattern = r'^\s*def\s+(\w+)\s*\([^)]*\)\s*:.*?(?=^\s*def|\Z)'
                function_matches = re.findall(function_pattern, content, re.MULTILINE | re.DOTALL)
                functions = [{"name": name} for name in function_matches]
            
            if not functions:
                return "[yellow]No functions found in the file.[/yellow]"
            
            # Display functions
            table = Table(title=f"Functions in {file_path}")
            table.add_column("Name", style="cyan")
            table.add_column("Arguments", style="yellow")
            table.add_column("Line", style="green")
            
            for func in functions:
                name = func.get('name', 'Unknown')
                args = ", ".join(func.get('args', []))
                line = str(func.get('line', 'Unknown'))
                table.add_row(name, args, line)
            
            self.console.print(table)
            return f"[green]Found {len(functions)} functions.[/green]"
            
        except Exception as e:
            return f"[red]Error extracting functions: {str(e)}[/red]"
    
    def _handle_extract_classes(self, args: List[str]) -> str:
        """Handle class extraction requests."""
        if not args:
            return "[red]Please specify a file to extract classes from.[/red]"
        
        file_path = args[0]
        if not self.file_tools.file_exists(file_path):
            return f"[red]File not found: {file_path}[/red]"
        
        try:
            content = self.file_tools.read_file(file_path)
            if content.startswith("Error reading file"):
                return f"[red]{content}[/red]"
            
            # Use model's analyzer if available
            if self.model:
                analysis = self.model.analyze_code(content)
                classes = analysis.get('classes', [])
            else:
                # Fallback extraction
                class_pattern = r'^\s*class\s+(\w+).*?(?=^\s*class|\Z)'
                class_matches = re.findall(class_pattern, content, re.MULTILINE | re.DOTALL)
                classes = [{"name": name} for name in class_matches]
            
            if not classes:
                return "[yellow]No classes found in the file.[/yellow]"
            
            # Display classes
            table = Table(title=f"Classes in {file_path}")
            table.add_column("Name", style="cyan")
            table.add_column("Methods", style="yellow")
            table.add_column("Line", style="green")
            
            for cls in classes:
                name = cls.get('name', 'Unknown')
                methods = ", ".join([m['name'] for m in cls.get('methods', [])])
                line = str(cls.get('line', 'Unknown'))
                table.add_row(name, methods, line)
            
            self.console.print(table)
            return f"[green]Found {len(classes)} classes.[/green]"
            
        except Exception as e:
            return f"[red]Error extracting classes: {str(e)}[/red]"
    
    def _handle_run_command(self, args: List[str]) -> str:
        """Handle shell command execution with intelligent output."""
        if not args:
            return "[red]Please specify a command to run.[/red]"
        
        command = " ".join(args)
        self.console.print(f"[yellow]Running: {command}[/yellow]")
        
        try:
            returncode, stdout, stderr = self.project_tools.run_command(command)
            
            if returncode == 0:
                if stdout:
                    self.console.print(f"[green]Command output:[/green]\n{stdout}")
                return "[green]Command executed successfully.[/green]"
            else:
                if stderr:
                    self.console.print(f"[red]Command error:[/red]\n{stderr}")
                return f"[red]Command failed with exit code {returncode}.[/red]"
                
        except Exception as e:
            return f"[red]Error running command: {str(e)}[/red]"
    
    def _handle_run_tests(self, args: List[str]) -> str:
        """Handle intelligent test execution."""
        self.console.print("[yellow]Detecting test framework...[/yellow]")
        
        # Try common test commands
        test_commands = [
            "python -m pytest",
            "python -m unittest discover",
            "npm test",
            "cargo test",
            "go test ./..."
        ]
        
        for cmd in test_commands:
            self.console.print(f"[yellow]Trying: {cmd}[/yellow]")
            returncode, stdout, stderr = self.project_tools.run_command(cmd)
            
            if returncode == 0:
                self.console.print(f"[green]Tests passed![/green]\n{stdout}")
                return "[green]Tests completed successfully.[/green]"
            elif "not found" not in stderr and "command not found" not in stderr:
                self.console.print(f"[red]Tests failed:[/red]\n{stderr}")
                return "[red]Tests failed.[/red]"
        
        return "[red]No test framework detected. Please run tests manually.[/red]"
    
    def _handle_install_deps(self, args: List[str]) -> str:
        """Handle intelligent dependency installation."""
        self.console.print("[yellow]Detecting package manager...[/yellow]")
        
        # Try common package managers
        install_commands = [
            ("pip install -r requirements.txt", "Python"),
            ("pip install -r requirements-dev.txt", "Python (dev)"),
            ("npm install", "Node.js"),
            ("yarn install", "Node.js (Yarn)"),
            ("cargo build", "Rust"),
            ("go mod download", "Go")
        ]
        
        for cmd, lang in install_commands:
            self.console.print(f"[yellow]Trying {lang} dependencies: {cmd}[/yellow]")
            returncode, stdout, stderr = self.project_tools.run_command(cmd)
            
            if returncode == 0:
                self.console.print(f"[green]{lang} dependencies installed![/green]")
                if stdout:
                    self.console.print(stdout)
                return f"[green]{lang} dependencies installed successfully.[/green]"
            elif "not found" not in stderr and "command not found" not in stderr:
                self.console.print(f"[red]{lang} installation failed:[/red]\n{stderr}")
        
        return "[red]No package manager detected. Please install dependencies manually.[/red]"
    
    def _handle_chat_mode(self, args: List[str]) -> str:
        """Handle intelligent chat mode."""
        if not self.model_available:
            return "[red]iLLuMinator-4.7B model not available. Only basic commands are supported.[/red]"
        
        self.console.print("[bold green]Entering chat mode with iLLuMinator-4.7B. Type 'exit' to return to command mode.[/bold green]")

        while True:
            try:
                user_input = Prompt.ask("\n[cyan]You[/cyan]")
                if user_input.lower() in ['exit', 'quit', 'back']:
                    break
                
                # The model now handles typing animations internally
                response = self._handle_natural_language(user_input)
                
            except KeyboardInterrupt:
                break
        
        return "[green]Exited chat mode.[/green]"
    
    def _handle_show_history(self, args: List[str]) -> str:
        """Handle conversation and command history display."""
        if not self.conversation_history and not self.command_history:
            return "[yellow]No history available.[/yellow]"
        
        # Show conversation history
        if self.conversation_history:
            self.console.print("[bold]Recent Conversations:[/bold]")
            for i, msg in enumerate(self.conversation_history[-10:], 1):
                role = "You" if msg["role"] == "user" else "Nexus"
                color = "cyan" if msg["role"] == "user" else "green"
                self.console.print(f"[{color}]{role}[/{color}]: {msg['content']}")
        
        # Show command history
        if self.command_history:
            self.console.print("\n[bold]Recent Commands:[/bold]")
            for i, cmd in enumerate(self.command_history[-10:], 1):
                self.console.print(f"  {i}. {cmd}")
        
        return "[green]History displayed.[/green]"
    
    def _handle_clear_history(self, args: List[str]) -> str:
        """Handle conversation history clearing."""
        self.conversation_history.clear()
        self.command_history.clear()
        return "[green]History cleared.[/green]"
    
    def _handle_show_context(self, args: List[str]) -> str:
        """Handle showing current context files."""
        if not self.read_files_context:
            return "[yellow]No files in context. Use 'read <filename>' to add files to context.[/yellow]"
        
        # Create context table
        table = Table(title="Files in Context")
        table.add_column("File", style="cyan")
        table.add_column("Lines", style="green")
        table.add_column("Size", style="yellow")
        table.add_column("Read At", style="dim")
        
        for file_path, file_info in self.read_files_context.items():
            table.add_row(
                file_path,
                str(file_info['lines']),
                self._format_file_size(file_info['size']),
                file_info['read_at'][:19]  # Show date/time without microseconds
            )
        
        self.console.print(table)
        return f"[green]{len(self.read_files_context)} files in context[/green]"
    
    def _handle_clear_context(self, args: List[str]) -> str:
        """Handle clearing read files context."""
        files_count = len(self.read_files_context)
        self.read_files_context.clear()
        return f"[green]Context cleared ({files_count} files removed from context).[/green]"
    
    def _handle_model_status(self, args: List[str]) -> str:
        """Handle model status check."""
        if not self.model:
            return "[red]No model loaded[/red]"
        
        model_info = self.model.get_model_info()
        
        # Create status table
        table = Table(title="iLLuMinator-4.7B Model Status")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in model_info.items():
            table.add_row(key.replace("_", " ").title(), str(value))
        
        # Test model availability
        is_working = self.model.test_connection()
        status_color = "green" if is_working else "red"
        status_text = "✓ Available" if is_working else "✗ Not Available"
        
        table.add_row("Model Test", f"[{status_color}]{status_text}[/{status_color}]")
        
        self.console.print(table)
        
        return f"[green]Model status: {'Available' if is_working else 'Unavailable'}[/green]"
    
    def _handle_train_model(self, args: List[str]) -> str:
        """Handle model training - show info about iLLuMinator-4.7B."""
        return "[yellow]iLLuMinator-4.7B is a pre-trained model from https://github.com/Anipaleja/iLLuMinator-4.7B. No additional training required![/yellow]"
    
    def _handle_ask_command(self, args: List[str]) -> str:
        """Handle direct ask command with web-enhanced intelligence."""
        if not args:
            return "[red]Please provide a question to ask.[/red]"
        
        if not self.model_available:
            return "[red]iLLuMinator-4.7B model not available. Only basic commands are supported.[/red]"
        
        question = " ".join(args)
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                task = progress.add_task("🌐 Searching web and generating response...", total=None)
                
                # Use the enhanced iLLuMinator with web search
                response = self.model.generate_response(question, max_length=512, temperature=0.7)
                
                progress.update(task, description="✓ Complete!")
                time.sleep(0.2)
            
            # Clean and display response
            response = self._clean_response(response)
            
            # Add to conversation history
            self.conversation_history.append({"role": "user", "content": f"ask {question}"})
            self.conversation_history.append({"role": "assistant", "content": response})
            
            return response
            
        except Exception as e:
            return f"[red]Error processing question: {str(e)}[/red]"
    
    def _handle_web_search(self, args: List[str]) -> str:
        """Handle direct web search command."""
        if not args:
            return "[red]Please provide a search query.[/red]"
        
        query = " ".join(args)
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                task = progress.add_task("🔍 Searching the web...", total=None)
                
                # Import and use the external knowledge APIs directly
                from model.illuminator_api import ExternalKnowledgeAPIs
                
                external_apis = ExternalKnowledgeAPIs()
                result = external_apis.search_web_comprehensive(query)
                
                progress.update(task, description="✓ Search complete!")
                time.sleep(0.2)
            
            if result and len(result.strip()) > 20:
                return result
            else:
                return f"[yellow]No comprehensive results found for '{query}'. Try rephrasing your search.[/yellow]"
                
        except Exception as e:
            return f"[red]Error performing web search: {str(e)}[/red]"
    
    def _handle_natural_language(self, user_input: str) -> str:
        """Handle natural language input using the enhanced web-intelligent iLLuMinator model."""
        if not self.model_available:
            return "[red]iLLuMinator-4.7B model not available. Only basic commands are supported.[/red]"
        if not self.model:
            return "[red]iLLuMinator model not available. Please check the model installation.[/red]"
        
        try:
            # Show a spinner for web search (when applicable)
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                task = progress.add_task("🧠 Processing with iLLuMinator...", total=None)
                
                # Prepare context-enhanced prompt
                context_prompt = self._prepare_context_enhanced_prompt(user_input)
                
                # The enhanced iLLuMinator will automatically use web search when appropriate
                response = self.model.generate_response(context_prompt, max_length=512, temperature=0.7)
                
                progress.update(task, description="✓ Response generated!")
                time.sleep(0.2)  # Brief pause to show completion
            
            # Clean up the response
            response = self._clean_response(response)
            
            # Add to conversation history
            self.conversation_history.append({"role": "user", "content": user_input})
            self.conversation_history.append({"role": "assistant", "content": response})
            
            # Add to memory if available
            try:
                self.memory.add_conversation(user_input, response)
            except Exception as mem_error:
                # Memory error shouldn't break the chat
                self.console.print(f"[dim yellow]Note: Memory system unavailable[/dim yellow]")
            
            return response
            
        except Exception as e:
            self.console.print(f"[red]Error generating response: {str(e)}[/red]")
            return "[red]I encountered an error processing your request. Please try again.[/red]"
    
    def _prepare_context_enhanced_prompt(self, user_input: str) -> str:
        """Prepare a context-enhanced prompt with read files information."""
        # Start with the original prompt
        enhanced_prompt = user_input
        
        # Add file context if any files have been read
        if self.read_files_context:
            context_info = "\n\nCONTEXT: The user has previously read the following files that may be relevant:\n"
            
            # Add information about each read file
            for file_path, file_info in self.read_files_context.items():
                context_info += f"\n--- {file_path} ---\n"
                # Include first part of content for smaller files, or summary for large files
                content = file_info['content']
                if len(content) <= 2000:  # For smaller files, include full content
                    context_info += f"{content}\n"
                else:  # For larger files, include first 1000 chars with truncation notice
                    context_info += f"{content[:1000]}...\n[FILE TRUNCATED - {file_info['lines']} total lines, {file_info['size']} bytes]\n"
            
            context_info += "\nPlease reference these files when answering the user's question if relevant.\n"
            enhanced_prompt = context_info + "\nUSER QUESTION: " + user_input
        
        return enhanced_prompt
    
    def _clean_response(self, response: str) -> str:
        """Clean up model response to prevent repetitive or nonsensical output."""
        # Remove excessive repetition
        lines = response.split('\n')
        cleaned_lines = []
        seen_lines = set()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip if we've seen this exact line too many times
            if line in seen_lines:
                continue
            
            # Skip repetitive patterns
            if (line.lower().count('alias') > 2 or 
                line.lower().count('script') > 3 or
                len(line.split()) < 2 and line.lower() not in ['hi', 'hello', 'ok', 'yes', 'no']):
                continue
            
            cleaned_lines.append(line)
            seen_lines.add(line)
            
            # Limit response length
            if len(cleaned_lines) >= 10:
                break

        # If we have no meaningful content, provide a fallback
        if not cleaned_lines or len(' '.join(cleaned_lines)) < 10:
            return "I understand your question. Let me help you with that. Could you please provide more details about what you'd like me to help you with?"
        
        return '\n'.join(cleaned_lines)
    
    def run(self):
        """Main CLI loop with intelligent features."""
        self.show_welcome()
        while True:
            try:
                user_input = Prompt.ask("\n[cyan]nexus[/cyan]")
                if user_input.lower() in ['exit', 'quit']:
                    self.console.print("[yellow]Goodbye![/yellow]")
                    break
                
                response = self.process_command(user_input)
                if response == "exit":
                    self.console.print("[yellow]Goodbye![/yellow]")
                    break
                elif response:
                    self.console.print(f"[green]Nexus[/green]: {response}")
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Use 'exit' to quit.[/yellow]")
            except EOFError:
                break
            except Exception as e:
                self.console.print(f"[red]Error: {str(e)}[/red]")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Nexus CLI - Intelligent AI Coding Assistant")
    parser.add_argument("--model-path", help="Path to custom model")
    parser.add_argument("--config", help="Path to model config file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--fast-mode", action="store_true", help="Enable fast mode for optimized local inference")
    parser.add_argument("query", nargs="?", help="Query to process in non-interactive mode")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        cli = IntelligentNexusCLI()
        
        # Enable fast mode if requested
        if args.fast_mode and hasattr(cli.model, 'api') and hasattr(cli.model.api, 'fast_mode'):
            cli.model.api.fast_mode = True
            cli.console.print("[green]🚀 Fast mode enabled for optimized local inference![/green]")
        
        # If a query is provided, process it directly
        if args.query:
            cli.console.print("[bold blue]Nexus CLI - Fast Query Mode[/bold blue]")
            try:
                # Parse and handle the query as if it were a CLI command
                query_parts = args.query.strip().split()
                
                if len(query_parts) >= 2 and query_parts[0].lower() == "code":
                    # Handle code generation command
                    if len(query_parts) >= 3 and query_parts[1].lower() in ['python', 'rust', 'javascript', 'java', 'c', 'cpp', 'go', 'php', 'ruby', 'swift', 'kotlin']:
                        # Format: code <language> <instruction>
                        language = query_parts[1].lower()
                        instruction = " ".join(query_parts[2:])
                        code = cli.model.generate_code(instruction, language)
                        highlighted_code = cli.code_tools.syntax_highlight(code, language)
                        cli.console.print(Panel(highlighted_code, title=f"{language.title()} Code", border_style="green"))
                    else:
                        # Format: code <instruction> (default to Python)
                        instruction = " ".join(query_parts[1:])
                        code = cli.model.generate_code(instruction, "python")
                        highlighted_code = cli.code_tools.syntax_highlight(code, "python")
                        cli.console.print(Panel(highlighted_code, title="Python Code", border_style="green"))
                else:
                    # Handle as regular conversational query
                    response = cli.model.generate_response(args.query)
                    cli.console.print(Panel(response, title="Response", border_style="blue"))
                    
            except Exception as e:
                cli.console.print(f"[red]Error processing query: {str(e)}[/red]")
        else:
            cli.run()
            
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()