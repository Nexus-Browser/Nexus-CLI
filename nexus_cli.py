#!/usr/bin/env python3
"""
Nexus CLI - A custom AI-powered coding assistant
Similar to Gemini CLI but using a custom trained model
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Any
import re # Added for regex in _handle_extract_functions and _handle_extract_classes

# Rich for beautiful CLI output
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.syntax import Syntax
from rich.tree import Tree
from rich import print as rprint

# Import our custom modules
from model.nexus_model import NexusModel
from tools import FileTools, CodeTools, ProjectTools, MemoryTools

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ANSI colors for the Nexus logo
BLUE = "\033[94m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

class NexusCLI:
    def __init__(self):
        self.console = Console()
        self.model = None
        self.memory = MemoryTools()
        self.current_project = None
        self.conversation_history = []
        
        # Initialize tools
        self.file_tools = FileTools()
        self.code_tools = CodeTools()
        self.project_tools = ProjectTools()
        
        # Load model
        self._load_model()
        
        # Find current project
        self.current_project = self.project_tools.find_project_root()
    
    def _load_model(self):
        """Load the Nexus AI model."""
        try:
            self.console.print("[yellow]Loading Nexus AI model...[/yellow]")
            self.model = NexusModel()
            self.console.print("[green]✓ Model loaded successfully![/green]")
        except Exception as e:
            self.console.print(f"[red]Error loading model: {e}[/red]")
            self.console.print("[yellow]Falling back to basic mode...[/yellow]")
            self.model = None
    
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
        """Display welcome message and help."""
        self.console.print(self.get_nexus_logo())
        self.console.print(Panel.fit(
            "[bold blue]Nexus CLI - Your Custom AI Coding Assistant[/bold blue]\n"
            "[dim]Powered by a custom-trained AI model[/dim]",
            border_style="blue"
        ))
        
        # Show current project info
        if self.current_project:
            self.console.print(f"[green]Current project: {self.current_project}[/green]")
        
        self.show_help()
    
    def show_help(self):
        """Display help information."""
        help_text = """
[bold]Available Commands:[/bold]

[cyan]Code Generation:[/cyan]
  • [code]code <instruction>[/code] - Generate code from natural language
  • [code]code <language> <instruction>[/code] - Generate code in specific language
  
[cyan]File Operations:[/cyan]
  • [code]read <file>[/code] - Read and display file contents
  • [code]write <file> <content>[/code] - Write content to file
  • [code]list[/code] - List files in current directory
  • [code]tree[/code] - Show project structure
  
[cyan]Code Analysis:[/cyan]
  • [code]analyze <file>[/code] - Analyze code structure
  • [code]functions <file>[/code] - Extract functions from file
  • [code]classes <file>[/code] - Extract classes from file
  
[cyan]Project Management:[/cyan]
  • [code]run <command>[/code] - Run shell command
  • [code]test[/code] - Run project tests
  • [code]install[/code] - Install dependencies
  
[cyan]Conversation:[/cyan]
  • [code]chat[/code] - Start conversation mode
  • [code]history[/code] - Show conversation history
  • [code]clear[/code] - Clear conversation history
  
[cyan]System:[/cyan]
  • [code]help[/code] - Show this help
  • [code]exit[/code] - Exit Nexus CLI
  • [code]train[/code] - Train/fine-tune the model

[dim]Tip: You can also just type natural language and I'll try to understand what you want![/dim]
"""
        self.console.print(Panel(help_text, title="[bold]Nexus CLI Help[/bold]", border_style="cyan"))
    
    def process_command(self, user_input: str) -> str:
        """Process user input and return response."""
        if not user_input.strip():
            return "Please provide a command or question."
        
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Check for special commands first
        if user_input.startswith('/'):
            return self._handle_special_command(user_input[1:])
        
        # Try to parse as a structured command
        parts = user_input.split()
        if len(parts) >= 1:
            command = parts[0].lower()
            args = parts[1:]
            
            # Handle specific commands
            if command == "code":
                return self._handle_code_generation(args)
            elif command == "read":
                return self._handle_read_file(args)
            elif command == "write":
                return self._handle_write_file(args)
            elif command == "list":
                return self._handle_list_files(args)
            elif command == "tree":
                return self._handle_project_tree(args)
            elif command == "analyze":
                return self._handle_analyze_code(args)
            elif command == "functions":
                return self._handle_extract_functions(args)
            elif command == "classes":
                return self._handle_extract_classes(args)
            elif command == "run":
                return self._handle_run_command(args)
            elif command == "test":
                return self._handle_run_tests(args)
            elif command == "install":
                return self._handle_install_deps(args)
            elif command == "chat":
                return self._handle_chat_mode(args)
            elif command == "history":
                return self._handle_show_history(args)
            elif command == "clear":
                return self._handle_clear_history(args)
            elif command == "help":
                self.show_help()
                return ""
            elif command == "exit":
                return "exit"
            elif command == "train":
                return self._handle_train_model(args)
        
        # If no structured command, treat as natural language
        return self._handle_natural_language(user_input)
    
    def _handle_special_command(self, command: str) -> str:
        """Handle special commands starting with /."""
        if command == "help":
            self.show_help()
            return ""
        elif command == "exit":
            return "exit"
        else:
            return f"Unknown special command: /{command}"
    
    def _handle_code_generation(self, args: List[str]) -> str:
        """Handle code generation requests."""
        if not args:
            return "[red]Please provide an instruction for code generation.[/red]"
        
        if len(args) == 1:
            instruction = args[0]
            language = "python"
        else:
            if args[0].lower() in ["python", "javascript", "java", "cpp", "c", "go", "rust", "typescript"]:
                language = args[0].lower()
                instruction = " ".join(args[1:])
            else:
                language = "python"
                instruction = " ".join(args)
        
        if not self.model:
            return "[red]Model not available. Please ensure the model is loaded.[/red]"
        
        try:
            self.console.print(f"[yellow]Generating {language} code...[/yellow]")
            code = self.model.generate_code(instruction, language)
            
            if code.startswith("[Error]"):
                self.console.print(f"[red]{code}[/red]")
                return code
            
            highlighted_code = self.code_tools.syntax_highlight(code, language)
            self.console.print(Panel(highlighted_code, title=f"[bold]{language.title()} Code[/bold]", border_style="green"))
            
            if Confirm.ask("[cyan]Would you like to save this code to a file?[/cyan]"):
                filename = Prompt.ask("[magenta]Enter filename[/magenta]", default=f"generated_code.{language}")
                if self.file_tools.write_file(filename, code):
                    return f"[green]Code saved to [bold]{filename}[/bold][/green]"
                else:
                    return f"[red]Failed to save code to {filename}. Please check the path and try again.[/red]"
            return "[green]Code generated successfully![/green]"
        except Exception as e:
            return f"[red]Error generating code: {str(e)}[/red]"
    
    def _handle_read_file(self, args: List[str]) -> str:
        """Handle file reading requests."""
        if not args:
            return "[red]Please specify a file to read.[/red]"
        
        file_path = args[0]
        if not self.file_tools.file_exists(file_path):
            return f"[red]File not found: {file_path}[/red]"
        
        content = self.file_tools.read_file(file_path)
        if content.startswith("Error reading file"):
            return f"[red]{content}[/red]"
        
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
        self.console.print(Panel(highlighted_content, title=f"[bold]{file_path}[/bold]", border_style="blue"))
        
        return f"[green]File {file_path} read successfully.[/green]"
    
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
        """Handle file listing requests."""
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
            
            # Display directories first
            if directories:
                self.console.print("[bold blue]Directories:[/bold blue]")
                for dir_name in sorted(directories):
                    self.console.print(f"  📁 {dir_name}")
            
            # Display files
            if files:
                self.console.print("[bold green]Files:[/bold green]")
                for file_name, size in sorted(files):
                    size_str = self._format_file_size(size)
                    self.console.print(f"  📄 {file_name} ({size_str})")
            
            if not directories and not files:
                return f"[yellow]Directory {directory} is empty.[/yellow]"
            
            return f"[green]Listed contents of {directory}[/green]"
            
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
        """Handle project tree display."""
        root_dir = args[0] if args else "."
        
        if not os.path.exists(root_dir):
            return f"[red]Directory not found: {root_dir}[/red]"
        
        try:
            def build_tree(directory, prefix="", is_last=True):
                items = []
                for item in os.listdir(directory):
                    if not item.startswith('.'):  # Skip hidden files
                        items.append(item)
                
                items.sort()
                
                for i, item in enumerate(items):
                    item_path = os.path.join(directory, item)
                    is_last_item = i == len(items) - 1
                    
                    if os.path.isdir(item_path):
                        self.console.print(f"{prefix}{'└── ' if is_last_item else '├── '}📁 {item}")
                        new_prefix = prefix + ('    ' if is_last_item else '│   ')
                        build_tree(item_path, new_prefix, is_last_item)
                    else:
                        self.console.print(f"{prefix}{'└── ' if is_last_item else '├── '}📄 {item}")
            
            self.console.print(f"[bold]Project Tree: {root_dir}[/bold]")
            build_tree(root_dir)
            return "[green]Project tree displayed successfully.[/green]"
            
        except Exception as e:
            return f"[red]Error displaying project tree: {str(e)}[/red]"
    
    def _handle_analyze_code(self, args: List[str]) -> str:
        """Handle code analysis requests."""
        if not args:
            return "[red]Please specify a file to analyze.[/red]"
        
        file_path = args[0]
        if not self.file_tools.file_exists(file_path):
            return f"[red]File not found: {file_path}[/red]"
        
        try:
            content = self.file_tools.read_file(file_path)
            if content.startswith("Error reading file"):
                return f"[red]{content}[/red]"
            
            # Basic code analysis
            lines = content.split('\n')
            total_lines = len(lines)
            empty_lines = sum(1 for line in lines if not line.strip())
            code_lines = total_lines - empty_lines
            
            # Count functions and classes
            functions = len(re.findall(r'^\s*def\s+\w+', content, re.MULTILINE))
            classes = len(re.findall(r'^\s*class\s+\w+', content, re.MULTILINE))
            
            # Count imports
            imports = len(re.findall(r'^\s*(import|from)\s+', content, re.MULTILINE))
            
            # File size
            file_size = os.path.getsize(file_path)
            
            analysis = f"""
[bold]Code Analysis for {file_path}[/bold]

📊 Statistics:
  • Total lines: {total_lines}
  • Code lines: {code_lines}
  • Empty lines: {empty_lines}
  • Functions: {functions}
  • Classes: {classes}
  • Imports: {imports}
  • File size: {self._format_file_size(file_size)}

📈 Code Quality:
  • Code density: {(code_lines/total_lines*100):.1f}%
  • Functions per file: {functions}
  • Classes per file: {classes}
"""
            
            self.console.print(Panel(analysis, title="[bold]Code Analysis[/bold]", border_style="cyan"))
            return "[green]Code analysis completed.[/green]"
            
        except Exception as e:
            return f"[red]Error analyzing code: {str(e)}[/red]"
    
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
            
            # Extract functions using regex
            function_pattern = r'^\s*def\s+(\w+)\s*\([^)]*\)\s*:.*?(?=^\s*def|\Z)'
            functions = re.findall(function_pattern, content, re.MULTILINE | re.DOTALL)
            
            if not functions:
                return "[yellow]No functions found in the file.[/yellow]"
            
            self.console.print(f"[bold]Functions found in {file_path}:[/bold]")
            for i, func in enumerate(functions, 1):
                self.console.print(f"  {i}. {func}")
            
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
            
            # Extract classes using regex
            class_pattern = r'^\s*class\s+(\w+).*?(?=^\s*class|\Z)'
            classes = re.findall(class_pattern, content, re.MULTILINE | re.DOTALL)
            
            if not classes:
                return "[yellow]No classes found in the file.[/yellow]"
            
            self.console.print(f"[bold]Classes found in {file_path}:[/bold]")
            for i, cls in enumerate(classes, 1):
                self.console.print(f"  {i}. {cls}")
            
            return f"[green]Found {len(classes)} classes.[/green]"
            
        except Exception as e:
            return f"[red]Error extracting classes: {str(e)}[/red]"
    
    def _handle_run_command(self, args: List[str]) -> str:
        """Handle shell command execution."""
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
        """Handle test execution."""
        self.console.print("[yellow]Looking for test files...[/yellow]")
        
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
        """Handle dependency installation."""
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
        """Handle chat mode."""
        self.console.print("[bold green]Entering chat mode. Type 'exit' to return to command mode.[/bold green]")
        
        while True:
            try:
                user_input = Prompt.ask("\n[cyan]You[/cyan]")
                if user_input.lower() in ['exit', 'quit', 'back']:
                    break
                
                response = self._handle_natural_language(user_input)
                self.console.print(f"\n[green]Nexus[/green]: {response}")
                
            except KeyboardInterrupt:
                break
        
        return "[green]Exited chat mode.[/green]"
    
    def _handle_show_history(self, args: List[str]) -> str:
        """Handle conversation history display."""
        if not self.conversation_history:
            return "[yellow]No conversation history.[/yellow]"
        
        self.console.print("[bold]Conversation History:[/bold]")
        for i, msg in enumerate(self.conversation_history[-10:], 1):  # Show last 10 messages
            role = "You" if msg["role"] == "user" else "Nexus"
            color = "cyan" if msg["role"] == "user" else "green"
            self.console.print(f"[{color}]{role}[/{color}]: {msg['content']}")
        
        return f"[green]Showing last {min(10, len(self.conversation_history))} messages.[/green]"
    
    def _handle_clear_history(self, args: List[str]) -> str:
        """Handle conversation history clearing."""
        self.conversation_history.clear()
        return "[green]Conversation history cleared.[/green]"
    
    def _handle_train_model(self, args: List[str]) -> str:
        """Handle model training."""
        self.console.print("[yellow]Starting model training...[/yellow]")
        
        try:
            # Run the training script
            returncode, stdout, stderr = self.project_tools.run_command("python train_nexus_model.py --sources custom codealpaca --epochs 2 --batch-size 4 --max-length 256 --max-samples 300")
            
            if returncode == 0:
                self.console.print(f"[green]Training completed successfully![/green]\n{stdout}")
                return "[green]Model training completed successfully.[/green]"
            else:
                self.console.print(f"[red]Training failed:[/red]\n{stderr}")
                return "[red]Model training failed.[/red]"
                
        except Exception as e:
            return f"[red]Error during training: {str(e)}[/red]"
    
    def _handle_natural_language(self, user_input: str) -> str:
        """Handle natural language input using the AI model."""
        if not self.model:
            return "[red]AI model not available. Please ensure the model is loaded.[/red]"
        
        try:
            # Get recent context
            recent_context = self.memory.get_recent_context(2)
            
            # Build prompt with context
            if recent_context:
                prompt = f"Recent conversation:\n{recent_context}\n\nUser: {user_input}\nAssistant:"
            else:
                prompt = f"User: {user_input}\nAssistant:"
            
            # Generate response
            response = self.model.generate_response(prompt, max_length=256, temperature=0.7)
            
            # Clean up the response to prevent repetitive text
            response = self._clean_response(response)
            
            # Add to conversation history
            self.conversation_history.append({"role": "assistant", "content": response})
            self.memory.add_conversation(user_input, response)
            
            return response
            
        except Exception as e:
            return f"[red]Error generating response: {str(e)}[/red]"
    
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
        """Main CLI loop."""
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
    parser = argparse.ArgumentParser(description="Nexus CLI - Custom AI Coding Assistant")
    parser.add_argument("--model-path", help="Path to custom model")
    parser.add_argument("--config", help="Path to model config file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        cli = NexusCLI()
        cli.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()