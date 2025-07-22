# Nexus CLI - Intelligent AI Coding Assistant

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![iLLuMinator](https://img.shields.io/badge/Powered%20by-iLLuMinator--4.7B-purple.svg)](https://github.com/Anipaleja/iLLuMinator-4.7B)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()

> **🚀 The most intelligent CLI coding assistant - powered by the local iLLuMinator-4.7B model!**

Nexus CLI is a revolutionary command-line interface that combines the best practices from successful CLI tools like **Warp**, **Cursor**, **Gemini CLI**, and **Claude Code** to provide an intelligent, context-aware coding experience. Powered by the **iLLuMinator-4.7B** model from [GitHub](https://github.com/Anipaleja/iLLuMinator-4.7B), it delivers professional-grade code generation and analysis using local AI processing.

## ✨ Key Features

### 🧠 **Intelligent Code Generation**
- **Natural Language Processing**: Generate code from plain English descriptions
- **Multi-Language Support**: Python, JavaScript, TypeScript, Java, C++, Go, Rust, and 20+ more
- **Context-Aware Generation**: Smart code patterns based on your project structure and file context
- **Local Processing**: Uses iLLuMinator-4.7B model running locally on your machine

### 🔍 **Advanced Code Analysis**
- **Deep Code Understanding**: Powered by iLLuMinator-4.7B's advanced transformer architecture
- **Function & Class Extraction**: Intelligent code structure analysis
- **Syntax Highlighting**: Beautiful, syntax-highlighted code display
- **Context-Aware Suggestions**: AI understands your project structure

### 📁 **Smart File Operations with Context**
- **Context-Aware File Reading**: Files are automatically added to AI context when read
- **Intelligent Path Resolution**: Automatically find files in common directories
- **Project Tree Visualization**: Beautiful tree structures with Rich terminal output
- **File Type Recognition**: Automatic language detection and syntax highlighting

### 🧠 **Context Management**
- **Persistent Context**: Files remain in AI memory across conversations
- **Context Commands**: View and manage files in AI context
- **Smart Referencing**: AI automatically references relevant files when answering questions

### 🛠️ **Project Management**
- **Project Detection**: Automatically identify project types (Python, Node.js, Rust, Go, etc.)
- **Dependency Analysis**: Smart detection and installation of project dependencies
- **Test Framework Detection**: Intelligent test execution across multiple frameworks
- **Build System Integration**: Support for various build tools and package managers

### 🎨 **Modern CLI Experience**
- **Rich Terminal Output**: Beautiful, colorful interface with progress indicators
- **Command Suggestions**: Context-aware command completion and suggestions
- **Memory Management**: Persistent conversation and command history
- **Conversational AI**: Chat with iLLuMinator-4.7B for coding help
- **Error Handling**: Graceful error recovery with helpful suggestions

### 🚀 **Performance & Reliability**
- **Local AI Model**: No external API dependencies once set up
- **Fast Execution**: Optimized for speed with intelligent caching
- **Cross-Platform**: Works on macOS, Linux, and Windows
- **Production Ready**: Robust error handling and edge case management
- **Offline Capable**: Works completely offline after initial setup

## 🎯 **What Makes Nexus CLI Special**

Unlike other CLI tools that rely on external AI services, Nexus CLI uses:

- **🤖 Local iLLuMinator-4.7B Model**: Advanced transformer model running on your hardware
- **🔧 No API Keys Required**: No external dependencies or rate limits
- **🧠 Context Awareness**: Files automatically added to AI context for better responses
- **📊 Deep Code Understanding**: Transformer-based code analysis and generation
- **🎨 Modern CLI Patterns**: Best practices from successful CLI tools
- **⚡ Privacy-First**: All processing happens locally on your machine

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Nexus-CLI.git
cd Nexus-CLI

# Install dependencies and set up iLLuMinator-4.7B model
python setup_illuminator.py

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the CLI
python nexus_cli.py
```

**Note**: The setup script will automatically download and configure the iLLuMinator-4.7B model from GitHub. This may take several minutes depending on your internet connection.

### Basic Usage

```bash
# Generate code from natural language
nexus> code function to add two numbers
nexus> code class calculator with basic operations
nexus> code web server with flask

# Read files and add to AI context
nexus> read myfile.py
nexus> context                    # View files in context
nexus> clearcontext              # Clear context

# Analyze existing code with context awareness
nexus> analyze myfile.py
nexus> functions myfile.py
nexus> classes myfile.py

# Ask questions about your code (AI will reference context files)
nexus> What does this project do?
nexus> How can I improve the main function?
nexus> Explain the class structure

# Project operations
nexus> list
nexus> tree
nexus> run python my_script.py
nexus> test
nexus> install

# Natural language conversation
nexus> What is a variable in programming?
nexus> How do I create a web server?
nexus> Explain object-oriented programming
```

## 📚 **Command Reference**

### Code Generation
```bash
code <instruction>                    # Generate code from natural language
code <language> <instruction>         # Generate code in specific language
```

### File Operations
```bash
read <file>                          # Read file and add to AI context
write <file> <content>               # Write content to file
list [directory]                     # List files with intelligent formatting
tree [directory]                     # Show project structure tree
```

### Context Management
```bash
context                              # Show files currently in AI context
clearcontext                         # Clear all files from AI context
```

### Code Analysis
```bash
analyze <file>                       # Intelligent code analysis with iLLuMinator-4.7B
functions <file>                     # Extract and analyze functions
classes <file>                       # Extract and analyze classes
```

### Project Management
```bash
run <command>                        # Run shell command with output capture
test                                 # Intelligent test detection and execution
install                              # Smart dependency installation
```

### Conversation & System
```bash
chat                                 # Start intelligent conversation mode
history                              # Show conversation and command history
clear                                # Clear conversation history
help                                 # Show comprehensive help
exit                                 # Exit Nexus CLI
train                                # Train/fine-tune the model
```

## **Examples**

### Generate a Web Server
```bash
nexus> code web server with flask
```
**Output:**
```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"message": "Hello, World!", "status": "running"})

@app.route('/api/data', methods=['GET'])
def get_data():
    return jsonify({"data": "Some data"})

@app.route('/api/data', methods=['POST'])
def post_data():
    data = request.get_json()
    return jsonify({"received": data})

if __name__ == '__main__':
    app.run(debug=True)
```

### Analyze Code Structure
```bash
nexus> analyze my_module.py
```
**Output:**
```
Code Analysis: my_module.py
┌──────────────┬─────────┐
│ Metric       │ Value   │
├──────────────┼─────────┤
│ Total Lines  │ 45      │
│ Code Lines   │ 38      │
│ Functions    │ 3       │
│ Classes      │ 1       │
│ Imports      │ 2       │
│ File Size    │ 1.2KB   │
└──────────────┴─────────┘
```

### Project Tree Visualization
```bash
nexus> tree
```
**Output:**
```
 .
├──  src
│   ├──  main.py
│   └──  utils.py
├──  tests
│   └──  test_main.py
├──  requirements.txt
└──  README.md
```

## 🔧 **Advanced Features**

### Context-Aware Suggestions
Nexus CLI provides intelligent command suggestions based on your current context:
- **File operations** when you're working with files
- **Code generation** when you mention functions or classes
- **Project management** when you're in a project directory

### Memory and History
- **Persistent conversation history** across sessions
- **Command history** with intelligent search
- **Context extraction** from user inputs
- **Smart memory management** with automatic cleanup

### Project Intelligence
- **Automatic project type detection** (Python, Node.js, Rust, Go, etc.)
- **Dependency analysis** and smart installation
- **Build system integration** for various frameworks
- **Test framework detection** and execution

## 🏗️ **Architecture**

Nexus CLI is built with a modular, extensible architecture:

```
nexus_cli.py          # Main CLI interface
├── model/
│   └── nexus_model.py # Intelligent code generation engine
├── tools.py          # Enhanced utility tools
├── memory/           # Conversation and context memory
└── examples/         # Demo and example files
```

### Core Components

1. **IntelligentNexusCLI**: Main CLI class with modern features
2. **NexusModel**: Advanced code generation with AST analysis
3. **FileTools**: Enhanced file operations with intelligent path resolution
4. **CodeTools**: Advanced code analysis and manipulation
5. **ProjectTools**: Smart project management and detection
6. **MemoryTools**: Context-aware memory and conversation management

## 🚀 **Performance**

- **⚡ Instant responses** - No API calls or network latency
- **🧠 Intelligent caching** - Smart memory management
- **📊 Optimized analysis** - Fast AST parsing and code generation
- **🎯 Context awareness** - Efficient project and file detection

## **Contributing**

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone and setup
git clone https://github.com/yourusername/Nexus-CLI.git
cd Nexus-CLI
pip install -r requirements.txt

# Run tests
python test_enhanced_cli.py

# Run demo
python demo_enhanced_features.py
```

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## **Acknowledgments**

Nexus CLI is inspired by and incorporates techniques from:
- **Warp** - Modern terminal experience
- **Cursor** - Intelligent code editing
- **Gemini CLI** - Natural language processing
- **Claude Code** - Advanced code analysis
- **Rich** - Beautiful terminal output
- **AST** - Abstract Syntax Tree analysis

## **Support**

- **Issues**: [GitHub Issues](https://github.com/yourusername/Nexus-CLI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/Nexus-CLI/discussions)
- **Documentation**: [Wiki](https://github.com/yourusername/Nexus-CLI/wiki)

---

** Ready to experience the future of CLI coding assistants? Try Nexus CLI today!**

*No GPT wrappers. No external dependencies. Pure intelligent coding assistance.*
