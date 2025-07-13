import logging
import re
import ast
import json
import os
import time
import threading
from typing import Optional, Dict, List, Any
from pathlib import Path
import subprocess
import tempfile

# HuggingFace Transformers for local LLM
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

from langchain_core.language_models.llms import LLM
from typing import Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import logging

class LangLLM(LLM):
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.2"):
        super().__init__()
        print(f"[LangLLM] Loading tokenizer for {model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"[LangLLM] Loading model for {model_name}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        print(f"[LangLLM] Creating pipeline for {model_name}")
        self._pipe = pipeline(
            "text-generation",
            model=self._model,
            tokenizer=self._tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            pad_token_id=self._tokenizer.eos_token_id
        )
        print(f"[LangLLM] Model, tokenizer, and pipeline loaded successfully!")

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model

    @property
    def pipe(self):
        return self._pipe

    @property
    def _llm_type(self) -> str:
        return "LLM Wrapper for Langchain"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, chatbot=None) -> str:
        generation_args = {
            "max_new_tokens": 256,
            "return_full_text": False,
            "temperature": 0.7,
            "do_sample": True,
            "top_p": 0.95,
            "top_k": 50
        }
        output = self._pipe(prompt, **generation_args)
        response = output[0]['generated_text']
        return response

class NexusModel:
    """LangChain LLM wrapper for Nexus CLI."""
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.2"):
        print(f"[NexusModel] Initializing with model: {model_name}")
        self.llm = LangLLM(model_name=model_name)
        self.memory = ConversationBufferMemory(llm=self.llm, max_token_limit=100)
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            verbose=False
        )
        # Mistral-style prompt templates
        self.code_prompt = PromptTemplate(
            input_variables=["instruction", "language"],
            template="""### Instruction:\nWrite a {language} function to {instruction}\n### Response:"""
        )
        self.chat_prompt = PromptTemplate(
            input_variables=["question"],
            template="""### Instruction:\n{question}\n### Response:"""
        )
        print(f"[NexusModel] Model loaded and ready.")

    def generate_code(self, instruction: str, language: str = "python") -> str:
        prompt = self.code_prompt.format(instruction=instruction, language=language)
        return self.llm._call(prompt)

    def generate_response(self, prompt: str, max_length: int = 128, temperature: float = 0.7) -> str:
        # Use Mistral-style prompt for chat
        chat_prompt = self.chat_prompt.format(question=prompt)
        return self.llm._call(chat_prompt)


class CodeAnalyzer:
    """Intelligent code analyzer using AST parsing."""
    
    def analyze(self, code: str) -> Dict[str, Any]:
        """Analyze code using AST parsing."""
        try:
            tree = ast.parse(code)
            analyzer = CodeASTVisitor()
            analyzer.visit(tree)
            return analyzer.get_results()
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
        self.lines = 0
    
    def visit_FunctionDef(self, node):
        self.functions.append({
            "name": node.name,
            "args": [arg.arg for arg in node.args.args],
            "line": node.lineno,
            "docstring": ast.get_docstring(node) or ""
        })
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
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
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        module = node.module or ""
        for alias in node.names:
            self.imports.append(f"{module}.{alias.name}")
        self.generic_visit(node)
    
    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.variables.append(target.id)
        self.generic_visit(node)
    
    def get_results(self) -> Dict[str, Any]:
        """Get analysis results."""
        return {
            "functions": self.functions,
            "classes": self.classes,
            "imports": self.imports,
            "variables": self.variables,
            "function_count": len(self.functions),
            "class_count": len(self.classes),
            "import_count": len(self.imports)
        }


class IntelligentCodeGenerator:
    """Intelligent code generator with semantic understanding."""
    
    def __init__(self):
        self.patterns = self._load_patterns()
    
    def _load_patterns(self) -> Dict[str, Any]:
        """Load intelligent code patterns."""
        return {
            "python": {
                "function_patterns": {
                    "add": {
                        "keywords": ["add", "sum", "plus", "addition", "total"],
                        "template": "def {name}(a, b):\n    return a + b",
                        "description": "Add two numbers"
                    },
                    "multiply": {
                        "keywords": ["multiply", "product", "times", "multiplication"],
                        "template": "def {name}(a, b):\n    return a * b",
                        "description": "Multiply two numbers"
                    },
                    "check_even": {
                        "keywords": ["even", "check even", "is even", "even number"],
                        "template": "def {name}(number):\n    return number % 2 == 0",
                        "description": "Check if number is even"
                    },
                    "factorial": {
                        "keywords": ["factorial", "fact", "n!", "factorial of"],
                        "template": "def {name}(n):\n    if n <= 1:\n        return 1\n    return n * {name}(n - 1)",
                        "description": "Calculate factorial"
                    },
                    "fibonacci": {
                        "keywords": ["fibonacci", "fib", "fibonacci sequence"],
                        "template": "def {name}(n):\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    return {name}(n-1) + {name}(n-2)",
                        "description": "Calculate Fibonacci number"
                    },
                    "prime": {
                        "keywords": ["prime", "is prime", "check prime", "prime number"],
                        "template": "def {name}(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
                        "description": "Check if number is prime"
                    },
                    "palindrome": {
                        "keywords": ["palindrome", "is palindrome", "check palindrome"],
                        "template": "def {name}(s):\n    s = s.lower().replace(' ', '')\n    return s == s[::-1]",
                        "description": "Check if string is palindrome"
                    },
                    "reverse": {
                        "keywords": ["reverse", "reverse string", "flip", "backwards"],
                        "template": "def {name}(s):\n    return s[::-1]",
                        "description": "Reverse a string"
                    },
                    "count_vowels": {
                        "keywords": ["count vowels", "vowels", "count vowel", "vowel count"],
                        "template": "def {name}(s):\n    vowels = 'aeiouAEIOU'\n    return sum(1 for char in s if char in vowels)",
                        "description": "Count vowels in string"
                    },
                    "sort_list": {
                        "keywords": ["sort", "sort list", "order", "arrange"],
                        "template": "def {name}(lst):\n    return sorted(lst)",
                        "description": "Sort a list"
                    },
                    "find_max": {
                        "keywords": ["max", "maximum", "find max", "highest", "largest"],
                        "template": "def {name}(lst):\n    if not lst:\n        return None\n    return max(lst)",
                        "description": "Find maximum in list"
                    },
                    "find_min": {
                        "keywords": ["min", "minimum", "find min", "lowest", "smallest"],
                        "template": "def {name}(lst):\n    if not lst:\n        return None\n    return min(lst)",
                        "description": "Find minimum in list"
                    }
                },
                "class_patterns": {
                    "calculator": {
                        "keywords": ["calculator", "calc", "math class", "arithmetic"],
                        "template": """class {name}:
    def add(self, a, b):
        return a + b
    
    def subtract(self, a, b):
        return a - b
    
    def multiply(self, a, b):
        return a * b
    
    def divide(self, a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    
    def power(self, a, b):
        return a ** b""",
                        "description": "Calculator class with basic operations"
                    },
                    "todo_list": {
                        "keywords": ["todo", "task list", "todo list", "task manager"],
                        "template": """class {name}:
    def __init__(self):
        self.tasks = []
    
    def add_task(self, task):
        self.tasks.append(task)
    
    def remove_task(self, index):
        if 0 <= index < len(self.tasks):
            self.tasks.pop(index)
    
    def list_tasks(self):
        return self.tasks
    
    def clear_tasks(self):
        self.tasks.clear()""",
                        "description": "Todo list manager"
                    }
                }
            },
            "javascript": {
                "function_patterns": {
                    "add": {
                        "keywords": ["add", "sum", "plus", "addition", "total"],
                        "template": "function {name}(a, b) {{\n    return a + b;\n}}",
                        "description": "Add two numbers"
                    },
                    "multiply": {
                        "keywords": ["multiply", "product", "times", "multiplication"],
                        "template": "function {name}(a, b) {{\n    return a * b;\n}}",
                        "description": "Multiply two numbers"
                    },
                    "check_even": {
                        "keywords": ["even", "check even", "is even", "even number"],
                        "template": "function {name}(number) {{\n    return number % 2 === 0;\n}}",
                        "description": "Check if number is even"
                    },
                    "reverse": {
                        "keywords": ["reverse", "reverse string", "flip", "backwards"],
                        "template": "function {name}(s) {{\n    return s.split('').reverse().join('');\n}}",
                        "description": "Reverse a string"
                    }
                },
                "class_patterns": {
                    "calculator": {
                        "keywords": ["calculator", "calc", "math class", "arithmetic"],
                        "template": """class {name} {{
    add(a, b) {{
        return a + b;
    }}
    
    subtract(a, b) {{
        return a - b;
    }}
    
    multiply(a, b) {{
        return a * b;
    }}
    
    divide(a, b) {{
        if (b === 0) {{
            throw new Error("Cannot divide by zero");
        }}
        return a / b;
    }}
    
    power(a, b) {{
        return Math.pow(a, b);
    }}
}}""",
                        "description": "Calculator class with basic operations"
                    }
                }
            },
            "java": {
                "function_patterns": {
                    "add": {
                        "keywords": ["add", "sum", "plus", "addition", "total"],
                        "template": "public static int {name}(int a, int b) {{\n    return a + b;\n}}",
                        "description": "Add two numbers"
                    },
                    "check_even": {
                        "keywords": ["even", "check even", "is even", "even number"],
                        "template": "public static boolean {name}(int number) {{\n    return number % 2 == 0;\n}}",
                        "description": "Check if number is even"
                    }
                },
                "class_patterns": {
                    "calculator": {
                        "keywords": ["calculator", "calc", "math class", "arithmetic"],
                        "template": """public class {name} {{
    public int add(int a, int b) {{
        return a + b;
    }}
    
    public int subtract(int a, int b) {{
        return a - b;
    }}
    
    public int multiply(int a, int b) {{
        return a * b;
    }}
    
    public double divide(int a, int b) {{
        if (b == 0) {{
            throw new IllegalArgumentException("Cannot divide by zero");
        }}
        return (double) a / b;
    }}
    
    public double power(int a, int b) {{
        return Math.pow(a, b);
    }}
}}""",
                        "description": "Calculator class with basic operations"
                    }
                }
            },
            "cpp": {
                "function_patterns": {
                    "add": {
                        "keywords": ["add", "sum", "plus", "addition", "total"],
                        "template": "int {name}(int a, int b) {{\n    return a + b;\n}}",
                        "description": "Add two numbers"
                    },
                    "check_even": {
                        "keywords": ["even", "check even", "is even", "even number"],
                        "template": "bool {name}(int number) {{\n    return number % 2 == 0;\n}}",
                        "description": "Check if number is even"
                    }
                },
                "class_patterns": {
                    "calculator": {
                        "keywords": ["calculator", "calc", "math class", "arithmetic"],
                        "template": """#include <iostream>
#include <stdexcept>

class {name} {{
public:
    int add(int a, int b) {{
        return a + b;
    }}
    
    int subtract(int a, int b) {{
        return a - b;
    }}
    
    int multiply(int a, int b) {{
        return a * b;
    }}
    
    double divide(int a, int b) {{
        if (b == 0) {{
            throw std::invalid_argument("Cannot divide by zero");
        }}
        return static_cast<double>(a) / b;
    }}
    
    double power(int a, int b) {{
        return std::pow(a, b);
    }}
}};""",
                        "description": "Calculator class with basic operations"
                    }
                }
            },
            "html": {
                "template_patterns": {
                    "basic": {
                        "keywords": ["html", "webpage", "page", "basic"],
                        "template": """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{name}</title>
</head>
<body>
    <h1>Welcome to {name}</h1>
    <p>This is a basic HTML page.</p>
</body>
</html>""",
                        "description": "Basic HTML page"
                    }
                }
            },
            "css": {
                "style_patterns": {
                    "basic": {
                        "keywords": ["css", "style", "stylesheet", "basic"],
                        "template": """body {{
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 20px;
    background-color: #f0f0f0;
}}

h1 {{
    color: #333;
    text-align: center;
}}

p {{
    color: #666;
    line-height: 1.6;
}}""",
                        "description": "Basic CSS styles"
                    }
                }
            },
            "sql": {
                "query_patterns": {
                    "select": {
                        "keywords": ["select", "query", "database", "table"],
                        "template": "SELECT * FROM {name};",
                        "description": "Basic SELECT query"
                    },
                    "create_table": {
                        "keywords": ["create table", "table", "database"],
                        "template": """CREATE TABLE {name} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);""",
                        "description": "Create table with basic structure"
                    }
                }
            },
            "markdown": {
                "template_patterns": {
                    "basic": {
                        "keywords": ["markdown", "md", "readme", "documentation"],
                        "template": """# {name}

This is a basic markdown document.

## Features

- Feature 1
- Feature 2
- Feature 3

## Usage

```bash
# Example command
echo "Hello World"
```

## Contributing

Please read the contributing guidelines before submitting a pull request.""",
                        "description": "Basic markdown document"
                    },
                    "readme": {
                        "keywords": ["readme", "read me", "project documentation"],
                        "template": """# {name}

A brief description of what this project does and who it's for.

## Installation

```bash
npm install {name}
```

## Usage

```javascript
import {{ {name} }} from '{name}';

// Example usage
{name}();
```

## API Reference

### `{name}(options)`

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `options` | `object` | **Required**. Configuration options |

## License

[MIT](https://choosealicense.com/licenses/mit/)""",
                        "description": "README template"
                    },
                    "bold": {
                        "keywords": ["bold", "**", "strong"],
                        "template": "**{name}**",
                        "description": "Bold text"
                    },
                    "italic": {
                        "keywords": ["italic", "*", "emphasis"],
                        "template": "*{name}*",
                        "description": "Italic text"
                    },
                    "code": {
                        "keywords": ["code", "`", "inline code"],
                        "template": "`{name}`",
                        "description": "Inline code"
                    }
                }
            }
        }
    
    def generate(self, instruction: str, language: str = "python") -> str:
        """
        Generate intelligent code based on instruction.
        
        Args:
            instruction: Natural language instruction
            language: Programming language
            
        Returns:
            Generated code
        """
        instruction_lower = instruction.lower()
        
        # Extract function/class name
        name = self._extract_name(instruction)
        
        # Check if language is supported
        if language not in self.patterns:
            return self._generate_default_code(instruction, name, language)
        
        lang_patterns = self.patterns[language]
        
        # Check for function patterns
        if "function_patterns" in lang_patterns:
            for pattern_name, pattern in lang_patterns["function_patterns"].items():
                if any(keyword in instruction_lower for keyword in pattern["keywords"]):
                    template = pattern["template"]
                    return template.format(name=name)
        
        # Check for class patterns
        if "class_patterns" in lang_patterns:
            for pattern_name, pattern in lang_patterns["class_patterns"].items():
                if any(keyword in instruction_lower for keyword in pattern["keywords"]):
                    template = pattern["template"]
                    return template.format(name=name)
        
        # Check for template patterns (HTML, etc.)
        if "template_patterns" in lang_patterns:
            for pattern_name, pattern in lang_patterns["template_patterns"].items():
                if any(keyword in instruction_lower for keyword in pattern["keywords"]):
                    template = pattern["template"]
                    return template.format(name=name)
        
        # Check for style patterns (CSS, etc.)
        if "style_patterns" in lang_patterns:
            for pattern_name, pattern in lang_patterns["style_patterns"].items():
                if any(keyword in instruction_lower for keyword in pattern["keywords"]):
                    template = pattern["template"]
                    return template.format(name=name)
        
        # Check for query patterns (SQL, etc.)
        if "query_patterns" in lang_patterns:
            for pattern_name, pattern in lang_patterns["query_patterns"].items():
                if any(keyword in instruction_lower for keyword in pattern["keywords"]):
                    template = pattern["template"]
                    return template.format(name=name)
        
        # Handle specific requests
        if "web server" in instruction_lower or "flask" in instruction_lower:
            return self._generate_web_server()
        
        if "file" in instruction_lower and ("read" in instruction_lower or "write" in instruction_lower):
            return self._generate_file_handler()
        
        if "database" in instruction_lower or "sql" in instruction_lower:
            return self._generate_database_code()
        
        if "api" in instruction_lower or "http" in instruction_lower:
            return self._generate_api_client()
        
        # Default intelligent response
        return self._generate_default_code(instruction, name, language)
    
    def _extract_name(self, instruction: str) -> str:
        """Extract a meaningful name from the instruction."""
        words = instruction.split()
        
        # Look for common patterns
        for i, word in enumerate(words):
            if word in ["function", "def", "create", "write", "make"] and i + 1 < len(words):
                name_parts = words[i+1:i+4]
                name = "_".join(name_parts).lower()
                name = re.sub(r'[^a-zA-Z0-9_]', '', name)
                return name if name else "my_function"
        
        # Fallback
        return "my_function"
    
    def _generate_web_server(self) -> str:
        """Generate a Flask web server."""
        return """from flask import Flask, request, jsonify

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
    app.run(debug=True)"""
    
    def _generate_file_handler(self) -> str:
        """Generate file handling code."""
        return """import os

class FileHandler:
    @staticmethod
    def read_file(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return None
    
    @staticmethod
    def write_file(filename, content):
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception:
            return False
    
    @staticmethod
    def list_files(directory="."):
        try:
            return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        except Exception:
            return []"""
    
    def _generate_database_code(self) -> str:
        """Generate database code."""
        return """import sqlite3

class DatabaseManager:
    def __init__(self, db_name="app.db"):
        self.db_name = db_name
    
    def connect(self):
        return sqlite3.connect(self.db_name)
    
    def create_table(self, table_name, columns):
        conn = self.connect()
        cursor = conn.cursor()
        column_definitions = ', '.join(columns)
        query = f"CREATE TABLE IF NOT EXISTS {table_name} ({column_definitions})"
        cursor.execute(query)
        conn.commit()
        conn.close()
    
    def insert_data(self, table_name, data):
        conn = self.connect()
        cursor = conn.cursor()
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data])
        query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        cursor.execute(query, list(data.values()))
        conn.commit()
        conn.close()"""
    
    def _generate_api_client(self) -> str:
        """Generate API client code."""
        return """import requests
import json

class APIClient:
    def __init__(self, base_url=""):
        self.base_url = base_url
        self.session = requests.Session()
    
    def get(self, endpoint=""):
        url = f"{self.base_url}/{endpoint.lstrip('/')}" if endpoint else self.base_url
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()
    
    def post(self, endpoint="", data=None):
        url = f"{self.base_url}/{endpoint.lstrip('/')}" if endpoint else self.base_url
        response = self.session.post(url, json=data)
        response.raise_for_status()
        return response.json()
    
    def put(self, endpoint="", data=None):
        url = f"{self.base_url}/{endpoint.lstrip('/')}" if endpoint else self.base_url
        response = self.session.put(url, json=data)
        response.raise_for_status()
        return response.json()
    
    def delete(self, endpoint=""):
        url = f"{self.base_url}/{endpoint.lstrip('/')}" if endpoint else self.base_url
        response = self.session.delete(url)
        response.raise_for_status()
        return True"""
    
    def _generate_default_code(self, instruction: str, name: str, language: str = "python") -> str:
        """Generate default code for unrecognized instructions."""
        if language == "python":
            return f"""# {instruction}

def {name}():
    \"\"\"
    {instruction}
    \"\"\"
    # TODO: Implement the functionality
    pass

# Example usage
result = {name}()
print(result)"""
        elif language == "javascript":
            return f"""// {instruction}

function {name}() {{
    // TODO: Implement the functionality
    return null;
}}

// Example usage
const result = {name}();
console.log(result);"""
        elif language == "java":
            return f"""// {instruction}

public class {name} {{
    public static void main(String[] args) {{
        // TODO: Implement the functionality
        System.out.println("Hello from {name}");
    }}
}}"""
        elif language == "cpp":
            return f"""// {instruction}

#include <iostream>

int {name}() {{
    // TODO: Implement the functionality
    return 0;
}}

int main() {{
    int result = {name}();
    std::cout << "Result: " << result << std::endl;
    return 0;
}}"""
        elif language == "html":
            return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{name}</title>
</head>
<body>
    <h1>{instruction}</h1>
    <p>This is a basic HTML page for {name}.</p>
</body>
</html>"""
        elif language == "css":
            return f"""/* {instruction} */

.{name} {{
    /* TODO: Add your styles here */
    color: #333;
    font-family: Arial, sans-serif;
}}"""
        elif language == "sql":
            return f"""-- {instruction}

-- TODO: Add your SQL query here
SELECT * FROM {name};"""
        elif language == "markdown":
            return f"""# {instruction}

This is a basic markdown document.

## Features

- Feature 1
- Feature 2
- Feature 3

## Usage

```bash
# Example command
echo "Hello World"
```

## Contributing

Please read the contributing guidelines before submitting a pull request."""
        else:
            return f"""# {instruction} - {language}

# TODO: Implement the functionality for {language}
# This is a placeholder for {name}

# Example usage:
# result = {name}()
# print(result)""" 