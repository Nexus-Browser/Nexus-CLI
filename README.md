# Nexus CLI - Custom AI Coding Assistant

A powerful command-line interface (CLI) similar to Gemini CLI, powered by a custom-trained AI model for code generation, file operations, and project management.

## 🚀 Features

### 🤖 AI-Powered Code Generation
- Generate code from natural language descriptions
- Support for multiple programming languages (Python, JavaScript, Java, C++, Go, Rust, TypeScript)
- Context-aware code generation with conversation memory
- Syntax highlighting for generated code

### 📁 File Operations
- Read and display file contents with syntax highlighting
- Write content to files with automatic directory creation
- List files and directories with detailed information
- Project structure visualization

### 🔍 Code Analysis
- Extract and analyze functions and classes from code
- Syntax validation for Python code
- Code formatting and structure analysis
- Project-wide code exploration

### 🛠️ Project Management
- Run shell commands and scripts
- Automatic test execution (pytest, unittest, npm test, cargo test, go test)
- Dependency installation (pip, npm, cargo, go)
- Project root detection

### 💬 Conversation & Memory
- Natural language conversation mode
- Conversation history tracking
- Project context memory
- Persistent memory across sessions

### 🎨 Beautiful CLI Interface
- Rich terminal output with colors and formatting
- Interactive prompts and confirmations
- Progress bars and status indicators
- Syntax-highlighted code display

## 🏗️ Architecture

```
Nexus-CLI/
├── nexus_cli.py          # Main CLI application
├── model/
│   └── nexus_model.py    # Custom AI model wrapper
├── tools.py              # File, code, and project tools
├── train_nexus_model.py  # Model training script
├── data/                 # Training data and datasets
├── memory/               # Conversation and project memory
├── model_config.json     # Model configuration
└── requirements.txt      # Python dependencies
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd Nexus-CLI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model (Optional)

```bash
# Create sample training data
python train_nexus_model.py --create-data

# Train the model with custom data
python train_nexus_model.py --epochs 3 --batch-size 4

# Train with additional datasets
python train_nexus_model.py --sources custom codealpaca --epochs 5
```

### 3. Run the CLI

```bash
# Start the CLI
python nexus_cli.py

# Or with custom model path
python nexus_cli.py --model-path ./model/nexus_model
```

## 📖 Usage Examples

### Code Generation

```bash
# Generate Python function
nexus> code create a function to calculate fibonacci numbers

# Generate JavaScript class
nexus> code javascript create a class for a todo list

# Generate code in specific language
nexus> code python write a function to sort a list
```

### File Operations

```bash
# Read a file with syntax highlighting
nexus> read main.py

# Write content to a file
nexus> write new_file.py "def hello(): print('Hello, World!')"

# List files in current directory
nexus> list

# Show project structure
nexus> tree
```

### Code Analysis

```bash
# Analyze code structure
nexus> analyze main.py

# Extract functions from file
nexus> functions utils.py

# Extract classes from file
nexus> classes models.py
```

### Project Management

```bash
# Run shell command
nexus> run python main.py

# Run tests
nexus> test

# Install dependencies
nexus> install
```

### Natural Language

```bash
# Ask questions in natural language
nexus> How do I create a web server in Python?

nexus> What's wrong with this code: def add(a, b return a + b

nexus> Help me refactor this function to be more efficient
```

## 🔧 Configuration

### Model Configuration (`model_config.json`)

```json
{
  "model_name": "microsoft/DialoGPT-medium",
  "vocab_size": 50257,
  "n_positions": 1024,
  "n_embd": 1024,
  "n_layer": 12,
  "n_head": 12,
  "max_length": 2048,
  "temperature": 0.7,
  "top_p": 0.9,
  "do_sample": true
}
```

### Training Configuration

```bash
# Basic training
python train_nexus_model.py --epochs 3 --batch-size 4

# Advanced training with custom parameters
python train_nexus_model.py \
  --model microsoft/DialoGPT-medium \
  --output ./model/custom_nexus \
  --epochs 5 \
  --batch-size 8 \
  --lr 3e-5 \
  --max-length 1024 \
  --sources custom codealpaca
```

## 🎯 Available Commands

### Code Generation
- `code <instruction>` - Generate code from natural language
- `code <language> <instruction>` - Generate code in specific language

### File Operations
- `read <file>` - Read and display file contents
- `write <file> <content>` - Write content to file
- `list [directory] [pattern]` - List files in directory
- `tree [directory] [depth]` - Show project structure

### Code Analysis
- `analyze <file>` - Analyze code structure
- `functions <file>` - Extract functions from file
- `classes <file>` - Extract classes from file

### Project Management
- `run <command>` - Run shell command
- `test` - Run project tests
- `install` - Install dependencies

### Conversation
- `chat` - Start conversation mode
- `history` - Show conversation history
- `clear` - Clear conversation history

### System
- `help` - Show help information
- `exit` - Exit Nexus CLI
- `train` - Train/fine-tune the model

## 🧠 Model Training

### Training Data Sources

1. **Custom Data** (`data/custom_training_data.json`)
   - Hand-crafted instruction-completion pairs
   - Focused on common programming tasks

2. **Code Alpaca** (`HuggingFaceH4/CodeAlpaca_20K`)
   - Large dataset of code generation examples
   - High-quality instruction-following format

3. **CodeParrot** (`codeparrot/codeparrot-clean-valid`)
   - Raw code dataset for language modeling
   - Good for understanding code patterns

### Training Process

```bash
# 1. Create sample data
python train_nexus_model.py --create-data

# 2. Train with custom data
python train_nexus_model.py --epochs 3

# 3. Train with multiple sources
python train_nexus_model.py --sources custom codealpaca --epochs 5

# 4. Fine-tune existing model
python train_nexus_model.py --model ./model/nexus_model --epochs 2
```

## 🔧 Development

### Project Structure

```
Nexus-CLI/
├── nexus_cli.py              # Main CLI application
├── model/
│   └── nexus_model.py        # AI model wrapper
├── tools.py                  # Utility tools
├── train_nexus_model.py      # Training script
├── data/
│   ├── custom_training_data.json
│   └── your_code_data.py
├── memory/
│   └── conversation_memory.json
├── model_config.json         # Model configuration
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

### Adding New Features

1. **New Commands**: Add handlers in `nexus_cli.py`
2. **New Tools**: Extend classes in `tools.py`
3. **Model Improvements**: Modify `model/nexus_model.py`
4. **Training Data**: Add to `data/custom_training_data.json`

### Testing

```bash
# Run the CLI in test mode
python nexus_cli.py --debug

# Test specific commands
echo "code create a hello world function" | python nexus_cli.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [Transformers](https://github.com/huggingface/transformers) by Hugging Face
- Inspired by Gemini CLI and similar AI coding assistants
- Uses [Rich](https://github.com/Textualize/rich) for beautiful CLI output
- Training data from Code Alpaca and CodeParrot datasets

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Error**
   ```bash
   # Ensure model is trained first
   python train_nexus_model.py --create-data
   python train_nexus_model.py
   ```

2. **Memory Issues**
   ```bash
   # Reduce batch size and sequence length
   python train_nexus_model.py --batch-size 2 --max-length 256
   ```

3. **Dependency Issues**
   ```bash
   # Reinstall dependencies
   pip install -r requirements.txt --force-reinstall
   ```

### Getting Help

- Check the help command: `help`
- Review the conversation history: `history`
- Check the logs for detailed error messages
- Ensure all dependencies are installed correctly

---

**Nexus CLI** - Your custom AI coding assistant, powered by your own trained model! 🚀
