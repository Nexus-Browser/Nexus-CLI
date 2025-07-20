# iLLuMinator-4.7B Setup Guide

## Overview
Nexus CLI now uses the advanced iLLuMinator-4.7B model via API, providing intelligent code generation without requiring local GPU resources.

## API Key Setup

### Method 1: Environment Variable (Recommended)
```bash
# Set the iLLuMinator API key as an environment variable
export ILLUMINATOR_API_KEY="your_actual_gemini_api_key_here"

# Alternative variable name
export GEMINI_API_KEY="your_actual_gemini_api_key_here"
```

### Method 2: Direct Code Modification
1. Open `model/illuminator_api.py`
2. Find the `_get_hidden_api_key` method
3. Replace `"YOUR_GEMINI_API_KEY_HERE"` with your actual API key

## Getting an API Key
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated key

## Testing the Setup
```bash
# Run the test script
python test_illuminator.py

# If successful, start Nexus CLI
python nexus_cli.py
```

## Features
- **Code Generation**: Generate code in 20+ programming languages
- **Code Analysis**: Intelligent code review and suggestions  
- **Conversational AI**: Chat with iLLuMinator for coding help
- **No GPU Required**: Lightweight API-based approach
- **Fast Response Times**: Optimized for quick code generation

## Usage Examples

### Code Generation
```bash
nexus> code python create a function to calculate fibonacci numbers
nexus> code javascript build a simple web server
nexus> code rust implement a binary search tree
```

### Code Analysis  
```bash
nexus> analyze my_script.py
nexus> functions my_code.js
nexus> classes main.py
```

### Conversational Mode
```bash
nexus> chat
You> How do I implement a REST API in Python?
iLLuMinator> I'll help you create a REST API using Flask...
```

## Troubleshooting

### "API key not valid" Error
- Ensure your API key is correctly set
- Check that the key has proper permissions
- Verify the key is from Google AI Studio (Gemini)

### "iLLuMinator API not available" Warning
- Check your internet connection
- Verify the API key is set correctly
- The system will fall back to basic mode if API is unavailable

### Import Errors
```bash
# Install required dependencies
pip install -r requirements.txt
```

## Model Information
- **Name**: iLLuMinator-4.7B
- **Parameters**: 4.7 billion
- **Architecture**: Transformer-based causal language model
- **Specialization**: Code generation and programming assistance
- **Repository**: https://github.com/Anipaleja/iLLuMinator-4.7B
- **Author**: Anish Paleja

## Security Notes
- Never commit API keys to version control
- Use environment variables in production
- Rotate API keys regularly
- Monitor API usage and set appropriate limits
