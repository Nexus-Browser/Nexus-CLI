# 🤖 Nexus CLI - API Configuration Guide

## Current Status
- ✅ Multi-provider API system implemented
- ❌ Gemini API quota exhausted (50/day limit reached)  
- ❌ Groq API quota exhausted
- 🔄 Alternative APIs available

## Available API Providers

### 1. 🤗 Hugging Face (Recommended - Free & Many Models)
```python
# Get your free token at: https://huggingface.co/settings/tokens
hf_key = "hf_your_token_here"
```

**Popular Models Available:**
- `meta-llama/Llama-2-7b-chat-hf` - Llama 2 Chat (7B)
- `meta-llama/Llama-2-13b-chat-hf` - Llama 2 Chat (13B) 
- `microsoft/DialoGPT-large` - Conversational AI
- `microsoft/CodeBERT-base` - Code understanding
- `bigcode/starcoder` - Code generation
- `HuggingFaceH4/zephyr-7b-beta` - Instruction following

**Note:** The model you mentioned (`meta-llama/Llama-4-Maverick-17B-128E-Instruct`) doesn't exist. Llama 3 is the latest version from Meta.

### 2. 🌟 Anthropic Claude
```python
# Get your key at: https://console.anthropic.com  
claude_key = "sk-ant-your-key-here"
```
- High quality responses
- Good free tier
- Excellent for coding tasks

### 3. ⚡ Groq (Currently Exhausted)
```python
groq_key = "gsk_your_key_here"
```
- Very fast responses
- Limited daily quota

### 4. 🧠 Google Gemini (Currently Exhausted)
```python
gemini_key = "AIza_your_key_here"
```
- 50 requests/day free
- Quota resets daily

### 5. 💰 OpenAI
```python  
openai_key = "sk-your_key_here"
```
- Industry standard
- Requires payment

## How to Switch APIs

### Method 1: Environment Variables (Recommended)
```bash
export ILLUMINATOR_API_KEY="hf_your_huggingface_token"
# OR
export OPENAI_API_KEY="sk-your_openai_key"
# OR  
export ANTHROPIC_API_KEY="sk-ant-your_anthropic_key"
```

### Method 2: Edit the Code
In `model/illuminator_api.py`, modify the `_get_hidden_api_key()` method:

```python
def _get_hidden_api_key(self) -> str:
    # Option 1: Hugging Face (free, many models)
    hf_key = "hf_your_token_here"
    
    # Option 2: Anthropic (high quality)  
    # claude_key = "sk-ant-your_key_here"
    
    return hf_key  # Return your preferred key
```

## Setting Up Hugging Face

1. **Get Token**: Visit https://huggingface.co/settings/tokens
2. **Create Token**: Click "New token" → "Read" access is enough
3. **Copy Token**: Starts with `hf_`
4. **Update Code**: Replace the key in the API configuration
5. **Choose Model**: Update `self.hf_model` in the init method

### Example Hugging Face Setup
```python
# In illuminator_api.py __init__ method:
elif self._api_key.startswith('hf_'):
    self.base_url = "https://api-inference.huggingface.co/models"
    self.api_type = "huggingface"
    # Choose your model:
    self.hf_model = "meta-llama/Llama-2-7b-chat-hf"  # Good chat model
    # self.hf_model = "microsoft/DialoGPT-large"      # Conversational
    # self.hf_model = "bigcode/starcoder"             # Code generation
```

## Web Interface Button Issues - FIXED ✅

The "New Tab" button was using incorrect selector (class vs ID). This has been fixed:
- ✅ New Tab button now works
- ✅ All other buttons functional
- ✅ Settings, fullscreen, split terminal all working

## Quick Test Commands

```bash
# Test API connection
python3 -c "from model.illuminator_api import iLLuMinatorAPI; api = iLLuMinatorAPI(); print(api.generate_response('Hello!'))"

# Start web server  
source nexus_venv/bin/activate
python -m uvicorn web.backend.main:app --host 0.0.0.0 --port 8001 --reload
```

## Next Steps

1. **Get a Hugging Face token** (free and easy)
2. **Update the API key** in the code
3. **Choose a good model** for your use case
4. **Test the system** 

The web interface is fully functional at http://localhost:8001 with all buttons working!
