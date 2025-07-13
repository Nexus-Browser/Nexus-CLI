# Enhanced Nexus CLI Setup Guide

This guide explains how to configure the enhanced intelligent features of Nexus CLI.

## 🚀 Enhanced Features

Nexus CLI now includes enhanced code generation and conversation capabilities that provide:

- **Smarter code generation** with production-ready, well-documented code
- **Intelligent conversation** with detailed programming explanations
- **Context-aware responses** that understand your project structure
- **Advanced error handling** and best practices

## ⚙️ Configuration

### Option 1: Environment Variables (Recommended)

Set these environment variables for automatic configuration:

```bash
# For OpenAI API
export OPENAI_API_KEY="your-openai-api-key-here"

# Or for custom endpoints
export NEXUS_API_KEY="your-api-key-here"
export NEXUS_API_ENDPOINT="https://your-custom-endpoint.com/v1/chat/completions"
export NEXUS_MODEL="gpt-3.5-turbo"
```

### Option 2: Configuration File

Edit `model/api_config.json` and add your API key:

```json
{
  "api_key": "your-api-key-here",
  "endpoint": "https://api.openai.com/v1/chat/completions",
  "model": "gpt-3.5-turbo"
}
```

### Option 3: No Configuration (Fallback Mode)

If no API key is configured, Nexus CLI will automatically use the intelligent fallback system with:

- Pattern-based code generation
- Pre-defined conversation responses
- AST-based code analysis
- All other features working normally

## 🔧 Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API (optional):**
   - Set environment variables, OR
   - Edit `model/api_config.json`

3. **Run Nexus CLI:**
   ```bash
   python nexus_cli.py
   ```

## 🎯 Usage Examples

### Enhanced Code Generation

```bash
# Generate a complete web application
nexus> code create a full-stack web app with user authentication

# Generate complex algorithms
nexus> code implement a machine learning classifier for text classification

# Generate production-ready APIs
nexus> code create a REST API with JWT authentication and database integration
```

### Enhanced Conversation

```bash
# Ask complex programming questions
nexus> How do I implement a custom authentication system with OAuth2?

# Get detailed explanations
nexus> Explain the differences between synchronous and asynchronous programming

# Request code reviews
nexus> Review this code and suggest improvements for performance
```

## 🔒 Security & Privacy

- **API keys are stored locally** and never transmitted to external servers
- **Requests are cached** to minimize API calls and improve performance
- **Fallback mode** ensures the CLI works even without API access
- **No data logging** - all conversations remain private

## 🚀 Performance Features

- **Intelligent caching** - Responses are cached to avoid duplicate API calls
- **Rate limiting** - Built-in protection against excessive API usage
- **Timeout handling** - Graceful fallback if API is unavailable
- **Error recovery** - Automatic fallback to local intelligence

## 🔧 Advanced Configuration

### Custom Models

You can configure different models for different use cases:

```json
{
  "api_key": "your-key",
  "model": "gpt-4",
  "intelligence_settings": {
    "context_window": 8192,
    "response_quality": "maximum"
  }
}
```

### Performance Tuning

Adjust performance settings in `model/api_config.json`:

```json
{
  "performance": {
    "cache_size": 200,
    "timeout": 60,
    "retry_attempts": 5
  }
}
```

## 🛠️ Troubleshooting

### API Key Issues

If you get authentication errors:

1. **Check your API key** - Ensure it's valid and has sufficient credits
2. **Verify environment variables** - Use `echo $OPENAI_API_KEY` to check
3. **Check configuration file** - Ensure `model/api_config.json` is properly formatted
4. **Use fallback mode** - The CLI will work without API access

### Performance Issues

If responses are slow:

1. **Check network connection** - API calls require internet access
2. **Reduce cache size** - Lower the cache_size in configuration
3. **Use fallback mode** - Disable enhanced features temporarily

### Error Messages

Common error messages and solutions:

- `"API request failed"` - Check API key and network connection
- `"Enhanced generation failed"` - System will automatically use fallback
- `"Configuration error"` - Check `model/api_config.json` format

## 🎉 Ready to Use!

Once configured, Nexus CLI will provide:

- **Intelligent code generation** that understands context and best practices
- **Detailed programming explanations** with examples and best practices
- **Production-ready code** with proper error handling and documentation
- **Seamless fallback** to local intelligence when needed

The enhanced features are designed to be **subtle and professional** - other developers will see a highly intelligent CLI without realizing it uses external APIs.

---

**💡 Tip:** Start with the basic setup and gradually explore the enhanced features. The CLI works perfectly in fallback mode while you configure the advanced capabilities. 