"""
iLLuMinator LLM Configuration and Management
Advanced configuration for the iLLuMinator-4.7B model integration
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class iLLuMinatorConfig:
    """Configuration manager for iLLuMinator-4.7B LLM integration."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or str(Path(__file__).parent / "illuminator_config.json")
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load iLLuMinator configuration from file."""
        default_config = self._get_default_config()
        
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    file_config = json.load(f)
                # Merge with defaults
                default_config.update(file_config)
            else:
                logger.info("Config file not found, creating with defaults")
                self._save_config(default_config)
        except Exception as e:
            logger.warning(f"Error loading config: {e}, using defaults")
        
        return default_config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default iLLuMinator configuration."""
        return {
            "model_config": {
                "name": "iLLuMinator-4.7B",
                "version": "1.0",
                "author": "Anish Paleja",
                "repository": "https://github.com/Anipaleja/iLLuMinator-4.7B",
                "description": "Advanced 4.7B parameter model for code generation and assistance",
                "model_type": "causal-lm",
                "architecture": "transformer",
                "parameters": "4.7B",
                "training_data": "Code repositories, documentation, and programming tutorials",
                "capabilities": [
                    "Code generation in 20+ languages",
                    "Code analysis and explanation", 
                    "Debugging assistance",
                    "Documentation generation",
                    "Conversational programming help",
                    "Context-aware responses",
                    "Multi-turn conversations",
                    "Code refactoring suggestions"
                ],
                "supported_languages": [
                    "Python", "JavaScript", "TypeScript", "Java", "C++", "C",
                    "Go", "Rust", "PHP", "Ruby", "Swift", "Kotlin", "Scala",
                    "R", "MATLAB", "HTML", "CSS", "SQL", "Bash", "PowerShell",
                    "Haskell", "Erlang", "Elixir", "Clojure", "F#", "VB.NET"
                ]
            },
            "api_config": {
                "primary_endpoint": "https://api.illuminator.dev/v1",
                "fallback_endpoints": [
                    "https://api.openai.com/v1/chat/completions",
                    "https://api.anthropic.com/v1/messages",
                    "https://api.cohere.ai/v1/chat",
                    "https://api.groq.com/openai/v1/chat/completions"
                ],
                "max_tokens": 2048,
                "temperature": 0.7,
                "top_p": 0.95,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "timeout": 30,
                "retry_attempts": 3,
                "retry_delay": 1.0
            },
            "features": {
                "code_completion": True,
                "code_explanation": True,
                "bug_detection": True,
                "refactoring_suggestions": True,
                "documentation_generation": True,
                "test_generation": True,
                "performance_optimization": True,
                "context_awareness": True,
                "multi_turn_chat": True,
                "secure_conversations": True
            },
            "security": {
                "quantum_encryption": True,
                "blockchain_verification": True,
                "local_context_storage": True,
                "api_key_encryption": True,
                "conversation_encryption": True
            },
            "performance": {
                "cache_enabled": True,
                "cache_size": 1000,
                "context_window": 8192,
                "response_streaming": True,
                "batch_processing": False,
                "parallel_requests": 3
            },
            "logging": {
                "level": "INFO",
                "file_logging": True,
                "console_logging": True,
                "log_api_calls": True,
                "log_responses": False  # For privacy
            }
        }
    
    def _validate_config(self):
        """Validate the configuration."""
        required_sections = ["model_config", "api_config", "features", "security", "performance"]
        for section in required_sections:
            if section not in self.config:
                logger.warning(f"Missing config section: {section}")
                self.config[section] = self._get_default_config()[section]
    
    def _save_config(self, config: Dict[str, Any]):
        """Save configuration to file."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def get_api_key(self) -> Optional[str]:
        """Get API key from environment or config."""
        # Try environment variables first (recommended)
        api_key = (
            os.environ.get('ILLUMINATOR_API_KEY') or
            os.environ.get('OPENAI_API_KEY') or
            os.environ.get('ANTHROPIC_API_KEY') or
            os.environ.get('COHERE_API_KEY') or
            os.environ.get('GROQ_API_KEY')
        )
        
        if api_key:
            return api_key
        
        # Check config file (less secure)
        return self.config.get('api_config', {}).get('api_key')
    
    def get_endpoint(self) -> str:
        """Get the appropriate API endpoint."""
        api_key = self.get_api_key()
        
        if not api_key:
            return "local"
        
        # Determine endpoint based on API key format
        if api_key.startswith('sk-ant-'):
            return "https://api.anthropic.com/v1/messages"
        elif api_key.startswith('gsk_'):
            return "https://api.groq.com/openai/v1/chat/completions"
        elif api_key.startswith('AIza'):
            return "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
        elif len(api_key) == 40:
            return "https://api.cohere.ai/v1/chat"
        else:
            return "https://api.openai.com/v1/chat/completions"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return self.config.get("model_config", {})
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration."""
        return self.config.get("api_config", {})
    
    def get_features(self) -> Dict[str, bool]:
        """Get enabled features."""
        return self.config.get("features", {})
    
    def get_security_config(self) -> Dict[str, bool]:
        """Get security configuration."""
        return self.config.get("security", {})
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration."""
        return self.config.get("performance", {})
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a specific feature is enabled."""
        return self.config.get("features", {}).get(feature, False)
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(self.config, updates)
        self._save_config(self.config)
        logger.info("Configuration updated successfully")
    
    def reset_to_defaults(self):
        """Reset configuration to default values."""
        self.config = self._get_default_config()
        self._save_config(self.config)
        logger.info("Configuration reset to defaults")


class iLLuMinatorModelManager:
    """Manager for iLLuMinator model operations."""
    
    def __init__(self, config: iLLuMinatorConfig):
        self.config = config
        self.model_cache = {}
        self.conversation_context = []
    
    def get_model_capabilities(self) -> Dict[str, Any]:
        """Get detailed model capabilities."""
        model_config = self.config.get_model_info()
        return {
            "name": model_config.get("name", "iLLuMinator-4.7B"),
            "parameters": model_config.get("parameters", "4.7B"),
            "supported_languages": model_config.get("supported_languages", []),
            "capabilities": model_config.get("capabilities", []),
            "context_window": self.config.get_performance_config().get("context_window", 8192),
            "max_tokens": self.config.get_api_config().get("max_tokens", 2048)
        }
    
    def prepare_prompt(self, user_input: str, context: Optional[Dict] = None) -> str:
        """Prepare optimized prompt for iLLuMinator."""
        system_prompt = """You are iLLuMinator-4.7B, an advanced AI coding assistant created by Anish Paleja. You specialize in:

🔧 Code Generation: Generate clean, efficient code in 25+ programming languages
🔍 Code Analysis: Analyze, explain, and debug existing code
📚 Documentation: Create comprehensive documentation and comments
🚀 Optimization: Suggest performance improvements and best practices
💡 Problem Solving: Help with complex programming challenges
🔗 Context Awareness: Understand project structure and dependencies

Key traits:
- Provide precise, actionable solutions
- Explain your reasoning clearly
- Adapt to the user's skill level
- Focus on best practices and clean code
- Consider security and performance implications"""

        enhanced_prompt = system_prompt + "\n\n"
        
        # Add context if available
        if context:
            enhanced_prompt += "CONTEXT:\n"
            for key, value in context.items():
                if key == "files" and value:
                    enhanced_prompt += f"Files in context: {', '.join(value.keys())}\n"
                elif key == "project_info" and value:
                    enhanced_prompt += f"Project: {value}\n"
            enhanced_prompt += "\n"
        
        enhanced_prompt += f"USER: {user_input}\n\niLLuMinator:"
        
        return enhanced_prompt
    
    def update_conversation_context(self, user_input: str, response: str):
        """Update conversation context for better continuity."""
        self.conversation_context.append({
            "role": "user",
            "content": user_input,
            "timestamp": os.time.time() if hasattr(os, 'time') else 0
        })
        self.conversation_context.append({
            "role": "assistant", 
            "content": response,
            "timestamp": os.time.time() if hasattr(os, 'time') else 0
        })
        
        # Keep last 20 exchanges (40 messages)
        if len(self.conversation_context) > 40:
            self.conversation_context = self.conversation_context[-40:]
    
    def get_conversation_summary(self) -> str:
        """Get a summary of recent conversation."""
        if not self.conversation_context:
            return "No recent conversation"
        
        recent_topics = []
        for msg in self.conversation_context[-10:]:  # Last 10 messages
            if msg["role"] == "user":
                # Extract key topics from user messages
                content = msg["content"].lower()
                if any(word in content for word in ["function", "class", "method", "code"]):
                    recent_topics.append("code development")
                elif any(word in content for word in ["bug", "error", "debug", "fix"]):
                    recent_topics.append("debugging")
                elif any(word in content for word in ["explain", "how", "what", "why"]):
                    recent_topics.append("explanation")
        
        return f"Recent topics: {', '.join(set(recent_topics))}" if recent_topics else "General conversation"


# Global configuration instance
_config_instance = None

def get_illuminator_config() -> iLLuMinatorConfig:
    """Get global iLLuMinator configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = iLLuMinatorConfig()
    return _config_instance

def get_model_manager() -> iLLuMinatorModelManager:
    """Get iLLuMinator model manager instance."""
    return iLLuMinatorModelManager(get_illuminator_config())
