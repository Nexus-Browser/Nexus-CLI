#!/usr/bin/env python3
"""
iLLuMinator LLM Setup Script
Interactive setup for configuring iLLuMinator-4.7B with Nexus CLI
"""

import os
import json
import sys
import subprocess
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.text import Text
from rich.align import Align

console = Console()

def show_banner():
    """Display the setup banner."""
    banner = Text.assemble(
        ("🤖 ", "bright_blue"),
        ("iLLuMinator-4.7B", "bright_white bold"),
        (" Setup Wizard", "bright_blue"),
        ("\n", ""),
        ("Advanced AI Integration for Nexus CLI", "dim")
    )
    
    panel = Panel(
        Align.center(banner),
        border_style="bright_blue",
        padding=(1, 2)
    )
    
    console.print("\n")
    console.print(panel)
    console.print("\n")

def check_dependencies():
    """Check if required dependencies are installed."""
    console.print("[yellow]Checking dependencies...[/yellow]")
    
    required_packages = [
        "rich",
        "requests", 
        "cryptography",
        "transformers"
    ]
    
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
            console.print(f"  ✓ {package}")
        except ImportError:
            console.print(f"  ✗ {package} [red](missing)[/red]")
            missing.append(package)
    
    if missing:
        console.print(f"\n[red]Missing dependencies: {', '.join(missing)}[/red]")
        if Confirm.ask("Install missing dependencies?"):
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
            console.print("[green]Dependencies installed successfully![/green]")
        else:
            console.print("[red]Cannot continue without dependencies.[/red]")
            sys.exit(1)
    else:
        console.print("[green]All dependencies satisfied![/green]")

def get_api_provider():
    """Interactive API provider selection."""
    console.print("\n[bold]Available AI API Providers:[/bold]")
    
    providers = [
        ("OpenAI", "gpt-3.5-turbo, gpt-4", "Pay-per-use", "High quality"),
        ("Anthropic", "Claude-3-Haiku", "Free tier + pay-per-use", "Very high quality"),
        ("Cohere", "command-light", "Generous free tier", "Good quality"),
        ("Google Gemini", "gemini-1.5-flash", "Free (50 requests/day)", "Fast and free"),
        ("Groq", "llama3-8b-8192", "Limited free tier", "Very fast"),
        ("Together AI", "Llama-2-7b-chat", "$25 free credit", "Good free option"),
        ("Hugging Face", "Various models", "Completely free", "Open source"),
        ("Local Mode", "Basic responses", "Always free", "Limited capabilities")
    ]
    
    table = Table(title="AI Provider Options")
    table.add_column("Provider", style="cyan")
    table.add_column("Models", style="green")
    table.add_column("Pricing", style="yellow")
    table.add_column("Quality", style="magenta")
    
    for provider, models, pricing, quality in providers:
        table.add_row(provider, models, pricing, quality)
    
    console.print(table)
    
    provider_choice = Prompt.ask(
        "\nWhich provider would you like to use?",
        choices=["openai", "anthropic", "cohere", "gemini", "groq", "together", "huggingface", "local"],
        default="local"
    )
    
    return provider_choice

def configure_api_key(provider):
    """Configure API key for chosen provider."""
    if provider == "local":
        console.print("[yellow]Local mode selected - no API key required.[/yellow]")
        return None
    
    console.print(f"\n[bold]Configuring {provider.title()} API[/bold]")
    
    setup_instructions = {
        "openai": "1. Visit https://platform.openai.com/api-keys\n2. Create account and generate API key\n3. Key starts with 'sk-'",
        "anthropic": "1. Visit https://console.anthropic.com\n2. Create account and generate API key\n3. Key starts with 'sk-ant-'",
        "cohere": "1. Visit https://dashboard.cohere.com/api-keys\n2. Create account and generate API key\n3. Key is 40 characters long",
        "gemini": "1. Visit https://aistudio.google.com/app/apikey\n2. Create Google account and generate API key\n3. Key starts with 'AIza'",
        "groq": "1. Visit https://console.groq.com/keys\n2. Create account and generate API key\n3. Key starts with 'gsk_'",
        "together": "1. Visit https://api.together.ai\n2. Sign up for $25 free credit\n3. Key starts with 'together_'",
        "huggingface": "1. Visit https://huggingface.co/settings/tokens\n2. Create account and generate token\n3. Key starts with 'hf_'"
    }
    
    if provider in setup_instructions:
        console.print(f"[dim]{setup_instructions[provider]}[/dim]")
    
    if Confirm.ask(f"\nDo you have a {provider.title()} API key?"):
        api_key = Prompt.ask("Enter your API key", password=True)
        
        # Basic validation
        expected_prefixes = {
            "openai": ["sk-"],
            "anthropic": ["sk-ant-"],
            "cohere": [],  # 40 char length check
            "gemini": ["AIza"],
            "groq": ["gsk_"],
            "together": ["together_"],
            "huggingface": ["hf_"]
        }
        
        if provider == "cohere" and len(api_key) != 40:
            console.print("[red]Warning: Cohere API keys are typically 40 characters long.[/red]")
        elif provider in expected_prefixes and expected_prefixes[provider]:
            if not any(api_key.startswith(prefix) for prefix in expected_prefixes[provider]):
                console.print(f"[red]Warning: {provider.title()} API keys typically start with {expected_prefixes[provider]}[/red]")
        
        return api_key
    else:
        console.print(f"[yellow]You can set up the API key later using environment variables.[/yellow]")
        console.print(f"[dim]export {provider.upper()}_API_KEY='your_key_here'[/dim]")
        return None

def setup_environment(provider, api_key):
    """Set up environment configuration."""
    console.print("\n[bold]Setting up environment...[/bold]")
    
    # Create .env file for local development
    env_file = Path(".env")
    env_content = []
    
    if env_file.exists():
        with open(env_file, 'r') as f:
            env_content = f.readlines()
    
    # Remove existing API key lines
    env_content = [line for line in env_content if not any(
        key in line for key in ['_API_KEY=', 'ILLUMINATOR_API_KEY=']
    )]
    
    if api_key and provider != "local":
        env_var = f"{provider.upper()}_API_KEY"
        env_content.append(f"{env_var}={api_key}\n")
        env_content.append(f"ILLUMINATOR_API_KEY={api_key}\n")
    
    # Write updated .env file
    with open(env_file, 'w') as f:
        f.writelines(env_content)
    
    console.print("[green]✓ Environment configured[/green]")
    
    # Add to shell profile for permanent setup
    if api_key and provider != "local":
        shell_config = None
        if os.path.exists(os.path.expanduser("~/.zshrc")):
            shell_config = "~/.zshrc"
        elif os.path.exists(os.path.expanduser("~/.bashrc")):
            shell_config = "~/.bashrc"
        elif os.path.exists(os.path.expanduser("~/.bash_profile")):
            shell_config = "~/.bash_profile"
        
        if shell_config and Confirm.ask(f"Add API key to {shell_config} for permanent setup?"):
            env_var = f"{provider.upper()}_API_KEY"
            export_line = f'export {env_var}="{api_key}"\n'
            
            with open(os.path.expanduser(shell_config), 'a') as f:
                f.write(f"\n# iLLuMinator API Key\n{export_line}")
            
            console.print(f"[green]✓ Added to {shell_config}[/green]")
            console.print("[yellow]Restart your terminal or run 'source ~/.zshrc' to apply changes[/yellow]")

def test_configuration():
    """Test the iLLuMinator configuration."""
    console.print("\n[bold]Testing configuration...[/bold]")
    
    try:
        # Import and test the model
        from model.nexus_model import NexusModel
        
        console.print("  ✓ Imports successful")
        
        # Initialize model
        model = NexusModel()
        console.print("  ✓ Model initialization")
        
        # Test API connection
        if model.is_available():
            console.print("  ✓ API connection successful")
            
            # Test code generation
            test_code = model.generate_code("create a hello world function", "python")
            if test_code and not test_code.startswith("[Error]"):
                console.print("  ✓ Code generation working")
            else:
                console.print("  ⚠ Code generation limited")
        else:
            console.print("  ⚠ API connection failed - using local mode")
        
        console.print("[green]Configuration test completed![/green]")
        return True
        
    except Exception as e:
        console.print(f"[red]Configuration test failed: {e}[/red]")
        return False

def create_startup_script():
    """Create a convenient startup script."""
    startup_script = """#!/bin/bash
# iLLuMinator Nexus CLI Startup Script

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d "nexus_venv" ]; then
    source nexus_venv/bin/activate
fi

# Start Nexus CLI
python nexus_cli.py "$@"
"""
    
    script_path = Path("start_nexus.sh")
    with open(script_path, 'w') as f:
        f.write(startup_script)
    
    # Make executable
    os.chmod(script_path, 0o755)
    
    console.print(f"[green]✓ Created startup script: {script_path}[/green]")
    console.print("[dim]Usage: ./start_nexus.sh[/dim]")

def main():
    """Main setup function."""
    show_banner()
    
    console.print("Welcome to the iLLuMinator-4.7B setup wizard!")
    console.print("This will help you configure your AI-powered coding assistant.\n")
    
    # Check dependencies
    check_dependencies()
    
    # Choose API provider
    provider = get_api_provider()
    
    # Configure API key
    api_key = configure_api_key(provider)
    
    # Setup environment
    setup_environment(provider, api_key)
    
    # Test configuration
    test_success = test_configuration()
    
    # Create startup script
    create_startup_script()
    
    # Final summary
    console.print("\n" + "="*60)
    console.print("[bold green]🎉 Setup Complete![/bold green]")
    console.print(f"Provider: {provider.title()}")
    console.print(f"API Key: {'✓ Configured' if api_key else '⚠ Not set (using local mode)'}")
    console.print(f"Test: {'✓ Passed' if test_success else '⚠ Limited functionality'}")
    
    console.print("\n[bold]Next steps:[/bold]")
    console.print("1. Run: ./start_nexus.sh")
    console.print("2. Try: 'code create a calculator function'")
    console.print("3. Explore: 'help' for all commands")
    
    if provider != "local" and not api_key:
        console.print("\n[yellow]To enable full AI features, set your API key:[/yellow]")
        console.print(f"export {provider.upper()}_API_KEY='your_key_here'")
    
    console.print("\nEnjoy coding with iLLuMinator-4.7B! 🚀")

if __name__ == "__main__":
    main()
