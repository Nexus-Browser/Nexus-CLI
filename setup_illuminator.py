#!/usr/bin/env python3
"""
Setup script for iLLuMinator-4.7B model
Downloads and sets up the model from the GitHub repository: https://github.com/Anipaleja/iLLuMinator-4.7B
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
console = Console()

class iLLuMinatorSetup:
    """Setup class for iLLuMinator-4.7B model"""
    
    def __init__(self):
        self.model_dir = Path("./model/nexus_model")
        self.github_repo = "Anipaleja/iLLuMinator-4.7B"
        self.model_files = [
            "config.json",
            "tokenizer.json", 
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "model.safetensors",
            "generation_config.json"
        ]
    
    def show_banner(self):
        """Display setup banner"""
        banner_text = """
[bold blue]iLLuMinator-4.7B Model Setup[/bold blue]
[dim]Repository: https://github.com/Anipaleja/iLLuMinator-4.7B[/dim]
[dim]Setting up local AI model for Nexus CLI[/dim]
        """
        console.print(Panel(banner_text, border_style="blue"))
    
    def install_dependencies(self):
        """Install required dependencies"""
        console.print("\n[yellow]Installing dependencies...[/yellow]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Installing Python packages...", total=None)
            
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
                ], capture_output=True)
                
                progress.update(task, description="✓ Dependencies installed successfully!")
                console.print("[green]✓ Dependencies installed successfully[/green]")
                return True
            except subprocess.CalledProcessError as e:
                progress.update(task, description="✗ Failed to install dependencies")
                console.print(f"[red]Failed to install dependencies: {e}[/red]")
                return False
    
    def create_model_directory(self):
        """Create model directory structure"""
        console.print("\n[yellow]Creating model directory...[/yellow]")
        
        try:
            self.model_dir.mkdir(parents=True, exist_ok=True)
            console.print(f"[green]✓ Model directory created: {self.model_dir}[/green]")
            return True
        except Exception as e:
            console.print(f"[red]Failed to create model directory: {e}[/red]")
            return False
    
    def download_model_from_hf(self):
        """Download model from Hugging Face Hub"""
        console.print("\n[yellow]Downloading iLLuMinator-4.7B model...[/yellow]")
        console.print("[dim]This may take several minutes depending on your internet connection[/dim]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Downloading model files...", total=None)
            
            try:
                # Try to import huggingface_hub
                try:
                    from huggingface_hub import snapshot_download
                except ImportError:
                    progress.update(task, description="Installing huggingface_hub...")
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", "huggingface_hub"
                    ])
                    from huggingface_hub import snapshot_download
                
                progress.update(task, description="Downloading iLLuMinator-4.7B...")
                
                snapshot_download(
                    repo_id=self.github_repo,
                    local_dir=str(self.model_dir),
                    token=None,  # Public model
                    resume_download=True
                )
                
                progress.update(task, description="✓ Model downloaded successfully!")
                console.print("[green]✓ Model downloaded successfully[/green]")
                return True
                
            except Exception as e:
                progress.update(task, description="✗ Failed to download model")
                console.print(f"[red]Failed to download model: {e}[/red]")
                return False
    
    def verify_model_files(self):
        """Verify that all required model files are present"""
        console.print("\n[yellow]Verifying model files...[/yellow]")
        
        missing_files = []
        present_files = []
        
        for file in self.model_files:
            file_path = self.model_dir / file
            if file_path.exists():
                present_files.append(file)
            else:
                missing_files.append(file)
        
        # Show status
        for file in present_files:
            console.print(f"[green]✓ {file}[/green]")
        
        for file in missing_files:
            console.print(f"[red]✗ {file}[/red]")
        
        if missing_files:
            console.print(f"[yellow]Warning: {len(missing_files)} files missing[/yellow]")
            return False
        else:
            console.print("[green]✓ All required model files present[/green]")
            return True
    
    def test_model_loading(self):
        """Test if the model can be loaded"""
        console.print("\n[yellow]Testing model loading...[/yellow]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Loading iLLuMinator-4.7B model...", total=None)
            
            try:
                from model.illuminator_api import iLLuMinatorAPI
                
                # Try to load the model
                model = iLLuMinatorAPI(str(self.model_dir))
                
                if model.is_available():
                    progress.update(task, description="Testing model generation...")
                    
                    # Test generation
                    test_response = model.generate_response("Hello", max_length=10, temperature=0.5)
                    if test_response and len(test_response.strip()) > 0:
                        progress.update(task, description="✓ Model test successful!")
                        console.print("[green]✓ Model loaded and tested successfully[/green]")
                        return True
                    else:
                        progress.update(task, description="✗ Model generation failed")
                        console.print("[red]Model loaded but generation test failed[/red]")
                        return False
                else:
                    progress.update(task, description="✗ Model failed to load")
                    console.print("[red]Model failed to load[/red]")
                    return False
                    
            except Exception as e:
                progress.update(task, description="✗ Model loading test failed")
                console.print(f"[red]Model loading test failed: {e}[/red]")
                return False
    
    def create_model_info_file(self):
        """Create model info file"""
        console.print("\n[yellow]Creating model configuration...[/yellow]")
        
        model_info = {
            "name": "iLLuMinator-4.7B",
            "version": "4.7B", 
            "repository": f"https://github.com/{self.github_repo}",
            "huggingface": f"https://huggingface.co/{self.github_repo}",
            "model_path": str(self.model_dir),
            "setup_complete": True,
            "local_model": True
        }
        
        try:
            import json
            with open(self.model_dir / "model_info.json", "w") as f:
                json.dump(model_info, f, indent=2)
            
            console.print("[green]✓ Model configuration created[/green]")
            return True
        except Exception as e:
            console.print(f"[red]Failed to create model configuration: {e}[/red]")
            return False
    
    def run_setup(self):
        """Run the complete setup process"""
        self.show_banner()
        
        # Ask user confirmation
        if not Confirm.ask("\n[cyan]Do you want to set up iLLuMinator-4.7B model?[/cyan]"):
            console.print("[yellow]Setup cancelled.[/yellow]")
            return False
        
        steps = [
            ("Installing dependencies", self.install_dependencies),
            ("Creating model directory", self.create_model_directory), 
            ("Downloading model", self.download_model_from_hf),
            ("Verifying model files", self.verify_model_files),
            ("Testing model loading", self.test_model_loading),
            ("Creating configuration", self.create_model_info_file)
        ]
        
        for i, (step_name, step_func) in enumerate(steps, 1):
            console.print(f"\n[bold]Step {i}/{len(steps)}: {step_name}[/bold]")
            if not step_func():
                console.print(f"[red]Setup failed at step: {step_name}[/red]")
                return False
        
        # Success message
        success_text = """
[bold green]🎉 Setup Completed Successfully![/bold green]

[green]✓ iLLuMinator-4.7B model installed[/green]
[green]✓ Model tested and working[/green]
[green]✓ Configuration complete[/green]

[bold]Model Location:[/bold] {model_dir}
[bold]Repository:[/bold] https://github.com/Anipaleja/iLLuMinator-4.7B

[cyan]To start using the CLI:[/cyan]
[bold]python nexus_cli.py[/bold]
        """.format(model_dir=self.model_dir)
        
        console.print(Panel(success_text, border_style="green"))
        return True

def main():
    """Main setup function"""
    setup = iLLuMinatorSetup()
    
    try:
        success = setup.run_setup()
        if not success:
            console.print("\n[red]Setup incomplete. Please resolve issues and try again.[/red]")
            sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Setup cancelled by user.[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Unexpected error during setup: {e}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main()

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
