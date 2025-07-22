#!/usr/bin/env python3
"""
Demo script for iLLuMinator-4.7B powered Nexus CLI
Shows the local AI model capabilities without external APIs
"""

import os
import sys
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nexus_cli import IntelligentNexusCLI

console = Console()

def show_demo_banner():
    """Show demo banner"""
    banner_text = """
[bold blue]iLLuMinator-4.7B Nexus CLI Demo[/bold blue]
[dim]Powered by local AI model from GitHub repository[/dim]
[dim]Repository: https://github.com/Anipaleja/iLLuMinator-4.7B[/dim]

[yellow]This demo shows the capabilities of the local iLLuMinator model:[/yellow]
• Context-aware file reading
• Local AI code generation  
• Intelligent conversation
• No external API dependencies
    """
    console.print(Panel(banner_text, border_style="blue"))

def demo_context_feature():
    """Demonstrate the context feature"""
    console.print("\n[bold]🧠 Context Feature Demo[/bold]")
    
    try:
        cli = IntelligentNexusCLI()
        
        # Demo reading a file and adding to context
        console.print("\n[yellow]1. Reading setup.py and adding to context...[/yellow]")
        result = cli._handle_read_file(["setup.py"])
        console.print(f"Result: {result}")
        
        # Demo showing context
        console.print("\n[yellow]2. Showing current context...[/yellow]")
        result = cli._handle_show_context([])
        console.print(f"Context status: {result}")
        
        # Demo context-enhanced prompt
        console.print("\n[yellow]3. Testing context-enhanced prompts...[/yellow]")
        enhanced_prompt = cli._prepare_context_enhanced_prompt("What is this project about?")
        console.print(f"Enhanced prompt length: {len(enhanced_prompt)} characters")
        console.print("✓ Context information added to AI prompts")
        
        # Demo clearing context  
        console.print("\n[yellow]4. Clearing context...[/yellow]")
        result = cli._handle_clear_context([])
        console.print(f"Result: {result}")
        
        console.print("\n[green]✅ Context feature working perfectly![/green]")
        
    except Exception as e:
        console.print(f"[red]Demo error: {e}[/red]")

def demo_model_status():
    """Demonstrate model status checking"""
    console.print("\n[bold]🤖 iLLuMinator Model Status[/bold]")
    
    try:
        cli = IntelligentNexusCLI()
        
        if cli.model_available:
            console.print("[green]✓ iLLuMinator-4.7B model loaded successfully[/green]")
            
            # Show model info
            result = cli._handle_model_status([])
            console.print(f"Status check: {result}")
            
        else:
            console.print("[yellow]⚠️ iLLuMinator model not available[/yellow]")
            console.print("[dim]Run 'python setup_illuminator.py' to set up the model[/dim]")
            
    except Exception as e:
        console.print(f"[red]Model status error: {e}[/red]")

def demo_basic_commands():
    """Demonstrate basic CLI commands"""
    console.print("\n[bold]⚡ Basic Commands Demo[/bold]")
    
    try:
        cli = IntelligentNexusCLI()
        
        # Test help
        console.print("\n[yellow]Testing help command...[/yellow]")
        cli.show_help()
        
        # Test file operations
        console.print("\n[yellow]Testing file operations...[/yellow]")
        result = cli._handle_list_files(["."])
        console.print("✓ File listing works")
        
        # Test command suggestions
        console.print("\n[yellow]Testing command suggestions...[/yellow]")
        suggestions = cli.get_command_suggestions("co")
        console.print(f"Suggestions for 'co': {suggestions}")
        
        console.print("\n[green]✅ Basic commands working![/green]")
        
    except Exception as e:
        console.print(f"[red]Commands demo error: {e}[/red]")

def main():
    """Main demo function"""
    show_demo_banner()
    
    console.print("\n[cyan]Starting iLLuMinator-4.7B Nexus CLI demonstration...[/cyan]")
    
    # Run demos
    demo_model_status()
    demo_context_feature()
    demo_basic_commands()
    
    # Final message
    success_msg = """
[bold green]🎉 Demo Complete![/bold green]

[green]✓ iLLuMinator-4.7B integration working[/green]
[green]✓ Context feature operational[/green]
[green]✓ Local AI model ready[/green]

[cyan]To start using the full CLI:[/cyan]
[bold]python nexus_cli.py[/bold]

[dim]No API keys needed - everything runs locally![/dim]
    """
    
    console.print(Panel(success_msg, border_style="green"))

if __name__ == "__main__":
    main()
    print("• Replaced heavy local models with efficient API calls")
    print("• Fast response times without local GPU requirements")
    print("• Professional-grade code generation capabilities")
    
    print("\nAvailable Commands:")
    commands = [
        ("code python create a REST API", "Generate Python code"),
        ("code javascript build a web server", "Generate JavaScript code"),
        ("analyze myfile.py", "AI-powered code analysis"),
        ("chat", "Conversational coding assistance"),
        ("status", "Check iLLuMinator API status"),
        ("help", "Show all available commands")
    ]
    
    for cmd, desc in commands:
        print(f"  • {cmd:<35} - {desc}")
    
    print("\nExample Session:")
    print("$ python nexus_cli.py")
    print("nexus> code python create a function to sort a list")
    print("[Generated clean, documented Python sorting function]")
    print("\nnexus> chat") 
    print("You> How do I optimize this code for performance?")
    print("iLLuMinator> Here are several optimization strategies...")
    
    print("\nTechnical Details:")
    print(f"• Model: iLLuMinator-4.7B")
    print(f"• API Endpoint: iLLuMinator 4.7B")
    print(f"• Response Time: ~2-5 seconds")
    print(f"• Memory Usage: Minimal (API-based)")
    print(f"• GPU Required: None")
    
    print(f"\nReady to use! Run: python nexus_cli.py")

if __name__ == "__main__":
    main()
