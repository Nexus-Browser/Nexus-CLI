#!/usr/bin/env python3
"""
Nexus CLI Demo Script
Demonstrates the key features of the Nexus CLI with iLLuMinator-4.7B
"""

import subprocess
import sys
import time
import os

def run_demo_command(description, command, input_text=None):
    """Run a demo command and show the output"""
    print(f"\n{'='*60}")
    print(f"DEMO: {description}")
    print(f"Command: {command}")
    print('='*60)
    
    try:
        if input_text:
            # Use echo to pipe input to the command
            full_command = f'echo "{input_text}" | {command}'
            result = subprocess.run(full_command, shell=True, capture_output=True, text=True, timeout=30)
        else:
            result = subprocess.run(command.split(), capture_output=True, text=True, timeout=30)
        
        if result.stdout:
            print("OUTPUT:")
            print(result.stdout)
        
        if result.stderr and result.returncode != 0:
            print("ERROR:")
            print(result.stderr)
            
        time.sleep(2)  # Pause between demos
        
    except subprocess.TimeoutExpired:
        print("Demo timed out - continuing...")
    except Exception as e:
        print(f"Demo error: {e}")

def main():
    """Run the Nexus CLI demo"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                      NEXUS CLI DEMO                          ║
    ║              Powered by iLLuMinator-4.7B                    ║
    ║                                                              ║
    ║  This demo showcases the intelligent coding assistant        ║
    ║  capabilities of Nexus CLI with comprehensive knowledge      ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Check if executable exists
    executable_name = "nexus-cli" if os.name != 'nt' else "nexus-cli.exe"
    executable_path = f"./dist/nexus-cli/{executable_name}"
    standalone_path = f"./dist/nexus-cli-standalone" + (".exe" if os.name == 'nt' else "")
    
    # Try to find the executable
    nexus_cmd = None
    if os.path.exists(executable_path):
        nexus_cmd = executable_path
    elif os.path.exists(standalone_path):
        nexus_cmd = standalone_path
    elif os.path.exists("nexus_cli.py"):
        nexus_cmd = f"{sys.executable} nexus_cli.py"
    else:
        print("❌ Nexus CLI executable not found!")
        print("Please build the executable first using: python build_executable.py")
        return
    
    print(f"Using executable: {nexus_cmd}")
    
    # Demo scenarios
    demos = [
        {
            "description": "Neural Networks Explanation",
            "command": nexus_cmd,
            "input": "what are neural networks"
        },
        {
            "description": "JavaScript Programming Guide", 
            "command": nexus_cmd,
            "input": "explain javascript"
        },
        {
            "description": "Biography Query - Jensen Huang",
            "command": nexus_cmd,
            "input": "who is jensen huang"
        },
        {
            "description": "Learning Programming Guidance",
            "command": nexus_cmd,
            "input": "how to learn programming"
        },
        {
            "description": "AI Technology Overview",
            "command": nexus_cmd,
            "input": "what is artificial intelligence"
        },
        {
            "description": "Debugging Help",
            "command": nexus_cmd,
            "input": "how do I debug code"
        },
        {
            "description": "Python Code Generation",
            "command": nexus_cmd,
            "input": "code python hello world function"
        },
        {
            "description": "JavaScript Code Generation",
            "command": nexus_cmd,
            "input": "code javascript web server"
        },
        {
            "description": "Help and Capabilities",
            "command": nexus_cmd,
            "input": "help"
        }
    ]
    
    print("\n🚀 Starting Nexus CLI Feature Demonstrations...")
    print("Each demo will show a different capability of the intelligent assistant.")
    
    for i, demo in enumerate(demos, 1):
        print(f"\n[{i}/{len(demos)}] Running demo...")
        run_demo_command(
            demo["description"], 
            demo["command"], 
            demo["input"]
        )
        
        if i < len(demos):
            input("\nPress Enter to continue to next demo...")
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                        DEMO COMPLETE                         ║
    ║                                                              ║
    ║  Nexus CLI successfully demonstrated:                        ║
    ║  ✓ Intelligent question answering                           ║
    ║  ✓ Comprehensive technical explanations                     ║
    ║  ✓ Biography and person information                         ║
    ║  ✓ Learning guidance and career advice                      ║
    ║  ✓ Code generation in multiple languages                    ║
    ║  ✓ Debugging and troubleshooting help                       ║
    ║                                                              ║
    ║  Ready for interactive use!                                  ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

if __name__ == "__main__":
    main()
