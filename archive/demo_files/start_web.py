#!/usr/bin/env python3
"""
Nexus CLI Web Interface Startup Script
Installs dependencies and starts the web server
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_status(message):
    print(f"\033[0;34m[INFO]\033[0m {message}")

def print_success(message):
    print(f"\033[0;32m[SUCCESS]\033[0m {message}")

def print_warning(message):
    print(f"\033[1;33m[WARNING]\033[0m {message}")

def print_error(message):
    print(f"\033[0;31m[ERROR]\033[0m {message}")

def run_command(command, capture_output=False):
    """Run a command and return success status"""
    try:
        if capture_output:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            return result.returncode == 0, result.stdout, result.stderr
        else:
            result = subprocess.run(command, shell=True)
            return result.returncode == 0, "", ""
    except Exception as e:
        print_error(f"Failed to run command '{command}': {e}")
        return False, "", str(e)

def check_mongodb():
    """Check if MongoDB is running"""
    print_status("Checking MongoDB connection...")
    
    # Try mongosh first (newer MongoDB client)
    success, _, _ = run_command("mongosh --eval \"db.adminCommand('ismaster')\" --quiet", capture_output=True)
    if success:
        print_success("MongoDB is running")
        return True
    
    # Try mongo (older MongoDB client)
    success, _, _ = run_command("mongo --eval \"db.adminCommand('ismaster')\" --quiet", capture_output=True)
    if success:
        print_success("MongoDB is running")
        return True
    
    print_warning("MongoDB is not running. Will use in-memory storage.")
    return False

def install_dependencies():
    """Install Python dependencies"""
    print_status("Installing dependencies...")
    
    # Upgrade pip first
    print_status("Upgrading pip...")
    success, _, _ = run_command(f"{sys.executable} -m pip install --upgrade pip")
    if not success:
        print_warning("Failed to upgrade pip")
    
    # Install requirements
    success, _, _ = run_command(f"{sys.executable} -m pip install -r requirements.txt")
    if success:
        print_success("Dependencies installed successfully")
        return True
    else:
        print_warning("Some dependencies failed to install. Continuing anyway...")
        return False

def start_server():
    """Start the web server"""
    print_status("Starting Nexus CLI Web Server...")
    print("")
    print("🌐 Web Interface will be available at: http://localhost:8000")
    print("📚 API Documentation will be available at: http://localhost:8000/docs")
    print("🔌 WebSocket endpoint: ws://localhost:8000/ws/{session_id}")
    print("")
    print("Press Ctrl+C to stop the server")
    print("")
    
    # Set PYTHONPATH
    current_dir = Path.cwd()
    os.environ["PYTHONPATH"] = str(current_dir) + os.pathsep + os.environ.get("PYTHONPATH", "")
    
    try:
        # Try to start with uvicorn module
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "web.backend.main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except FileNotFoundError:
        print_error("uvicorn not found. Please install it with: pip install uvicorn")
        sys.exit(1)
    except Exception as e:
        print_error(f"Failed to start server: {e}")
        sys.exit(1)

def main():
    """Main startup function"""
    print("🚀 Starting Nexus CLI Web Interface...")
    
    # Check if we're in the right directory
    if not Path("nexus_cli.py").exists():
        print_error("Please run this script from the Nexus-CLI directory")
        sys.exit(1)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print_error("Python 3.8 or higher is required")
        sys.exit(1)
    
    print_success(f"Python {sys.version.split()[0]} detected")
    print_success(f"Platform: {platform.system()} {platform.release()}")
    
    # Install dependencies
    install_dependencies()
    
    # Check MongoDB
    check_mongodb()
    
    # Start the server
    start_server()

if __name__ == "__main__":
    main()
