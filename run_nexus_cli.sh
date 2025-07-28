#!/bin/bash
echo "
╔══════════════════════════════════════════════════════════════╗
║                      NEXUS CLI                               ║
║              Intelligent Coding Assistant                    ║
║              Powered by iLLuMinator-4.7B                    ║
╚══════════════════════════════════════════════════════════════╝
"

if [ -f "dist/nexus-cli/nexus-cli" ]; then
    echo "Starting Nexus CLI..."
    ./dist/nexus-cli/nexus-cli
elif [ -f "dist/nexus-cli-standalone" ]; then
    echo "Starting Nexus CLI (standalone)..."
    ./dist/nexus-cli-standalone
else
    echo "ERROR: Nexus CLI executable not found!"
    echo "Please build the executable first by running: python build_executable.py"
    read -p "Press Enter to continue..."
fi
