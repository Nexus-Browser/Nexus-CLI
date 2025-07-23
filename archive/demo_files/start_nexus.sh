#!/bin/bash
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
