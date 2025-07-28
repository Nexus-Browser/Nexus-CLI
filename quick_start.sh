#!/bin/bash
# Quick Start Script for Enhanced Nexus CLI

echo "Enhanced Nexus CLI - Quick Start"
echo "=================================="

# Check Python version
python_version=$(python3 --version 2>&1)
echo "Python: $python_version"

# Check current directory
echo "Directory: $(pwd)"

# Check if setup has been run
if [ ! -f "model_config.json" ]; then
    echo ""
    echo "First time setup required!"
    echo "Run: python setup_nexus.py"
    echo ""
    exit 1
fi

echo ""
echo "  Available commands:"
echo "  python nexus.py --interactive     # Start interactive mode"
echo "  python nexus.py 'your question'   # Single query"
echo "  python demo_integration.py        # See integration demo"
echo "  python train_nexus.py --help      # Training options"
echo ""

# Check for dependencies
if python3 -c "import torch" 2>/dev/null; then
    echo "PyTorch available"
else
    echo "PyTorch not installed - run setup_nexus.py"
fi

echo ""
echo "Quick test:"
echo "python nexus.py 'Hello Nexus, are you working?'"
echo ""
