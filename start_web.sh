#!/bin/bash

# Nexus CLI Web Interface Startup Script
# Installs dependencies and starts the web server

echo "🚀 Starting Nexus CLI Web Interface..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "nexus_cli.py" ]; then
    print_error "Please run this script from the Nexus-CLI directory"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -eq 0 ]; then
        print_success "Virtual environment created"
    else
        print_error "Failed to create virtual environment"
        exit 1
    fi
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
print_status "Installing dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    print_warning "Some dependencies failed to install. Continuing anyway..."
fi

# Check if MongoDB is running (optional)
print_status "Checking MongoDB connection..."
if command -v mongosh &> /dev/null; then
    if mongosh --eval "db.adminCommand('ismaster')" --quiet &> /dev/null; then
        print_success "MongoDB is running"
    else
        print_warning "MongoDB is not running. Will use in-memory storage."
    fi
elif command -v mongo &> /dev/null; then
    if mongo --eval "db.adminCommand('ismaster')" --quiet &> /dev/null; then
        print_success "MongoDB is running"
    else
        print_warning "MongoDB is not running. Will use in-memory storage."
    fi
else
    print_warning "MongoDB client not found. Will use in-memory storage."
fi

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Start the web server
print_status "Starting Nexus CLI Web Server..."
echo ""
echo "🌐 Web Interface will be available at: http://localhost:8000"
echo "📚 API Documentation will be available at: http://localhost:8000/docs"
echo "🔌 WebSocket endpoint: ws://localhost:8000/ws/{session_id}"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Try to start with uvicorn
if command -v uvicorn &> /dev/null; then
    uvicorn web.backend.main:app --host 0.0.0.0 --port 8000 --reload
else
    # Fallback to running with python
    python -m uvicorn web.backend.main:app --host 0.0.0.0 --port 8000 --reload
fi
