# Nexus CLI Web Interface

A modern, Warp-inspired web terminal interface for Nexus CLI powered by the iLLuMinator-4.7B AI model.

![Nexus CLI Web Interface](https://img.shields.io/badge/Nexus-CLI-blue?style=for-the-badge&logo=terminal)
![Python](https://img.shields.io/badge/Python-3.8+-green?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-teal?style=for-the-badge&logo=fastapi)
![WebSocket](https://img.shields.io/badge/WebSocket-Enabled-orange?style=for-the-badge)

## Features

### 🎨 **Warp-Inspired Design**
- Sleek, modern terminal interface
- Dark theme with beautiful syntax highlighting
- Smooth animations and transitions
- Responsive design for all screen sizes

### 🤖 **AI-Powered Development**
- **iLLuMinator-4.7B** AI model integration
- Intelligent code generation
- Real-time chat assistance
- Code analysis and optimization
- Context-aware suggestions

### ⚡ **Advanced Terminal Features**
- Command completion with Tab
- Command history with arrow keys
- Multiple tabs and sessions
- WebSocket real-time communication
- Syntax highlighting for code output

### 🔧 **Developer Tools**
- **Terminal View**: Full-featured command-line interface
- **Chat View**: Conversational AI assistance
- **Code Generation**: Prompt-to-code generation
- **Command Palette**: Quick actions with Ctrl+K
- **Session Management**: Persistent conversation history

### 📊 **Database Integration**
- MongoDB for session persistence
- In-memory fallback when MongoDB unavailable
- Command history and analytics
- User session management

## Quick Start

### Option 1: Use the Shell Script (macOS/Linux)
```bash
./start_web.sh
```

### Option 2: Use the Python Script (Cross-platform)
```bash
python start_web.py
```

### Option 3: Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn web.backend.main:app --host 0.0.0.0 --port 8000 --reload
```

## Usage

Once the server is running, open your browser and navigate to:
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### Terminal Commands

The web terminal supports all Nexus CLI commands:

```bash
# Basic commands
help                    # Show help information
generate [description]  # Generate code
chat [message]         # Chat with AI
analyze [file]         # Analyze code
create [filename]      # Create new file
list                   # List files
clear                  # Clear terminal

# Examples
generate "Create a REST API for user management"
chat "How do I optimize this Python function?"
analyze main.py
create new_project.py
```

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Tab` | Command completion |
| `↑/↓` | Navigate command history |
| `Ctrl+K` | Open command palette |
| `Ctrl+L` | Clear terminal |
| `Ctrl+C` | Interrupt command |
| `Ctrl+G` | Quick code generation |
| `Ctrl+T` | Switch to chat |
| `Ctrl+A` | Analyze code |
| `Enter` | Execute command (Terminal) |
| `Shift+Enter` | New line (Chat) |

### Views

#### 1. Terminal View
- Full terminal experience with command execution
- Real-time output streaming
- Command history and completion
- Syntax highlighting for code

#### 2. Chat View  
- Conversational interface with iLLuMinator-4.7B
- Markdown support in messages
- Persistent chat history
- Code snippets and explanations

#### 3. Code Generation View
- Prompt-based code generation
- Multiple language support
- Syntax highlighting
- Copy and save generated code

## API Endpoints

### REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Main web interface |
| `GET` | `/health` | Health check |
| `POST` | `/api/session/create` | Create new session |
| `GET` | `/api/session/{session_id}` | Get session info |
| `POST` | `/api/command` | Execute command |
| `POST` | `/api/chat` | Send chat message |
| `POST` | `/api/code/generate` | Generate code |
| `POST` | `/api/analyze` | Analyze code |

### WebSocket

Connect to `ws://localhost:8000/ws/{session_id}` for real-time communication.

**Message Types:**
- `command_result` - Command execution results
- `chat_response` - AI chat responses  
- `code_generated` - Generated code
- `status_update` - System status updates

### Request Examples

#### Generate Code
```bash
curl -X POST "http://localhost:8000/api/code/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Create a Python function to calculate fibonacci numbers",
    "language": "python",
    "session_id": "your-session-id"
  }'
```

#### Chat with AI
```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Explain how async/await works in Python",
    "session_id": "your-session-id"
  }'
```

## Architecture

```
web/
├── backend/
│   ├── main.py          # FastAPI application
│   └── database.py      # MongoDB integration
└── frontend/
    ├── index.html       # Main HTML
    ├── style.css        # Warp-inspired styling
    ├── app.js           # Main application logic
    └── terminal.js      # Terminal enhancements
```

### Technologies Used

- **Backend**: FastAPI, WebSockets, Motor (MongoDB), Pydantic
- **Frontend**: Vanilla JavaScript, HTML5, CSS3
- **AI**: iLLuMinator-4.7B (Gemini-powered)
- **Database**: MongoDB (with in-memory fallback)
- **Styling**: Custom CSS inspired by Warp terminal

## Configuration

### Environment Variables

```bash
# MongoDB connection (optional)
MONGODB_URL=mongodb://localhost:27017/nexus_cli

# API configuration  
NEXUS_API_HOST=0.0.0.0
NEXUS_API_PORT=8000

# Development mode
NEXUS_DEBUG=true
```

### MongoDB Setup (Optional)

If you want persistent sessions and history:

```bash
# Install MongoDB (macOS with Homebrew)
brew install mongodb-community

# Start MongoDB
brew services start mongodb-community

# Or use Docker
docker run -d -p 27017:27017 --name nexus-mongo mongo
```

## Development

### Project Structure
```
Nexus-CLI/
├── nexus_cli.py         # Main CLI application
├── model/
│   ├── nexus_model.py   # AI model interface
│   └── illuminator_api.py # iLLuMinator API
├── web/
│   ├── backend/         # FastAPI backend
│   └── frontend/        # Web interface
├── requirements.txt     # Dependencies
├── start_web.sh        # Startup script (bash)
└── start_web.py        # Startup script (Python)
```

### Adding New Features

1. **Backend**: Add new endpoints in `web/backend/main.py`
2. **Frontend**: Extend functionality in `web/frontend/app.js`
3. **Terminal**: Enhance terminal features in `web/frontend/terminal.js`
4. **Styling**: Modify appearance in `web/frontend/style.css`

### Testing

```bash
# Test the API endpoints
curl http://localhost:8000/health

# Test WebSocket connection
wscat -c ws://localhost:8000/ws/test-session

# Run the CLI directly
python nexus_cli.py
```

## Troubleshooting

### Common Issues

**Port 8000 already in use:**
```bash
# Find and kill the process
lsof -ti:8000 | xargs kill -9

# Or use a different port
uvicorn web.backend.main:app --port 8001
```

**Dependencies not installing:**
```bash
# Create fresh virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

**MongoDB connection issues:**
- The application will work without MongoDB using in-memory storage
- Check if MongoDB is running: `brew services list | grep mongodb`
- Verify connection: `mongosh` or `mongo`

**Module import errors:**
```bash
# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or run from the project root
cd /path/to/Nexus-CLI
python start_web.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is part of Nexus CLI and follows the same licensing terms.

## Acknowledgments

- Inspired by [Warp Terminal](https://warp.dev/) design
- Powered by iLLuMinator-4.7B AI model
- Built with FastAPI and modern web technologies
