"""
Nexus CLI Web Backend
FastAPI server with MongoDB integration for web-based terminal interface
Designed to replicate Warp's sleek terminal experience
"""

import asyncio
import json
import os
import sys
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add the parent directory to sys.path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("Warning: FastAPI dependencies not available. Please install them.")

from web.backend.database import MongoDB
from model.nexus_model import NexusModel

# Initialize components
if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="Nexus CLI Web API",
        description="Web interface for Nexus CLI with iLLuMinator-4.7B AI",
        version="1.0.0"
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, specify exact origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount static files (frontend)
    frontend_dir = Path(__file__).parent.parent / "frontend"
    if frontend_dir.exists():
        app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")
    
    # Initialize database and model
    db = MongoDB()
    nexus_model = NexusModel()
    
    # WebSocket connection manager
    class ConnectionManager:
        def __init__(self):
            self.active_connections: Dict[str, WebSocket] = {}

        async def connect(self, websocket: WebSocket, session_id: str):
            await websocket.accept()
            self.active_connections[session_id] = websocket
            print(f"🔌 WebSocket connected: {session_id}")

        def disconnect(self, session_id: str):
            if session_id in self.active_connections:
                del self.active_connections[session_id]
                print(f"🔌 WebSocket disconnected: {session_id}")

        async def send_personal_message(self, message: dict, session_id: str):
            if session_id in self.active_connections:
                try:
                    websocket = self.active_connections[session_id]
                    await websocket.send_json(message)
                except Exception as e:
                    print(f"Failed to send message to {session_id}: {e}")
                    self.disconnect(session_id)

    manager = ConnectionManager()

# Request/Response models
class CodeGenerationRequest(BaseModel):
    instruction: Optional[str] = None
    prompt: Optional[str] = None  # Alternative field for frontend compatibility
    language: str = "python"
    session_id: Optional[str] = None

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    max_length: int = 256
    temperature: float = 0.7

class FileAnalysisRequest(BaseModel):
    code: str
    filename: Optional[str] = None
    session_id: Optional[str] = None

class CommandRequest(BaseModel):
    command: str
    session_id: Optional[str] = None

class SessionResponse(BaseModel):
    session_id: str
    timestamp: datetime
    status: str

class UserInfo(BaseModel):
    id: str
    email: str
    name: str
    picture: Optional[str] = None

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web interface"""
    try:
        frontend_dir = Path(__file__).parent.parent / "frontend"
        html_file = frontend_dir / "index.html"
        if html_file.exists():
            return FileResponse(html_file)
        else:
            return HTMLResponse("""
            <html>
                <body>
                    <h1>Nexus CLI Web Terminal</h1>
                    <p>Frontend files not found. Please ensure the frontend is built.</p>
                    <p>API documentation is available at <a href="/docs">/docs</a></p>
                </body>
            </html>
            """)
    except Exception as e:
        return HTMLResponse(f"<html><body><h1>Error: {e}</h1></body></html>")

    # Test route to check if routing works
    @app.get("/test")
    async def test_route():
        return {"message": "Test route works!"}

    # Serve static files directly
    @app.get("/{file_path}")
    async def serve_static_files(file_path: str):
        """Serve static files from frontend directory"""
        # Only serve specific files we know about
        allowed_files = ["style.css", "app.js", "terminal.js", "favicon.ico"]
        
        if file_path not in allowed_files:
            raise HTTPException(status_code=404, detail="File not found")
        
        try:
            frontend_dir = Path(__file__).parent.parent / "frontend"
            file = frontend_dir / file_path
            
            if not file.exists():
                raise HTTPException(status_code=404, detail="File not found")
            
            # Determine media type
            if file_path.endswith('.css'):
                media_type = "text/css"
            elif file_path.endswith('.js'):
                media_type = "application/javascript"
            elif file_path.endswith('.ico'):
                media_type = "image/x-icon"
            else:
                media_type = "text/plain"
            
            return FileResponse(file, media_type=media_type)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_status = "available" if nexus_model and nexus_model.is_available() else "unavailable"
    db_status = "connected" if await db.ping() else "disconnected"
    
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "services": {
            "illuminator_model": model_status,
            "database": db_status
        }
    }

@app.post("/api/sessions")
async def create_session():
    """Create a new session"""
    session_id = await db.create_session()
    return SessionResponse(
        session_id=session_id,
        timestamp=datetime.now(),
        status="created"
    )

@app.post("/api/session/create")
async def create_session_alt():
    """Create a new session (alternative endpoint for frontend compatibility)"""
    session_id = await db.create_session()
    return SessionResponse(
        session_id=session_id,
        timestamp=datetime.now(),
        status="created"
    )

@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session history"""
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

@app.post("/api/code/generate")
async def generate_code(request: CodeGenerationRequest):
    """Generate code using iLLuMinator-4.7B"""
    if not nexus_model or not nexus_model.is_available():
        raise HTTPException(status_code=503, detail="iLLuMinator model not available")
    
    try:
        # Use prompt if provided, otherwise instruction
        instruction_text = request.prompt or request.instruction
        if not instruction_text:
            raise HTTPException(status_code=400, detail="Either 'instruction' or 'prompt' field is required")
        
        # Generate code
        code = nexus_model.generate_code(instruction_text, request.language)
        
        # Save to session if provided
        if request.session_id:
            await db.add_to_session(request.session_id, {
                "type": "code_generation",
                "input": {
                    "instruction": instruction_text,
                    "language": request.language
                },
                "output": code,
                "timestamp": datetime.now()
            })
        
        return {
            "code": code,
            "language": request.language,
            "instruction": instruction_text,
            "timestamp": datetime.now(),
            "model": "iLLuMinator-4.7B"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Code generation failed: {str(e)}")

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Chat with iLLuMinator-4.7B"""
    if not nexus_model or not nexus_model.is_available():
        raise HTTPException(status_code=503, detail="iLLuMinator model not available")
    
    try:
        # Generate response
        response = nexus_model.generate_response(
            request.message, 
            request.max_length, 
            request.temperature
        )
        
        # Save to session if provided
        if request.session_id:
            await db.add_to_session(request.session_id, {
                "type": "chat",
                "input": request.message,
                "output": response,
                "timestamp": datetime.now()
            })
        
        return {
            "response": response,
            "message": request.message,
            "timestamp": datetime.now(),
            "model": "iLLuMinator-4.7B"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.post("/api/analyze")
async def analyze_code(request: FileAnalysisRequest):
    """Analyze code using iLLuMinator-4.7B"""
    if not nexus_model:
        raise HTTPException(status_code=503, detail="Analysis service not available")
    
    try:
        # Analyze code
        analysis = nexus_model.analyze_code(request.code)
        
        # Save to session if provided
        if request.session_id:
            await db.add_to_session(request.session_id, {
                "type": "code_analysis",
                "input": {
                    "code": request.code[:200] + "..." if len(request.code) > 200 else request.code,
                    "filename": request.filename
                },
                "output": analysis,
                "timestamp": datetime.now()
            })
        
        return {
            "analysis": analysis,
            "filename": request.filename,
            "timestamp": datetime.now(),
            "model": "iLLuMinator-4.7B"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Code analysis failed: {str(e)}")

@app.post("/api/command")
async def execute_command(request: CommandRequest):
    """Execute a Nexus CLI command"""
    try:
        # Import the CLI here to avoid circular imports
        from nexus_cli import IntelligentNexusCLI
        
        cli = IntelligentNexusCLI()
        result = cli.process_command(request.command)
        
        # Save to session if provided
        if request.session_id:
            await db.add_to_session(request.session_id, {
                "type": "command",
                "input": request.command,
                "output": result,
                "timestamp": datetime.now()
            })
        
        return {
            "result": result,
            "command": request.command,
            "timestamp": datetime.now()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Command execution failed: {str(e)}")

@app.get("/api/model/info")
async def get_model_info():
    """Get information about the iLLuMinator model"""
    if not nexus_model:
        raise HTTPException(status_code=503, detail="Model not available")
    
    return nexus_model.get_model_info()

@app.get("/api/model/status")
async def get_model_status():
    """Get model status"""
    if not nexus_model:
        return {"status": "unavailable", "message": "Model not initialized"}
    
    return {
        "status": "available" if nexus_model.is_available() else "unavailable",
        "connection_test": nexus_model.test_connection(),
        "info": nexus_model.get_model_info()
    }

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time communication"""
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Process different types of messages
            if message_data.get("type") == "chat":
                if nexus_model and nexus_model.is_available():
                    response = nexus_model.generate_response(message_data.get("message", ""))
                    await websocket.send_text(json.dumps({
                        "type": "chat_response",
                        "response": response,
                        "timestamp": datetime.now().isoformat()
                    }))
                else:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "iLLuMinator model not available"
                    }))
            
            elif message_data.get("type") == "code":
                if nexus_model and nexus_model.is_available():
                    code = nexus_model.generate_code(
                        message_data.get("instruction", ""),
                        message_data.get("language", "python")
                    )
                    await websocket.send_text(json.dumps({
                        "type": "code_response",
                        "code": code,
                        "language": message_data.get("language", "python"),
                        "timestamp": datetime.now().isoformat()
                    }))
                else:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "iLLuMinator model not available"
                    }))
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
