"""
Nexus CLI Web Backend - DEBUG VERSION
Minimal FastAPI server to test static file serving
"""

from pathlib import Path
import sys
import os

# Add the parent directory to sys.path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
    print("✅ FastAPI imported successfully")
except ImportError as e:
    FASTAPI_AVAILABLE = False
    print(f"❌ FastAPI import failed: {e}")

if FASTAPI_AVAILABLE:
    # Create the FastAPI app
    app = FastAPI(title="Nexus CLI Debug", version="1.0.0")
    print("✅ FastAPI app created")
    
    # Get frontend directory
    frontend_dir = Path(__file__).parent / "web" / "frontend"
    print(f"📁 Frontend directory: {frontend_dir}")
    print(f"📁 Directory exists: {frontend_dir.exists()}")
    
    if frontend_dir.exists():
        files = list(frontend_dir.glob("*"))
        print(f"📄 Files found: {[f.name for f in files]}")

    @app.get("/")
    async def root():
        """Serve the main HTML file"""
        html_file = frontend_dir / "index.html"
        if html_file.exists():
            return FileResponse(html_file)
        else:
            return HTMLResponse("<h1>Frontend files not found</h1>")

    @app.get("/debug")
    async def debug():
        """Debug endpoint"""
        return {
            "message": "Debug endpoint works!",
            "frontend_dir": str(frontend_dir),
            "frontend_exists": frontend_dir.exists(),
            "files": [f.name for f in frontend_dir.glob("*")] if frontend_dir.exists() else []
        }

    @app.get("/style.css")
    async def get_css():
        """Serve CSS file"""
        css_file = frontend_dir / "style.css"
        print(f"🎨 CSS file requested: {css_file}")
        print(f"🎨 CSS file exists: {css_file.exists()}")
        
        if css_file.exists():
            return FileResponse(css_file, media_type="text/css")
        else:
            raise HTTPException(status_code=404, detail="CSS file not found")

    @app.get("/app.js")
    async def get_app_js():
        """Serve JS file"""
        js_file = frontend_dir / "app.js"
        print(f"⚡ JS file requested: {js_file}")
        print(f"⚡ JS file exists: {js_file.exists()}")
        
        if js_file.exists():
            return FileResponse(js_file, media_type="application/javascript")
        else:
            raise HTTPException(status_code=404, detail="JS file not found")

    @app.get("/terminal.js")
    async def get_terminal_js():
        """Serve terminal JS file"""
        js_file = frontend_dir / "terminal.js"
        print(f"🖥️ Terminal JS file requested: {js_file}")
        print(f"🖥️ Terminal JS file exists: {js_file.exists()}")
        
        if js_file.exists():
            return FileResponse(js_file, media_type="application/javascript")
        else:
            raise HTTPException(status_code=404, detail="Terminal JS file not found")

    print("✅ All routes defined")

else:
    print("❌ FastAPI not available, cannot create app")
    
if __name__ == "__main__":
    if FASTAPI_AVAILABLE:
        print("🚀 Starting debug server...")
        uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
    else:
        print("❌ Cannot start server - FastAPI not available")
