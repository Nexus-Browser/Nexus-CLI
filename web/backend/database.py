"""
MongoDB Database Module for Nexus CLI Web Interface
Handles session management and data persistence
"""

import asyncio
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import uuid

try:
    from motor.motor_asyncio import AsyncIOMotorClient
    from pymongo import ASCENDING, DESCENDING
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    print("Warning: MongoDB dependencies not available. Using in-memory storage.")

class MongoDB:
    """MongoDB database handler with fallback to in-memory storage"""
    
    def __init__(self, connection_string: Optional[str] = None):
        self.connection_string = connection_string or os.getenv(
            'MONGODB_URL', 'mongodb://localhost:27017/nexus_cli'
        )
        self.client = None
        self.db = None
        self.sessions_collection = None
        
        # In-memory fallback storage
        self.memory_storage = {
            "sessions": {},
            "commands": [],
            "analytics": []
        }
        
        if MONGODB_AVAILABLE:
            try:
                self.client = AsyncIOMotorClient(self.connection_string)
                self.db = self.client.nexus_cli
                self.sessions_collection = self.db.sessions
                self.commands_collection = self.db.commands
                self.analytics_collection = self.db.analytics
                print("MongoDB client initialized successfully")
            except Exception as e:
                print(f"Failed to connect to MongoDB: {e}")
                print("Using in-memory storage instead")
                self.client = None
        else:
            print("Using in-memory storage (MongoDB not available)")
    
    async def ping(self) -> bool:
        """Test database connection"""
        if self.client:
            try:
                await self.client.admin.command('ping')
                return True
            except Exception:
                return False
        return True  # In-memory storage is always "available"
    
    async def create_session(self) -> str:
        """Create a new session"""
        session_id = str(uuid.uuid4())
        session_data = {
            "session_id": session_id,
            "created_at": datetime.now(),
            "last_activity": datetime.now(),
            "history": [],
            "metadata": {
                "user_agent": None,
                "ip_address": None
            }
        }
        
        if self.sessions_collection:
            try:
                await self.sessions_collection.insert_one(session_data)
            except Exception as e:
                print(f"Failed to create session in MongoDB: {e}")
                # Fall back to memory storage
                self.memory_storage["sessions"][session_id] = session_data
        else:
            self.memory_storage["sessions"][session_id] = session_data
        
        return session_id
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID"""
        if self.sessions_collection:
            try:
                session = await self.sessions_collection.find_one({"session_id": session_id})
                if session:
                    # Convert ObjectId to string for JSON serialization
                    session["_id"] = str(session["_id"])
                    return session
            except Exception as e:
                print(f"Failed to get session from MongoDB: {e}")
        
        # Fall back to memory storage
        return self.memory_storage["sessions"].get(session_id)
    
    async def add_to_session(self, session_id: str, data: Dict[str, Any]) -> bool:
        """Add data to session history"""
        if self.sessions_collection:
            try:
                result = await self.sessions_collection.update_one(
                    {"session_id": session_id},
                    {
                        "$push": {"history": data},
                        "$set": {"last_activity": datetime.now()}
                    }
                )
                return result.modified_count > 0
            except Exception as e:
                print(f"Failed to update session in MongoDB: {e}")
        
        # Fall back to memory storage
        if session_id in self.memory_storage["sessions"]:
            self.memory_storage["sessions"][session_id]["history"].append(data)
            self.memory_storage["sessions"][session_id]["last_activity"] = datetime.now()
            return True
        
        return False
    
    async def get_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent sessions"""
        if self.sessions_collection:
            try:
                cursor = self.sessions_collection.find().sort("last_activity", DESCENDING).limit(limit)
                sessions = []
                async for session in cursor:
                    session["_id"] = str(session["_id"])
                    sessions.append(session)
                return sessions
            except Exception as e:
                print(f"Failed to get recent sessions from MongoDB: {e}")
        
        # Fall back to memory storage
        sessions = list(self.memory_storage["sessions"].values())
        sessions.sort(key=lambda x: x["last_activity"], reverse=True)
        return sessions[:limit]
    
    async def log_command(self, command: str, result: str, session_id: Optional[str] = None) -> bool:
        """Log a command execution"""
        log_data = {
            "command": command,
            "result": result,
            "session_id": session_id,
            "timestamp": datetime.now(),
            "success": not result.startswith("[Error]")
        }
        
        if self.commands_collection:
            try:
                await self.commands_collection.insert_one(log_data)
                return True
            except Exception as e:
                print(f"Failed to log command to MongoDB: {e}")
        
        # Fall back to memory storage
        self.memory_storage["commands"].append(log_data)
        return True
    
    async def get_command_history(self, session_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get command history"""
        if self.commands_collection:
            try:
                query = {"session_id": session_id} if session_id else {}
                cursor = self.commands_collection.find(query).sort("timestamp", DESCENDING).limit(limit)
                commands = []
                async for command in cursor:
                    command["_id"] = str(command["_id"])
                    commands.append(command)
                return commands
            except Exception as e:
                print(f"Failed to get command history from MongoDB: {e}")
        
        # Fall back to memory storage
        commands = self.memory_storage["commands"]
        if session_id:
            commands = [cmd for cmd in commands if cmd.get("session_id") == session_id]
        commands.sort(key=lambda x: x["timestamp"], reverse=True)
        return commands[:limit]
    
    async def log_analytics(self, event_type: str, data: Dict[str, Any], session_id: Optional[str] = None) -> bool:
        """Log analytics data"""
        analytics_data = {
            "event_type": event_type,
            "data": data,
            "session_id": session_id,
            "timestamp": datetime.now()
        }
        
        if self.analytics_collection:
            try:
                await self.analytics_collection.insert_one(analytics_data)
                return True
            except Exception as e:
                print(f"Failed to log analytics to MongoDB: {e}")
        
        # Fall back to memory storage
        self.memory_storage["analytics"].append(analytics_data)
        return True
    
    async def get_analytics(self, event_type: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get analytics data"""
        if self.analytics_collection:
            try:
                query = {"event_type": event_type} if event_type else {}
                cursor = self.analytics_collection.find(query).sort("timestamp", DESCENDING).limit(limit)
                analytics = []
                async for item in cursor:
                    item["_id"] = str(item["_id"])
                    analytics.append(item)
                return analytics
            except Exception as e:
                print(f"Failed to get analytics from MongoDB: {e}")
        
        # Fall back to memory storage
        analytics = self.memory_storage["analytics"]
        if event_type:
            analytics = [item for item in analytics if item.get("event_type") == event_type]
        analytics.sort(key=lambda x: x["timestamp"], reverse=True)
        return analytics[:limit]
    
    async def cleanup_old_sessions(self, days_old: int = 30) -> int:
        """Clean up old sessions"""
        cutoff_date = datetime.now() - asyncio.timedelta(days=days_old)
        
        if self.sessions_collection:
            try:
                result = await self.sessions_collection.delete_many(
                    {"last_activity": {"$lt": cutoff_date}}
                )
                return result.deleted_count
            except Exception as e:
                print(f"Failed to cleanup sessions in MongoDB: {e}")
        
        # Fall back to memory storage
        original_count = len(self.memory_storage["sessions"])
        self.memory_storage["sessions"] = {
            k: v for k, v in self.memory_storage["sessions"].items()
            if v["last_activity"] >= cutoff_date
        }
        return original_count - len(self.memory_storage["sessions"])
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        stats = {
            "sessions_count": 0,
            "commands_count": 0,
            "analytics_count": 0,
            "storage_type": "memory" if not self.client else "mongodb"
        }
        
        if self.client:
            try:
                stats["sessions_count"] = await self.sessions_collection.count_documents({})
                stats["commands_count"] = await self.commands_collection.count_documents({})
                stats["analytics_count"] = await self.analytics_collection.count_documents({})
            except Exception as e:
                print(f"Failed to get stats from MongoDB: {e}")
                # Fall back to memory counts
                stats["sessions_count"] = len(self.memory_storage["sessions"])
                stats["commands_count"] = len(self.memory_storage["commands"])
                stats["analytics_count"] = len(self.memory_storage["analytics"])
                stats["storage_type"] = "memory_fallback"
        else:
            stats["sessions_count"] = len(self.memory_storage["sessions"])
            stats["commands_count"] = len(self.memory_storage["commands"])
            stats["analytics_count"] = len(self.memory_storage["analytics"])
        
        return stats
