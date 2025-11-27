"""
FastAPI Web Application for SAM3 Conversational Agent

Provides REST API endpoints for conversational SAM3 operations
with WebSocket support for real-time communication.
"""

import os
import time
import asyncio
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Import our agent components
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.agent.conversational_agent import SAM3ConversationalAgent, AgentConfig, SAM3Config
from src.core.sam3_engine import SAM3Config as EngineConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global agent instance
agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global agent
    
    # Startup
    logger.info("🚀 Starting SAM3 Conversational Agent...")
    
    try:
        # Initialize configurations
        agent_config = AgentConfig(
            max_history_turns=20,
            enable_learning=True,
            confidence_threshold=0.7,
            cache_responses=True,
            enable_monitoring=True
        )
        
        sam3_config = SAM3Config(
            device="cuda",
            batch_size=4,
            enable_jit=True,
            precompile_models=True,
            memory_efficient=True
        )
        
        # Initialize agent
        agent = SAM3ConversationalAgent(agent_config, sam3_config)
        
        logger.info("✅ SAM3 Conversational Agent initialized successfully")
        
        # Perform health check
        health = await agent.health_check()
        if health["status"] == "healthy":
            logger.info("✅ Health check passed")
        else:
            logger.warning(f"⚠️ Health check warning: {health}")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize agent: {e}")
        raise
    finally:
        # Shutdown
        logger.info("🛑 Shutting down SAM3 Conversational Agent...")

# Create FastAPI app
app = FastAPI(
    title="SAM3 Conversational Agent",
    description="High-performance conversational agent for SAM3 segmentation operations",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class QueryRequest(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None
    options: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    success: bool
    response: str
    processing_time: float
    sam3_results: Optional[Dict[str, Any]] = None
    parsed_query: Optional[Dict[str, Any]] = None
    processing_plan: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: float
    details: Dict[str, Any]

class PerformanceResponse(BaseModel):
    performance_summary: Dict[str, Any]
    system_resources: Dict[str, Any]
    sam3_performance: Dict[str, Any]
    conversation_turns: int
    optimization_status: Dict[str, Any]

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "SAM3 Conversational Agent API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "/health": "Health check",
            "/query": "Process natural language query",
            "/upload": "Upload image and process",
            "/websocket": "Real-time communication",
            "/performance": "Performance metrics",
            "/history": "Conversation history"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    health = await agent.health_check()
    return HealthResponse(**health)

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process natural language query"""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        logger.info(f"Processing query: {request.query}")
        
        result = await agent.process_query(
            user_query=request.query,
            context=request.context
        )
        
        return QueryResponse(
            success=result.get("error") is None,
            response=result["response"],
            processing_time=result["processing_time"],
            sam3_results=result.get("sam3_results"),
            parsed_query=result.get("parsed_query"),
            processing_plan=result.get("processing_plan"),
            error=result.get("error")
        )
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_and_process(
    file: UploadFile = File(...),
    query: str = "Describe what you see in this image",
    context: Optional[Dict[str, Any]] = None
):
    """Upload image and process with query"""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read file content
        contents = await file.read()
        
        # For simplicity, we'll just process the query without the image
        # In a full implementation, you'd save the image and process it
        logger.info(f"Processing uploaded image: {file.filename}")
        
        result = await agent.process_query(
            user_query=f"{query} (for image: {file.filename})",
            context={"filename": file.filename, **context} if context else {"filename": file.filename}
        )
        
        return {
            "success": result.get("error") is None,
            "response": result["response"],
            "processing_time": result["processing_time"],
            "filename": file.filename,
            "query": query,
            "sam3_results": result.get("sam3_results"),
            "parsed_query": result.get("parsed_query"),
            "error": result.get("error")
        }
        
    except Exception as e:
        logger.error(f"Upload processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/performance", response_model=PerformanceResponse)
async def get_performance_report():
    """Get performance metrics"""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        report = agent.get_performance_report()
        return PerformanceResponse(**report)
        
    except Exception as e:
        logger.error(f"Failed to get performance report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
async def get_conversation_history():
    """Get conversation history"""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        history = agent.get_conversation_history()
        return {
            "success": True,
            "conversation_turns": len(history),
            "history": history
        }
        
    except Exception as e:
        logger.error(f"Failed to get conversation history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/history")
async def clear_conversation_history():
    """Clear conversation history"""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        agent.clear_history()
        return {"success": True, "message": "Conversation history cleared"}
        
    except Exception as e:
        logger.error(f"Failed to clear history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/cache")
async def clear_cache():
    """Clear cache"""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        agent.clear_cache()
        return {"success": True, "message": "Cache cleared"}
        
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time communication
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            logger.info(f"WebSocket received: {data}")
            
            # Process the query
            if agent is None:
                await websocket.send_text("Agent not initialized")
                continue
            
            try:
                # Parse the message (could be JSON or simple text)
                import json
                try:
                    message_data = json.loads(data)
                    query = message_data.get("query", "")
                    context = message_data.get("context", {})
                except json.JSONDecodeError:
                    query = data
                    context = {}
                
                # Process query
                result = await agent.process_query(
                    user_query=query,
                    context=context
                )
                
                # Send response back
                response_data = {
                    "success": result.get("error") is None,
                    "response": result["response"],
                    "processing_time": result["processing_time"],
                    "sam3_results": result.get("sam3_results"),
                    "parsed_query": result.get("parsed_query"),
                    "from_cache": result.get("from_cache", False)
                }
                
                await websocket.send_text(json.dumps(response_data))
                
            except Exception as e:
                error_response = {
                    "success": False,
                    "error": str(e),
                    "response": "I apologize, but I encountered an error processing your query."
                }
                await websocket.send_text(json.dumps(error_response))
                logger.error(f"WebSocket processing error: {e}")
                
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()

# Static file serving for web interface (optional)
@app.get("/static/{file_path:path}")
async def serve_static(file_path: str):
    """Serve static files"""
    # Implementation for serving static web interface
    pass

if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )