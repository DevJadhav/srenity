"""Main FastAPI application for SRE workflow agent."""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

try:
    # Try relative imports first (when run as module)
    from .endpoints import router, set_agent
    from ..agent.workflow import SREWorkflowAgent
except ImportError:
    # Fall back to absolute imports (when run directly)
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from api.endpoints import router, set_agent
    from agent.workflow import SREWorkflowAgent


# Global agent instance
sre_agent: SREWorkflowAgent = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    # Startup
    logger.info("Starting SRE Workflow Agent application...")
    
    try:
        # Initialize the SRE agent
        global sre_agent
        
        # Use environment variables for configuration or defaults
        log_file = os.getenv("LOG_FILE_PATH", "data/Apache_2k.log")
        templates_file = os.getenv("TEMPLATES_FILE_PATH", "data/Apache_2k.log_templates.csv")
        llm_model = os.getenv("LLM_MODEL", "gpt-4")
        
        logger.info(f"Initializing SRE agent with log file: {log_file}")
        logger.info(f"Templates file: {templates_file}")
        logger.info(f"LLM model: {llm_model}")
        
        sre_agent = SREWorkflowAgent(
            log_file_path=log_file,
            templates_file_path=templates_file,
            llm_model=llm_model
        )
        
        # Set the agent in the endpoints module
        set_agent(sre_agent)
        
        logger.info("SRE Workflow Agent initialized successfully")
        
        # Perform health check
        health_result = sre_agent.health_check()
        if health_result["success"]:
            logger.info(f"Health check passed: {health_result['status']}")
            logger.info(f"Log entries: {health_result.get('log_entries', 0)}")
            logger.info(f"Templates: {health_result.get('templates', 0)}")
        else:
            logger.warning(f"Health check warning: {health_result}")
        
    except Exception as e:
        logger.error(f"Failed to initialize SRE agent: {e}")
        raise e
    
    yield
    
    # Shutdown
    logger.info("Shutting down SRE Workflow Agent application...")


# Create FastAPI app with lifespan
app = FastAPI(
    title="SRE Workflow Agent",
    description="Site Reliability Engineering workflow agent using LangGraph for incident management and log analysis",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api", tags=["SRE Agent"])


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "SRE Workflow Agent API",
        "version": "0.1.0",
        "status": "running",
        "endpoints": {
            "health": "/api/health",
            "incident_analysis": "/api/incidents/analyze",
            "logs": "/api/logs",
            "metrics": "/api/metrics", 
            "traces": "/api/traces",
            "statistics": "/api/statistics",
            "error_analysis": "/api/errors/analysis",
            "dashboard_summary": "/api/dashboard/summary",
            "docs": "/docs",
            "openapi": "/openapi.json"
        }
    }


@app.get("/status")
async def app_status():
    """Application status endpoint."""
    try:
        if sre_agent:
            health_result = sre_agent.health_check()
            return {
                "application": "healthy",
                "agent_status": health_result["status"],
                "components": health_result.get("components", {}),
                "log_entries": health_result.get("log_entries", 0),
                "templates": health_result.get("templates", 0)
            }
        else:
            return {
                "application": "unhealthy",
                "reason": "SRE agent not initialized"
            }
    except Exception as e:
        logger.error(f"Error in status check: {e}")
        return {
            "application": "error",
            "reason": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    
    # Configure logging
    logger.add(
        "logs/sre_agent_{time}.log",
        rotation="1 day",
        retention="7 days",
        level="INFO"
    )
    
    # Run the application
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 