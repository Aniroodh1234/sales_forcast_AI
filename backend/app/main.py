"""SalesCast AI - FastAPI Backend Entry Point."""
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.services.artifact_registry import registry
from app.services.redis_service import redis_service
from app.services import mysql_service
from app.config import settings

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-Powered Sales Forecasting API"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)

@app.on_event("startup")
async def startup():
    """Load all artifacts and connect services on startup."""
    print(f"[{settings.APP_NAME}] Starting up...")
    
    # Load ML artifacts
    registry.load_all()
    
    # Connect Redis
    redis_service.connect()
    
    # Connect MySQL
    mysql_service.connect()
    
    print(f"[{settings.APP_NAME}] Ready! Champion: {registry.champion}")

@app.get("/")
async def root():
    return {"name": settings.APP_NAME, "version": settings.APP_VERSION, "status": "running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
