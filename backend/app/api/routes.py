"""REST + SSE API routes."""
import json
import asyncio
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from app.schemas import ForecastRequest
from app.core.orchestrator import run_forecast_job, create_job_id
from app.services.artifact_registry import registry
from app.services.redis_service import redis_service

router = APIRouter(prefix="/api")

@router.post("/forecast")
async def start_forecast(req: ForecastRequest):
    """Start a new forecast job. Returns job_id for SSE streaming."""
    job_id = create_job_id()
    
    # Validate state
    if req.state != "all" and req.state not in registry.states:
        raise HTTPException(400, f"Invalid state '{req.state}'. Use 'all' or one of: {registry.states}")
    
    redis_service.set_job(job_id, {"status": "pending", "state": req.state})
    
    return {"job_id": job_id, "state": req.state, "status": "pending",
            "stream_url": f"/api/forecast/{job_id}/stream"}

@router.get("/forecast/{job_id}/stream")
async def stream_forecast(job_id: str):
    """SSE endpoint - streams live progress events during inference."""
    
    async def event_generator():
        async for event in run_forecast_job(job_id, _get_state_query(job_id)):
            data = json.dumps(event, default=str)
            yield f"data: {data}\n\n"
        yield "data: {\"stage\": \"done\", \"message\": \"Stream complete\"}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        }
    )

@router.get("/forecast/{job_id}/result")
async def get_result(job_id: str):
    """Get completed forecast result."""
    result = redis_service.get_result(job_id)
    if not result:
        raise HTTPException(404, f"No result found for job {job_id}")
    return result

@router.get("/models/leaderboard")
async def get_leaderboard():
    """Get model leaderboard from offline validation metrics."""
    ranking = registry.get_ranking()
    return {
        "champion": registry.champion,
        "models": ranking,
        "selection_criteria": "Primary: wMAPE (lower is better). Secondary: RMSE, MAE.",
    }

@router.get("/states")
async def get_states():
    """Get list of available states."""
    return {"states": registry.states, "count": len(registry.states)}

@router.get("/health")
async def health():
    return {"status": "healthy", "version": "1.0.0", 
            "models_loaded": True, "champion": registry.champion}

def _get_state_query(job_id: str) -> str:
    job = redis_service.get_job(job_id)
    if job and "state" in job:
        return job["state"]
    return "all"
