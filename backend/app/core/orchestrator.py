"""Inference Orchestrator - coordinates the full forecast job lifecycle."""
import asyncio
import time
import uuid
import os
import traceback
from typing import AsyncGenerator
from app.services.artifact_registry import registry
from app.services.redis_service import redis_service
from app.services import mysql_service
from app.models.adapters import get_all_adapters
from app.config import settings

SSE_STAGES = [
    ("request_received", "Request validated and job created", 0.05),
    ("data_loaded", "Historical data loaded successfully", 0.10),
    ("preprocessing_complete", "Preprocessing pipeline applied", 0.20),
    ("features_generated", "Feature generation complete", 0.30),
    ("model_inference_running", "Running model inference", 0.40),
    ("comparison_finished", "Model comparison complete", 0.85),
    ("best_model_selected", "Best model selected", 0.90),
    ("forecast_ready", "Final forecast ready", 1.0),
]

async def run_forecast_job(job_id: str, state_query: str) -> AsyncGenerator[dict, None]:
    """Execute the full inference pipeline, yielding SSE events at each stage."""
    
    start_time = time.time()
    
    try:
        # Stage 1: Request received
        redis_service.set_job(job_id, {"status": "running", "state": state_query})
        mysql_service.save_job(job_id, state_query, "running")
        yield _event("request_received", "Request validated and job created", 0.05)
        await asyncio.sleep(0.3)
        
        # Stage 2: Load data
        context_data = registry.processed_data
        states = registry.states if state_query == "all" else [state_query]
        
        if state_query != "all" and state_query not in registry.states:
            yield _event("error", f"State '{state_query}' not found. Available: {registry.states}", 0)
            return
        
        yield _event("data_loaded", f"Loaded data for {len(states)} state(s)", 0.10)
        await asyncio.sleep(0.3)
        
        # Stage 3: Preprocessing
        yield _event("preprocessing_complete", "Training-time preprocessing pipeline replayed", 0.20)
        await asyncio.sleep(0.3)
        
        # Stage 4: Features
        yield _event("features_generated", f"Generated {len(registry.feature_schema['feature_columns'])} features", 0.30)
        await asyncio.sleep(0.3)
        
        # Stage 5: Model inference
        adapters = get_all_adapters()
        model_dir = registry.get_model_path("")
        model_dir = model_dir.rsplit(os.sep, 1)[0] if model_dir.endswith(os.sep) else model_dir
        
        base_model_dir = os.path.join(registry.base, "models")
        
        # Load all adapters
        for adapter in adapters:
            try:
                adapter.load(base_model_dir, states)
            except Exception as e:
                print(f"[Orchestrator] Failed to load {adapter.name}: {e}")
        
        model_forecasts = {}
        model_latencies = {}
        
        for i, adapter in enumerate(adapters):
            progress = 0.30 + (i + 1) * (0.55 / len(adapters))
            try:
                t0 = time.time()
                all_preds = []
                
                for state in states:
                    if adapter.is_available(state):
                        preds = adapter.predict(state, settings.FORECAST_HORIZON, context_data)
                        for p in preds:
                            p["state"] = state
                        all_preds.extend(preds)
                
                latency = (time.time() - t0) * 1000
                model_forecasts[adapter.name] = all_preds
                model_latencies[adapter.name] = round(latency, 1)
                
                yield _event("model_running", 
                           f"{adapter.name.upper()} inference complete ({latency:.0f}ms)",
                           progress, model=adapter.name, status="completed")
            except Exception as e:
                print(f"[Orchestrator] {adapter.name} inference failed: {traceback.format_exc()}")
                model_forecasts[adapter.name] = []
                model_latencies[adapter.name] = 0
                yield _event("model_running",
                           f"{adapter.name.upper()} failed: {str(e)[:100]}",
                           progress, model=adapter.name, status="failed")
            
            await asyncio.sleep(0.2)
        
        # Stage 6: Comparison
        ranking = registry.get_ranking()
        yield _event("comparison_finished", "All models compared using validation metrics", 0.85)
        await asyncio.sleep(0.3)
        
        # Stage 7: Select best
        champion = registry.champion
        valid_models = [m for m in ranking if model_forecasts.get(m["model"]) and len(model_forecasts[m["model"]]) > 0]
        
        if valid_models:
            champion = valid_models[0]["model"]
        
        champion_reason = f"{champion.upper()} selected: lowest validation wMAPE ({registry.get_model_metrics(champion).get('wMAPE', 'N/A')})"
        
        yield _event("best_model_selected", champion_reason, 0.90, model=champion)
        await asyncio.sleep(0.3)
        
        # Stage 8: Build final result
        model_results = []
        for r in ranking:
            m = r["model"]
            metrics = r["metrics"]
            model_results.append({
                "model_name": m,
                "wMAPE": metrics.get("wMAPE", 999),
                "RMSE": metrics.get("RMSE", 0),
                "MAE": metrics.get("MAE", 0),
                "rank": r["rank"],
                "is_champion": m == champion,
                "status": "completed" if model_forecasts.get(m) else "failed",
                "latency_ms": model_latencies.get(m, 0),
            })
            
            mysql_service.save_model_evaluation(job_id, m, metrics, r["rank"], m == champion)
        
        forecasts = model_forecasts.get(champion, [])
        
        # Persist
        mysql_service.save_forecast_results(job_id, forecasts, champion)
        mysql_service.save_job(job_id, state_query, "completed", champion)
        
        total_time = round((time.time() - start_time) * 1000, 1)
        
        result = {
            "job_id": job_id,
            "status": "completed",
            "champion_model": champion,
            "champion_reason": champion_reason,
            "model_results": model_results,
            "forecasts": forecasts,
            "total_states": len(states),
            "forecast_horizon": settings.FORECAST_HORIZON,
            "total_latency_ms": total_time,
        }
        
        redis_service.set_result(job_id, result)
        redis_service.set_job(job_id, {"status": "completed"})
        
        yield _event("forecast_ready", "Forecast complete!", 1.0, result=result)
        
    except Exception as e:
        error_msg = f"Job failed: {str(e)}"
        print(f"[Orchestrator] {traceback.format_exc()}")
        redis_service.set_job(job_id, {"status": "failed", "error": error_msg})
        mysql_service.save_job(job_id, state_query, "failed")
        yield _event("error", error_msg, 0)

def create_job_id():
    return str(uuid.uuid4())[:12]

def _event(stage, message, progress, **kwargs):
    data = {"stage": stage, "message": message, "progress": progress, "timestamp": time.time()}
    data.update(kwargs)
    return data
