from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum

class ForecastRequest(BaseModel):
    state: str = Field(default="all", description="State name or 'all' for all states")

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class ForecastPoint(BaseModel):
    state: str
    forecast_date: str
    predicted_value: float
    confidence_lower: Optional[float] = None
    confidence_upper: Optional[float] = None

class ModelResult(BaseModel):
    model_name: str
    wMAPE: float
    RMSE: float
    MAE: float
    rank: int
    is_champion: bool = False
    status: str = "completed"
    latency_ms: Optional[float] = None

class ForecastResponse(BaseModel):
    job_id: str
    status: str
    champion_model: str
    champion_reason: str
    model_results: List[ModelResult]
    forecasts: List[ForecastPoint]
    total_states: int
    forecast_horizon: int
