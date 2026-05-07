"""Redis service for job state management."""
import json
import redis
from app.config import settings

class RedisService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._client = None
            cls._instance._memory = {}
        return cls._instance
    
    def connect(self):
        try:
            self._client = redis.Redis(
                host=settings.REDIS_HOST, port=settings.REDIS_PORT,
                db=settings.REDIS_DB, decode_responses=True
            )
            self._client.ping()
            print("[Redis] Connected")
            return True
        except Exception as e:
            print(f"[Redis] Connection failed: {e}. Using in-memory fallback.")
            self._client = None
            self._memory = {}
            return False
    
    @property
    def available(self):
        return self._client is not None
    
    def set_job(self, job_id: str, data: dict, ttl: int = None):
        ttl = ttl or settings.REDIS_TTL
        payload = json.dumps(data, default=str)
        if self.available:
            self._client.setex(f"job:{job_id}", ttl, payload)
        else:
            self._memory[f"job:{job_id}"] = payload
    
    def get_job(self, job_id: str):
        if self.available:
            data = self._client.get(f"job:{job_id}")
        else:
            data = self._memory.get(f"job:{job_id}")
        return json.loads(data) if data else None
    
    def update_progress(self, job_id: str, stage: str, progress: float, detail: str = ""):
        data = {"stage": stage, "progress": progress, "detail": detail}
        payload = json.dumps(data)
        if self.available:
            self._client.setex(f"job:{job_id}:progress", settings.REDIS_TTL, payload)
        else:
            self._memory[f"job:{job_id}:progress"] = payload
    
    def get_progress(self, job_id: str):
        if self.available:
            data = self._client.get(f"job:{job_id}:progress")
        else:
            data = self._memory.get(f"job:{job_id}:progress")
        return json.loads(data) if data else None
    
    def set_result(self, job_id: str, result: dict):
        payload = json.dumps(result, default=str)
        if self.available:
            self._client.setex(f"job:{job_id}:result", settings.REDIS_TTL, payload)
        else:
            self._memory[f"job:{job_id}:result"] = payload
    
    def get_result(self, job_id: str):
        if self.available:
            data = self._client.get(f"job:{job_id}:result")
        else:
            data = self._memory.get(f"job:{job_id}:result")
        return json.loads(data) if data else None

redis_service = RedisService()
