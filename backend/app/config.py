import os
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    APP_NAME: str = "SalesCast AI"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Paths
    BASE_DIR: str = str(Path(__file__).resolve().parent.parent.parent)
    ARTIFACT_DIR: str = ""
    
    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_TTL: int = 3600
    
    # MySQL
    MYSQL_HOST: str = "localhost"
    MYSQL_PORT: int = 3306
    MYSQL_USER: str = "salescast"
    MYSQL_PASSWORD: str = "salescast_pass"
    MYSQL_DATABASE: str = "salescast_db"
    
    # Inference
    FORECAST_HORIZON: int = 8
    REQUEST_TIMEOUT: int = 120
    
    # CORS
    CORS_ORIGINS: str = "*"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    def get_artifact_dir(self):
        if self.ARTIFACT_DIR:
            return self.ARTIFACT_DIR
        return os.path.join(self.BASE_DIR, "artifacts")

settings = Settings()
