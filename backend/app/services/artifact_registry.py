import os
import json
import joblib
import numpy as np
import torch
from app.config import settings

class ArtifactRegistry:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance
    
    def load_all(self):
        if self._loaded:
            return
        
        base = settings.get_artifact_dir()
        self.base = base
        
        # Load preprocessor artifacts
        prep_dir = os.path.join(base, "preprocessor")
        self.label_encoder = joblib.load(os.path.join(prep_dir, "label_encoder.pkl"))
        self.scaler = joblib.load(os.path.join(prep_dir, "scaler.pkl"))
        with open(os.path.join(prep_dir, "feature_schema.json")) as f:
            self.feature_schema = json.load(f)
        with open(os.path.join(prep_dir, "imputer_config.json")) as f:
            self.imputer_config = json.load(f)
        
        # Load validation metrics
        with open(os.path.join(base, "metrics", "validation_metrics.json")) as f:
            self.metrics_registry = json.load(f)
        
        self.champion = self.metrics_registry["champion"]
        self.states = self.imputer_config["states"]
        
        # Load processed data for inference context
        import pandas as pd
        self.processed_data = pd.read_parquet(os.path.join(base, "data", "processed_data.parquet"))
        self.processed_data['Date'] = pd.to_datetime(self.processed_data['Date'])
        
        self.featured_data = pd.read_parquet(os.path.join(base, "data", "featured_data.parquet"))
        self.featured_data['Date'] = pd.to_datetime(self.featured_data['Date'])
        
        self._loaded = True
        print(f"[ArtifactRegistry] Loaded all artifacts. Champion: {self.champion}")
    
    def get_model_path(self, model_name: str) -> str:
        return os.path.join(self.base, "models", model_name)
    
    def get_ranking(self):
        return self.metrics_registry["ranking"]
    
    def get_model_metrics(self, model_name: str):
        return self.metrics_registry["detailed_metrics"].get(model_name, {}).get("overall", {})

registry = ArtifactRegistry()
