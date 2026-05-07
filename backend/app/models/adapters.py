"""Model adapters - each model family has its own inference adapter."""
import os
import json
import joblib
import numpy as np
import pandas as pd
import time
from abc import ABC, abstractmethod

class BaseAdapter(ABC):
    name: str = ""
    
    @abstractmethod
    def load(self, model_dir: str, states: list):
        pass
    
    @abstractmethod
    def predict(self, state: str, horizon: int, context_data: pd.DataFrame) -> list:
        """Return list of dicts: [{forecast_date, predicted_value, confidence_lower, confidence_upper}]"""
        pass
    
    def is_available(self, state: str) -> bool:
        return True

class SARIMAAdapter(BaseAdapter):
    name = "sarima"
    
    def load(self, model_dir, states):
        self._model_dir = os.path.join(model_dir, "sarima")
        self._available_states = []
        self._models = {}
        for state in states:
            fpath = os.path.join(self._model_dir, f"{state}.pkl")
            if os.path.exists(fpath):
                self._available_states.append(state)
    
    def is_available(self, state):
        return state in self._available_states
    
    def _get_model(self, state):
        if state not in self._models:
            fpath = os.path.join(self._model_dir, f"{state}.pkl")
            try:
                self._models[state] = joblib.load(fpath)
            except Exception as e:
                print(f"[SARIMAAdapter] Failed to load model for {state}: {e}")
                self._available_states = [s for s in self._available_states if s != state]
                return None
        return self._models[state]
    
    def predict(self, state, horizon, context_data):
        model = self._get_model(state)
        if model is None:
            return []
        
        try:
            forecast = model.forecast(steps=horizon)
            forecast = np.clip(forecast, 0, None)
        except Exception as e:
            print(f"[SARIMAAdapter] Forecast failed for {state}: {e}")
            return []
        
        last_date = context_data[context_data['State'] == state]['Date'].max()
        dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=horizon, freq='W-SUN')
        
        results = []
        for i, (d, v) in enumerate(zip(dates, forecast)):
            results.append({
                "forecast_date": d.strftime("%Y-%m-%d"),
                "predicted_value": round(float(v), 2),
                "confidence_lower": round(float(v * 0.85), 2),
                "confidence_upper": round(float(v * 1.15), 2),
            })
        
        # Free memory after prediction
        if state in self._models:
            del self._models[state]
        
        return results

class ProphetAdapter(BaseAdapter):
    name = "prophet"
    
    def load(self, model_dir, states):
        self.models = {}
        path = os.path.join(model_dir, "prophet")
        for state in states:
            fpath = os.path.join(path, f"{state}.pkl")
            if os.path.exists(fpath):
                self.models[state] = joblib.load(fpath)
    
    def is_available(self, state):
        return state in self.models
    
    def predict(self, state, horizon, context_data):
        model = self.models[state]
        future = model.make_future_dataframe(periods=horizon, freq='W-SUN')
        forecast = model.predict(future)
        preds = forecast.tail(horizon)
        
        results = []
        for _, row in preds.iterrows():
            results.append({
                "forecast_date": row['ds'].strftime("%Y-%m-%d"),
                "predicted_value": round(max(float(row['yhat']), 0), 2),
                "confidence_lower": round(max(float(row['yhat_lower']), 0), 2),
                "confidence_upper": round(float(row['yhat_upper']), 2),
            })
        return results

class ETSAdapter(BaseAdapter):
    name = "ets"
    
    def load(self, model_dir, states):
        self.models = {}
        path = os.path.join(model_dir, "ets")
        for state in states:
            fpath = os.path.join(path, f"{state}.pkl")
            if os.path.exists(fpath):
                self.models[state] = joblib.load(fpath)
    
    def is_available(self, state):
        return state in self.models
    
    def predict(self, state, horizon, context_data):
        model = self.models[state]
        forecast = model.forecast(steps=horizon)
        forecast = np.clip(forecast, 0, None)
        
        last_date = context_data[context_data['State'] == state]['Date'].max()
        dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=horizon, freq='W-SUN')
        
        results = []
        for d, v in zip(dates, forecast):
            results.append({
                "forecast_date": d.strftime("%Y-%m-%d"),
                "predicted_value": round(float(v), 2),
                "confidence_lower": round(float(v * 0.88), 2),
                "confidence_upper": round(float(v * 1.12), 2),
            })
        return results

class XGBoostAdapter(BaseAdapter):
    name = "xgboost"
    
    def load(self, model_dir, states):
        import xgboost as xgb
        path = os.path.join(model_dir, "xgboost", "xgboost_model.json")
        self.model = xgb.XGBRegressor()
        self.model.load_model(path)
    
    def predict(self, state, horizon, context_data):
        from app.services.artifact_registry import registry
        
        state_data = registry.featured_data[registry.featured_data['State'] == state].sort_values('Date')
        feature_cols = registry.feature_schema['feature_columns']
        
        last_date = state_data['Date'].max()
        dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=horizon, freq='W-SUN')
        
        # Use last row's features as base, shift for multi-step
        last_features = state_data[feature_cols].iloc[-1:].values
        pred = self.model.predict(last_features)[0]
        pred = max(pred, 0)
        
        results = []
        for i, d in enumerate(dates):
            # Simple decay factor for future uncertainty
            factor = 1.0 + (i * 0.005)
            results.append({
                "forecast_date": d.strftime("%Y-%m-%d"),
                "predicted_value": round(float(pred * factor), 2),
                "confidence_lower": round(float(pred * factor * 0.87), 2),
                "confidence_upper": round(float(pred * factor * 1.13), 2),
            })
        return results

class LSTMAdapter(BaseAdapter):
    name = "lstm"
    
    def load(self, model_dir, states):
        import torch, torch.nn as nn
        
        path = os.path.join(model_dir, "lstm")
        with open(os.path.join(path, "config.json")) as f:
            self.config = json.load(f)
        
        class SalesLSTM(nn.Module):
            def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
                self.fc = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1))
            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :]).squeeze(-1)
        
        self.model = SalesLSTM(self.config['input_size'], self.config['hidden_size'],
                               self.config['num_layers'], self.config['dropout'])
        self.model.load_state_dict(torch.load(os.path.join(path, "lstm_model.pt"), 
                                              map_location='cpu', weights_only=True))
        self.model.eval()
        self.feat_scaler = joblib.load(os.path.join(path, "feat_scaler.pkl"))
    
    def predict(self, state, horizon, context_data):
        import torch
        from app.services.artifact_registry import registry
        
        state_data = registry.featured_data[registry.featured_data['State'] == state].sort_values('Date')
        feature_cols = self.config['feature_cols']
        seq_len = self.config['seq_len']
        
        last_seq = state_data[feature_cols].tail(seq_len).values.astype(np.float32)
        last_seq = self.feat_scaler.transform(last_seq.reshape(-1, len(feature_cols))).reshape(1, seq_len, -1)
        last_seq = np.nan_to_num(last_seq, nan=0.0)
        
        with torch.no_grad():
            pred_scaled = self.model(torch.tensor(last_seq)).cpu().numpy()
        pred = registry.scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]
        pred = max(pred, 0)
        
        last_date = state_data['Date'].max()
        dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=horizon, freq='W-SUN')
        
        results = []
        for i, d in enumerate(dates):
            factor = 1.0 + (i * 0.003)
            results.append({
                "forecast_date": d.strftime("%Y-%m-%d"),
                "predicted_value": round(float(pred * factor), 2),
                "confidence_lower": round(float(pred * factor * 0.82), 2),
                "confidence_upper": round(float(pred * factor * 1.18), 2),
            })
        return results

class TFTAdapter(BaseAdapter):
    name = "tft"
    
    def load(self, model_dir, states):
        import torch, torch.nn as nn
        
        path = os.path.join(model_dir, "tft")
        with open(os.path.join(path, "config.json")) as f:
            self.config = json.load(f)
        
        class TemporalFusionModel(nn.Module):
            def __init__(self, input_size, hidden=64, heads=4, dropout=0.1):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden, num_layers=2, batch_first=True, dropout=dropout)
                self.attention = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
                self.gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GLU(dim=-1))
                self.fc = nn.Sequential(nn.LayerNorm(hidden // 2), nn.Linear(hidden // 2, 32),
                                       nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
                combined = torch.cat([lstm_out[:, -1, :], attn_out[:, -1, :]], dim=-1)
                return self.fc(self.gate(combined)).squeeze(-1)
        
        self.model = TemporalFusionModel(self.config['input_size'], self.config['hidden'],
                                         self.config['heads'], self.config['dropout'])
        self.model.load_state_dict(torch.load(os.path.join(path, "tft_model.pt"),
                                              map_location='cpu', weights_only=True))
        self.model.eval()
        self.feat_scaler = joblib.load(os.path.join(path, "feat_scaler.pkl"))
    
    def predict(self, state, horizon, context_data):
        import torch
        from app.services.artifact_registry import registry
        
        state_data = registry.featured_data[registry.featured_data['State'] == state].sort_values('Date')
        feature_cols = self.config['feature_cols']
        seq_len = self.config['seq_len']
        
        last_seq = state_data[feature_cols].tail(seq_len).values.astype(np.float32)
        last_seq = self.feat_scaler.transform(last_seq.reshape(-1, len(feature_cols))).reshape(1, seq_len, -1)
        last_seq = np.nan_to_num(last_seq, nan=0.0)
        
        with torch.no_grad():
            pred_scaled = self.model(torch.tensor(last_seq)).cpu().numpy()
        pred = registry.scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]
        pred = max(pred, 0)
        
        last_date = state_data['Date'].max()
        dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=horizon, freq='W-SUN')
        
        results = []
        for i, d in enumerate(dates):
            factor = 1.0 + (i * 0.004)
            results.append({
                "forecast_date": d.strftime("%Y-%m-%d"),
                "predicted_value": round(float(pred * factor), 2),
                "confidence_lower": round(float(pred * factor * 0.80), 2),
                "confidence_upper": round(float(pred * factor * 1.20), 2),
            })
        return results

def get_all_adapters():
    return [SARIMAAdapter(), ProphetAdapter(), ETSAdapter(), 
            XGBoostAdapter(), LSTMAdapter(), TFTAdapter()]
