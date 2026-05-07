"""
Step 3: Model Training - SARIMA, Prophet, ETS, XGBoost, LSTM, TFT
All models trained offline, saved as artifacts.
"""
import pandas as pd
import numpy as np
import json
import os
import sys
import joblib
import warnings
import time
import traceback
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(BASE, "artifacts")
DATA_DIR = os.path.join(ARTIFACT_DIR, "data")
PREP_DIR = os.path.join(ARTIFACT_DIR, "preprocessor")
MODEL_DIR = os.path.join(ARTIFACT_DIR, "models")
METRIC_DIR = os.path.join(ARTIFACT_DIR, "metrics")

def load_data():
    proc = pd.read_parquet(os.path.join(DATA_DIR, "processed_data.parquet"))
    train = pd.read_parquet(os.path.join(DATA_DIR, "train.parquet"))
    val = pd.read_parquet(os.path.join(DATA_DIR, "val.parquet"))
    for d in [proc, train, val]:
        d['Date'] = pd.to_datetime(d['Date'])
    return proc, train, val

def wmape(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    mask = actual != 0
    if mask.sum() == 0:
        return 0.0
    return np.sum(np.abs(actual[mask] - predicted[mask])) / np.sum(np.abs(actual[mask]))

def compute_metrics(actual, predicted):
    actual, predicted = np.array(actual, dtype=float), np.array(predicted, dtype=float)
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    w = wmape(actual, predicted)
    return {"wMAPE": round(float(w), 6), "RMSE": round(float(rmse), 2), "MAE": round(float(mae), 2)}

# ===================== SARIMA =====================
def train_sarima(proc, val):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    print("\n--- Training SARIMA (per-state) ---")
    save_dir = os.path.join(MODEL_DIR, "sarima")
    os.makedirs(save_dir, exist_ok=True)
    
    states = sorted(proc['State'].unique())
    all_preds, all_actuals, state_metrics = [], [], {}
    
    val_start = val['Date'].min()
    
    for i, state in enumerate(states):
        try:
            state_data = proc[proc['State'] == state].sort_values('Date').set_index('Date')['Total']
            state_data = state_data.asfreq('W-SUN')
            
            train_data = state_data[state_data.index < val_start]
            val_data = state_data[(state_data.index >= val_start)][:8]
            
            model = SARIMAX(train_data, order=(1,1,1), seasonal_order=(1,1,0,52),
                           enforce_stationarity=False, enforce_invertibility=False)
            fitted = model.fit(disp=False, maxiter=200)
            
            preds = fitted.forecast(steps=len(val_data))
            preds = preds.clip(lower=0)
            
            joblib.dump(fitted, os.path.join(save_dir, f"{state}.pkl"))
            
            if len(val_data) > 0:
                m = compute_metrics(val_data.values, preds.values)
                state_metrics[state] = m
                all_preds.extend(preds.values)
                all_actuals.extend(val_data.values)
            
            sys.stdout.write(f"\r  SARIMA: {i+1}/{len(states)} states done")
            sys.stdout.flush()
        except Exception as e:
            print(f"\n  Warning: SARIMA failed for {state}: {e}")
            state_metrics[state] = {"wMAPE": 999, "RMSE": 999, "MAE": 999, "error": str(e)}
    
    overall = compute_metrics(all_actuals, all_preds)
    print(f"\n  SARIMA Overall: {overall}")
    return {"overall": overall, "per_state": state_metrics}

# ===================== PROPHET =====================
def train_prophet(proc, val):
    from prophet import Prophet
    print("\n--- Training Prophet (per-state) ---")
    save_dir = os.path.join(MODEL_DIR, "prophet")
    os.makedirs(save_dir, exist_ok=True)
    
    states = sorted(proc['State'].unique())
    all_preds, all_actuals, state_metrics = [], [], {}
    val_start = val['Date'].min()
    
    for i, state in enumerate(states):
        try:
            state_data = proc[proc['State'] == state][['Date', 'Total']].copy()
            state_data.columns = ['ds', 'y']
            state_data = state_data.sort_values('ds')
            
            train_data = state_data[state_data['ds'] < val_start]
            val_data = state_data[state_data['ds'] >= val_start].head(8)
            
            model = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                          daily_seasonality=False, changepoint_prior_scale=0.05)
            model.fit(train_data)
            
            future = model.make_future_dataframe(periods=8, freq='W-SUN')
            forecast = model.predict(future)
            preds = forecast.tail(len(val_data))['yhat'].values
            preds = np.clip(preds, 0, None)
            
            joblib.dump(model, os.path.join(save_dir, f"{state}.pkl"))
            
            if len(val_data) > 0:
                m = compute_metrics(val_data['y'].values, preds[:len(val_data)])
                state_metrics[state] = m
                all_preds.extend(preds[:len(val_data)])
                all_actuals.extend(val_data['y'].values)
            
            sys.stdout.write(f"\r  Prophet: {i+1}/{len(states)} states done")
            sys.stdout.flush()
        except Exception as e:
            print(f"\n  Warning: Prophet failed for {state}: {e}")
            state_metrics[state] = {"wMAPE": 999, "RMSE": 999, "MAE": 999, "error": str(e)}
    
    overall = compute_metrics(all_actuals, all_preds)
    print(f"\n  Prophet Overall: {overall}")
    return {"overall": overall, "per_state": state_metrics}

# ===================== ETS =====================
def train_ets(proc, val):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    print("\n--- Training ETS (per-state) ---")
    save_dir = os.path.join(MODEL_DIR, "ets")
    os.makedirs(save_dir, exist_ok=True)
    
    states = sorted(proc['State'].unique())
    all_preds, all_actuals, state_metrics = [], [], {}
    val_start = val['Date'].min()
    
    for i, state in enumerate(states):
        try:
            state_data = proc[proc['State'] == state].sort_values('Date').set_index('Date')['Total']
            state_data = state_data.asfreq('W-SUN')
            
            train_data = state_data[state_data.index < val_start]
            val_data = state_data[(state_data.index >= val_start)][:8]
            
            model = ExponentialSmoothing(train_data, trend='add', seasonal='add',
                                        seasonal_periods=52, damped_trend=True)
            fitted = model.fit(optimized=True, use_brute=False)
            
            preds = fitted.forecast(steps=len(val_data))
            preds = preds.clip(lower=0)
            
            joblib.dump(fitted, os.path.join(save_dir, f"{state}.pkl"))
            
            if len(val_data) > 0:
                m = compute_metrics(val_data.values, preds.values)
                state_metrics[state] = m
                all_preds.extend(preds.values)
                all_actuals.extend(val_data.values)
            
            sys.stdout.write(f"\r  ETS: {i+1}/{len(states)} states done")
            sys.stdout.flush()
        except Exception as e:
            print(f"\n  Warning: ETS failed for {state}: {e}")
            state_metrics[state] = {"wMAPE": 999, "RMSE": 999, "MAE": 999, "error": str(e)}
    
    overall = compute_metrics(all_actuals, all_preds)
    print(f"\n  ETS Overall: {overall}")
    return {"overall": overall, "per_state": state_metrics}

# ===================== XGBOOST =====================
def train_xgboost(train, val):
    import xgboost as xgb
    print("\n--- Training XGBoost (global) ---")
    save_dir = os.path.join(MODEL_DIR, "xgboost")
    os.makedirs(save_dir, exist_ok=True)
    
    feature_cols = [c for c in train.columns if c not in ['Date', 'State', 'Category', 'Total']]
    
    X_train = train[feature_cols].values
    y_train = train['Total'].values
    X_val = val[feature_cols].values
    y_val = val['Total'].values
    
    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        min_child_weight=5, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    preds = model.predict(X_val)
    preds = np.clip(preds, 0, None)
    
    model.save_model(os.path.join(save_dir, "xgboost_model.json"))
    
    # Per-state metrics
    state_metrics = {}
    for state in sorted(val['State'].unique()):
        mask = val['State'] == state
        if mask.sum() > 0:
            m = compute_metrics(y_val[mask], preds[mask])
            state_metrics[state] = m
    
    overall = compute_metrics(y_val, preds)
    print(f"  XGBoost Overall: {overall}")
    
    # Save feature importance
    imp = dict(zip(feature_cols, model.feature_importances_.tolist()))
    with open(os.path.join(save_dir, "feature_importance.json"), 'w') as f:
        json.dump(imp, f, indent=2)
    
    return {"overall": overall, "per_state": state_metrics}

# ===================== LSTM =====================
def train_lstm(train, val):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    print("\n--- Training LSTM (global) ---")
    save_dir = os.path.join(MODEL_DIR, "lstm")
    os.makedirs(save_dir, exist_ok=True)
    
    feature_cols = [c for c in train.columns if c not in ['Date', 'State', 'Category', 'Total']]
    scaler = joblib.load(os.path.join(PREP_DIR, "scaler.pkl"))
    
    # Prepare sequences per state
    SEQ_LEN = 8
    
    def make_sequences(df):
        X_all, y_all = [], []
        for state in sorted(df['State'].unique()):
            sdf = df[df['State'] == state].sort_values('Date')
            feats = sdf[feature_cols].values
            target = sdf['Total'].values
            for j in range(SEQ_LEN, len(sdf)):
                X_all.append(feats[j-SEQ_LEN:j])
                y_all.append(target[j])
        return np.array(X_all, dtype=np.float32), np.array(y_all, dtype=np.float32)
    
    X_train, y_train = make_sequences(train)
    
    # Scale targets
    y_train_scaled = scaler.transform(y_train.reshape(-1, 1)).flatten()
    
    # Normalize features
    from sklearn.preprocessing import StandardScaler as SS
    feat_scaler = SS()
    n, seq, feat = X_train.shape
    X_flat = X_train.reshape(-1, feat)
    feat_scaler.fit(X_flat)
    X_train = feat_scaler.transform(X_flat).reshape(n, seq, feat).astype(np.float32)
    joblib.dump(feat_scaler, os.path.join(save_dir, "feat_scaler.pkl"))
    
    # Handle NaN/inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    y_train_scaled = np.nan_to_num(y_train_scaled, nan=0.0)
    
    class SalesLSTM(nn.Module):
        def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
            self.fc = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1))
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :]).squeeze(-1)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SalesLSTM(input_size=len(feature_cols)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train_scaled))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model.train()
    for epoch in range(50):
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/50, Loss: {total_loss/len(loader):.6f}")
    
    # Validate
    X_val_seq, y_val_actual = make_sequences(val)
    if len(X_val_seq) > 0:
        X_val_seq = feat_scaler.transform(X_val_seq.reshape(-1, feat)).reshape(-1, SEQ_LEN, feat).astype(np.float32)
        X_val_seq = np.nan_to_num(X_val_seq, nan=0.0)
        model.eval()
        with torch.no_grad():
            preds_scaled = model(torch.tensor(X_val_seq).to(device)).cpu().numpy()
        preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        preds = np.clip(preds, 0, None)
        overall = compute_metrics(y_val_actual, preds)
    else:
        overall = {"wMAPE": 999, "RMSE": 999, "MAE": 999}
        preds = np.array([])
    
    print(f"  LSTM Overall: {overall}")
    
    # Save
    torch.save(model.state_dict(), os.path.join(save_dir, "lstm_model.pt"))
    config = {"input_size": len(feature_cols), "hidden_size": 64, "num_layers": 2, 
              "dropout": 0.2, "seq_len": SEQ_LEN, "feature_cols": feature_cols}
    with open(os.path.join(save_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    return {"overall": overall, "per_state": {}}

# ===================== TFT (Simplified Attention LSTM) =====================
def train_tft(train, val):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    print("\n--- Training TFT (Temporal Fusion - simplified) ---")
    save_dir = os.path.join(MODEL_DIR, "tft")
    os.makedirs(save_dir, exist_ok=True)
    
    feature_cols = [c for c in train.columns if c not in ['Date', 'State', 'Category', 'Total']]
    scaler = joblib.load(os.path.join(PREP_DIR, "scaler.pkl"))
    
    SEQ_LEN = 8
    
    def make_sequences(df):
        X_all, y_all = [], []
        for state in sorted(df['State'].unique()):
            sdf = df[df['State'] == state].sort_values('Date')
            feats = sdf[feature_cols].values
            target = sdf['Total'].values
            for j in range(SEQ_LEN, len(sdf)):
                X_all.append(feats[j-SEQ_LEN:j])
                y_all.append(target[j])
        return np.array(X_all, dtype=np.float32), np.array(y_all, dtype=np.float32)
    
    X_train, y_train = make_sequences(train)
    y_train_scaled = scaler.transform(y_train.reshape(-1, 1)).flatten()
    
    # Load LSTM feature scaler
    lstm_dir = os.path.join(MODEL_DIR, "lstm")
    feat_scaler = joblib.load(os.path.join(lstm_dir, "feat_scaler.pkl"))
    n, seq, feat_dim = X_train.shape
    X_train = feat_scaler.transform(X_train.reshape(-1, feat_dim)).reshape(n, seq, feat_dim).astype(np.float32)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    y_train_scaled = np.nan_to_num(y_train_scaled, nan=0.0)
    
    class TemporalFusionModel(nn.Module):
        """Simplified TFT: LSTM encoder + multi-head attention + gated output."""
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
            gated = self.gate(combined)
            return self.fc(gated).squeeze(-1)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TemporalFusionModel(input_size=len(feature_cols)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.HuberLoss()
    
    dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train_scaled))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model.train()
    for epoch in range(60):
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/60, Loss: {total_loss/len(loader):.6f}")
    
    # Validate
    X_val_seq, y_val_actual = make_sequences(val)
    if len(X_val_seq) > 0:
        X_val_seq = feat_scaler.transform(X_val_seq.reshape(-1, feat_dim)).reshape(-1, SEQ_LEN, feat_dim).astype(np.float32)
        X_val_seq = np.nan_to_num(X_val_seq, nan=0.0)
        model.eval()
        with torch.no_grad():
            preds_scaled = model(torch.tensor(X_val_seq).to(device)).cpu().numpy()
        preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        preds = np.clip(preds, 0, None)
        overall = compute_metrics(y_val_actual, preds)
    else:
        overall = {"wMAPE": 999, "RMSE": 999, "MAE": 999}
    
    print(f"  TFT Overall: {overall}")
    
    torch.save(model.state_dict(), os.path.join(save_dir, "tft_model.pt"))
    # Copy feat_scaler for TFT too
    joblib.dump(feat_scaler, os.path.join(save_dir, "feat_scaler.pkl"))
    config = {"input_size": len(feature_cols), "hidden_size": 64, "num_heads": 4,
              "num_layers": 2, "dropout": 0.1, "seq_len": SEQ_LEN, "feature_cols": feature_cols}
    with open(os.path.join(save_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    return {"overall": overall, "per_state": {}}

def main():
    print("=" * 60)
    print("STEP 3: MODEL TRAINING")
    print("=" * 60)
    
    proc, train, val = load_data()
    all_metrics = {}
    
    t0 = time.time()
    
    # Train each model
    all_metrics["sarima"] = train_sarima(proc, val)
    all_metrics["prophet"] = train_prophet(proc, val)
    all_metrics["ets"] = train_ets(proc, val)
    all_metrics["xgboost"] = train_xgboost(train, val)
    all_metrics["lstm"] = train_lstm(train, val)
    all_metrics["tft"] = train_tft(train, val)
    
    # Save validation metrics registry
    os.makedirs(METRIC_DIR, exist_ok=True)
    
    # Determine champion
    leaderboard = {}
    for name, m in all_metrics.items():
        leaderboard[name] = m["overall"]
    
    # Sort by wMAPE (primary)
    ranked = sorted(leaderboard.items(), key=lambda x: x[1].get("wMAPE", 999))
    champion = ranked[0][0]
    
    registry = {
        "champion": champion,
        "ranking": [{"rank": i+1, "model": name, "metrics": metrics} for i, (name, metrics) in enumerate(ranked)],
        "detailed_metrics": all_metrics,
        "selection_criteria": "Primary: wMAPE (lower is better). Secondary: RMSE, MAE.",
        "validation_window": "Last 8 weeks of dataset",
    }
    
    with open(os.path.join(METRIC_DIR, "validation_metrics.json"), 'w') as f:
        json.dump(registry, f, indent=2, default=str)
    
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE in {elapsed:.1f}s")
    print(f"Champion Model: {champion}")
    print(f"\nLeaderboard:")
    for i, (name, metrics) in enumerate(ranked):
        tag = " *** CHAMPION ***" if name == champion else ""
        print(f"  #{i+1} {name}: wMAPE={metrics['wMAPE']:.4f}, RMSE={metrics['RMSE']:.0f}, MAE={metrics['MAE']:.0f}{tag}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
