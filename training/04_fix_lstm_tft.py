import pandas as pd
import numpy as np
import json
import os
import joblib
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler as SS
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(BASE, "artifacts")
DATA_DIR = os.path.join(ARTIFACT_DIR, "data")
PREP_DIR = os.path.join(ARTIFACT_DIR, "preprocessor")
MODEL_DIR = os.path.join(ARTIFACT_DIR, "models")
METRIC_DIR = os.path.join(ARTIFACT_DIR, "metrics")

def wmape(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    mask = actual != 0
    if mask.sum() == 0: return 0.0
    return np.sum(np.abs(actual[mask] - predicted[mask])) / np.sum(np.abs(actual[mask]))

def compute_metrics(actual, predicted):
    actual, predicted = np.array(actual, dtype=float), np.array(predicted, dtype=float)
    return {
        "wMAPE": round(float(wmape(actual, predicted)), 6),
        "RMSE": round(float(np.sqrt(np.mean((actual - predicted) ** 2))), 2),
        "MAE": round(float(np.mean(np.abs(actual - predicted))), 2)
    }

def load_data():
    full = pd.read_parquet(os.path.join(DATA_DIR, "featured_data.parquet"))
    full['Date'] = pd.to_datetime(full['Date'])
    return full

def make_sequences_with_split(df, feature_cols, seq_len, val_start):
    """Build sequences from full data, then split by whether the TARGET date falls in val."""
    X_train, y_train, X_val, y_val = [], [], [], []
    for state in sorted(df['State'].unique()):
        sdf = df[df['State'] == state].sort_values('Date').reset_index(drop=True)
        feats = sdf[feature_cols].values
        targets = sdf['Total'].values
        dates = sdf['Date'].values
        for j in range(seq_len, len(sdf)):
            seq = feats[j-seq_len:j]
            target = targets[j]
            if pd.Timestamp(dates[j]) >= val_start:
                X_val.append(seq)
                y_val.append(target)
            else:
                X_train.append(seq)
                y_train.append(target)
    return (np.array(X_train, dtype=np.float32), np.array(y_train, dtype=np.float32),
            np.array(X_val, dtype=np.float32), np.array(y_val, dtype=np.float32))

class SalesLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)

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

def retrain_model(model_class, model_kwargs, save_name, df, feature_cols, epochs):
    SEQ_LEN = 8
    scaler = joblib.load(os.path.join(PREP_DIR, "scaler.pkl"))
    
    max_date = df['Date'].max()
    val_start = max_date - pd.Timedelta(weeks=8)
    
    X_train, y_train, X_val, y_val = make_sequences_with_split(df, feature_cols, SEQ_LEN, val_start)
    print(f"  Train sequences: {len(X_train)}, Val sequences: {len(X_val)}")
    
    y_train_scaled = scaler.transform(y_train.reshape(-1, 1)).flatten()
    
    # Fit feature scaler on train only
    n, seq, feat = X_train.shape
    feat_scaler = SS()
    feat_scaler.fit(X_train.reshape(-1, feat))
    X_train_s = feat_scaler.transform(X_train.reshape(-1, feat)).reshape(n, seq, feat).astype(np.float32)
    X_train_s = np.nan_to_num(X_train_s, nan=0.0, posinf=0.0, neginf=0.0)
    y_train_scaled = np.nan_to_num(y_train_scaled, nan=0.0)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_class(**model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.HuberLoss()
    
    dataset = TensorDataset(torch.tensor(X_train_s), torch.tensor(y_train_scaled))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
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
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.6f}")
    

    nv = len(X_val)
    X_val_s = feat_scaler.transform(X_val.reshape(-1, feat)).reshape(nv, SEQ_LEN, feat).astype(np.float32)
    X_val_s = np.nan_to_num(X_val_s, nan=0.0)
    
    model.eval()
    with torch.no_grad():
        preds_scaled = model(torch.tensor(X_val_s).to(device)).cpu().numpy()
    preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    preds = np.clip(preds, 0, None)
    
    metrics = compute_metrics(y_val, preds)
    print(f"  {save_name} Metrics: {metrics}")

    save_dir = os.path.join(MODEL_DIR, save_name)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, f"{save_name}_model.pt"))
    joblib.dump(feat_scaler, os.path.join(save_dir, "feat_scaler.pkl"))
    config = {**model_kwargs, "seq_len": SEQ_LEN, "feature_cols": feature_cols}
    with open(os.path.join(save_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    return metrics

def main():
    print("=" * 60)
    print("FIXING LSTM & TFT VALIDATION")
    print("=" * 60)
    
    df = load_data()
    feature_cols = [c for c in df.columns if c not in ['Date', 'State', 'Category', 'Total']]
    input_size = len(feature_cols)
    
    print("\n--- Retraining LSTM ---")
    lstm_metrics = retrain_model(
        SalesLSTM, {"input_size": input_size, "hidden_size": 64, "num_layers": 2, "dropout": 0.2},
        "lstm", df, feature_cols, epochs=80
    )
    
    print("\n--- Retraining TFT ---")
    tft_metrics = retrain_model(
        TemporalFusionModel, {"input_size": input_size, "hidden": 64, "heads": 4, "dropout": 0.1},
        "tft", df, feature_cols, epochs=80
    )
    

    metrics_path = os.path.join(METRIC_DIR, "validation_metrics.json")
    with open(metrics_path) as f:
        registry = json.load(f)
    
    registry["detailed_metrics"]["lstm"]["overall"] = lstm_metrics
    registry["detailed_metrics"]["tft"]["overall"] = tft_metrics
    

    leaderboard = {name: m["overall"] for name, m in registry["detailed_metrics"].items()}
    ranked = sorted(leaderboard.items(), key=lambda x: x[1].get("wMAPE", 999))
    registry["champion"] = ranked[0][0]
    registry["ranking"] = [{"rank": i+1, "model": n, "metrics": m} for i, (n, m) in enumerate(ranked)]
    
    with open(metrics_path, 'w') as f:
        json.dump(registry, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"Champion: {registry['champion']}")
    for r in registry["ranking"]:
        tag = " ***" if r["model"] == registry["champion"] else ""
        print(f"  #{r['rank']} {r['model']}: wMAPE={r['metrics']['wMAPE']:.4f}{tag}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
