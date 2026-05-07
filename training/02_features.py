"""
Step 2: Feature Engineering
- Lag features, rolling stats, calendar features
- Train/validation split (last 8 weeks = validation)
- Save feature schema
"""
import pandas as pd
import numpy as np
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(BASE, "artifacts")
DATA_DIR = os.path.join(ARTIFACT_DIR, "data")
PREP_DIR = os.path.join(ARTIFACT_DIR, "preprocessor")

def load_processed_data():
    df = pd.read_parquet(os.path.join(DATA_DIR, "processed_data.parquet"))
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['State', 'Date']).reset_index(drop=True)
    return df

def add_lag_features(df):
    lags = [1, 2, 4, 7, 8, 13, 26, 30, 52]
    for lag in lags:
        df[f'lag_{lag}'] = df.groupby('State')['Total'].shift(lag)
    print(f"Added {len(lags)} lag features")
    return df

def add_rolling_features(df):
    windows = [4, 8, 13, 26, 52]
    for w in windows:
        df[f'rolling_mean_{w}'] = df.groupby('State')['Total'].transform(
            lambda x: x.shift(1).rolling(window=w, min_periods=1).mean()
        )
        df[f'rolling_std_{w}'] = df.groupby('State')['Total'].transform(
            lambda x: x.shift(1).rolling(window=w, min_periods=1).std()
        )
    # Fill NaN stds with 0
    std_cols = [c for c in df.columns if 'rolling_std' in c]
    df[std_cols] = df[std_cols].fillna(0)
    print(f"Added {len(windows)*2} rolling features")
    return df

def add_calendar_features(df):
    df['month'] = df['Date'].dt.month
    df['quarter'] = df['Date'].dt.quarter
    df['year'] = df['Date'].dt.year
    df['day_of_year'] = df['Date'].dt.dayofyear
    
    # Simple US holiday proximity (within 1 week of major holidays)
    us_holidays_md = [(1,1),(7,4),(11,25),(12,25),(1,15),(2,19),(5,27),(9,2),(10,14),(11,11)]
    def holiday_flag(date):
        for m, d in us_holidays_md:
            try:
                hol = pd.Timestamp(year=date.year, month=m, day=d)
                if abs((date - hol).days) <= 7:
                    return 1
            except:
                pass
        return 0
    
    df['is_holiday'] = df['Date'].apply(holiday_flag)
    print(f"Added calendar features + holiday flag")
    return df

def train_val_split(df):
    """Time-based split: last 8 weeks per state = validation."""
    df = df.sort_values(['State', 'Date']).reset_index(drop=True)
    
    # Find cutoff: 8 weeks before the last date
    max_date = df['Date'].max()
    cutoff = max_date - pd.Timedelta(weeks=8)
    
    train = df[df['Date'] <= cutoff].copy()
    val = df[df['Date'] > cutoff].copy()
    
    print(f"Train: {len(train)} rows (up to {cutoff.date()})")
    print(f"Validation: {len(val)} rows ({val['Date'].min().date()} to {val['Date'].max().date()})")
    return train, val

def save_feature_schema(df):
    feature_cols = [c for c in df.columns if c not in ['Date', 'State', 'Category', 'Total']]
    schema = {
        "target": "Total",
        "date_col": "Date",
        "state_col": "State",
        "feature_columns": feature_cols,
        "lag_features": [c for c in feature_cols if c.startswith('lag_')],
        "rolling_features": [c for c in feature_cols if c.startswith('rolling_')],
        "calendar_features": ["week_of_year", "month", "quarter", "year", "day_of_year", "is_holiday"],
        "missingness_features": ["missing_date_flag", "gap_length", "is_imputed"],
        "encoding_features": ["state_encoded"],
        "forecast_horizon": 8,
        "frequency": "W-SUN",
    }
    with open(os.path.join(PREP_DIR, "feature_schema.json"), 'w') as f:
        json.dump(schema, f, indent=2)
    print(f"Feature schema saved: {len(feature_cols)} features")

def main():
    print("=" * 60)
    print("STEP 2: FEATURE ENGINEERING")
    print("=" * 60)
    
    df = load_processed_data()
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_calendar_features(df)
    
    # Drop rows where lags are NaN (first 52 weeks per state)
    before = len(df)
    df = df.dropna(subset=['lag_52']).reset_index(drop=True)
    print(f"Dropped {before - len(df)} rows with insufficient lag history")
    
    save_feature_schema(df)
    
    train, val = train_val_split(df)
    
    # Save
    df.to_parquet(os.path.join(DATA_DIR, "featured_data.parquet"), index=False)
    train.to_parquet(os.path.join(DATA_DIR, "train.parquet"), index=False)
    val.to_parquet(os.path.join(DATA_DIR, "val.parquet"), index=False)
    
    print(f"\nSaved: featured_data.parquet, train.parquet, val.parquet")
    print("FEATURE ENGINEERING COMPLETE")

if __name__ == "__main__":
    main()
