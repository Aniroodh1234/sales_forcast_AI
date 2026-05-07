"""
Step 1: Data Preprocessing
- Load raw Excel, parse dates, normalize to week-ending Sunday
- Fill missing weeks, impute gaps, handle outliers
- Save processed data + preprocessing artifacts
"""
import pandas as pd
import numpy as np
import json
import os
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(BASE, "artifacts")
DATA_DIR = os.path.join(ARTIFACT_DIR, "data")
PREP_DIR = os.path.join(ARTIFACT_DIR, "preprocessor")

def load_raw_data():
    path = os.path.join(BASE, "dataset", "Forecasting Case- Study.xlsx")
    df = pd.read_excel(path)
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True)
    df = df.sort_values(['State', 'Date']).reset_index(drop=True)
    print(f"Loaded {len(df)} rows, {df['State'].nunique()} states")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    return df

def normalize_to_weekly_sunday(df):
    """Anchor every timestamp to the nearest week-ending Sunday."""
    df['Date'] = df['Date'].dt.to_period('W-SUN').dt.end_time.dt.normalize()
    # Remove duplicates after normalization (keep mean if duplicated)
    df = df.groupby(['State', 'Date'], as_index=False).agg({'Total': 'mean', 'Category': 'first'})
    df = df.sort_values(['State', 'Date']).reset_index(drop=True)
    print(f"After weekly normalization: {len(df)} rows")
    return df

def fill_missing_weeks(df):
    """Create continuous weekly timeline per state, mark missing dates."""
    all_states = df['State'].unique()
    date_min = df['Date'].min()
    date_max = df['Date'].max()
    full_dates = pd.date_range(start=date_min, end=date_max, freq='W-SUN')
    
    frames = []
    for state in all_states:
        state_df = df[df['State'] == state].set_index('Date')
        state_full = state_df.reindex(full_dates)
        state_full['State'] = state
        state_full['Category'] = 'Beverages'
        state_full['missing_date_flag'] = state_full['Total'].isna().astype(int)
        state_full.index.name = 'Date'
        state_full = state_full.reset_index()
        frames.append(state_full)
    
    result = pd.concat(frames, ignore_index=True)
    
    # Compute gap_length
    result['gap_length'] = 0
    for state in all_states:
        mask = result['State'] == state
        missing = result.loc[mask, 'missing_date_flag'].values
        gaps = np.zeros(len(missing), dtype=int)
        count = 0
        for i in range(len(missing)):
            if missing[i] == 1:
                count += 1
                gaps[i] = count
            else:
                count = 0
        result.loc[mask, 'gap_length'] = gaps
    
    total_missing = result['missing_date_flag'].sum()
    print(f"After gap fill: {len(result)} rows, {total_missing} missing entries filled")
    return result

def impute_missing_values(df):
    """Impute missing sales: short gaps via interpolation, seasonal gaps via same-week-prior-year."""
    df['is_imputed'] = 0
    df['week_of_year'] = df['Date'].dt.isocalendar().week.astype(int)
    
    for state in df['State'].unique():
        mask = df['State'] == state
        state_idx = df[mask].index
        
        # First pass: interpolation for short gaps (<=3 weeks)
        total = df.loc[state_idx, 'Total'].copy()
        missing = total.isna()
        
        # Interpolate all, then we'll handle seasonal separately
        total_interp = total.interpolate(method='linear', limit=3)
        
        # For gaps > 3, use seasonal (same week prior year)
        still_missing = total_interp.isna()
        if still_missing.any():
            for idx in state_idx[still_missing]:
                woy = df.loc[idx, 'week_of_year']
                hist = df.loc[(df['State'] == state) & (df['week_of_year'] == woy) & df['Total'].notna(), 'Total']
                if len(hist) > 0:
                    total_interp.loc[idx] = hist.mean()
                else:
                    total_interp.loc[idx] = df.loc[mask & df['Total'].notna(), 'Total'].mean()
        
        # Mark imputed
        imputed_mask = missing & total_interp.notna()
        df.loc[state_idx[imputed_mask], 'is_imputed'] = 1
        df.loc[state_idx, 'Total'] = total_interp
    
    remaining_na = df['Total'].isna().sum()
    print(f"After imputation: {remaining_na} remaining NAs, {df['is_imputed'].sum()} imputed values")
    
    # Final fallback
    if remaining_na > 0:
        df['Total'] = df.groupby('State')['Total'].transform(lambda x: x.fillna(x.mean()))
    
    return df

def handle_outliers(df):
    """Conservative IQR-based outlier clipping per state. Preserve genuine spikes."""
    clip_count = 0
    for state in df['State'].unique():
        mask = df['State'] == state
        vals = df.loc[mask, 'Total']
        Q1 = vals.quantile(0.05)
        Q3 = vals.quantile(0.95)
        IQR = Q3 - Q1
        lower = Q1 - 2.0 * IQR  # Conservative: 2x IQR
        upper = Q3 + 2.0 * IQR
        
        outliers = (vals < lower) | (vals > upper)
        clip_count += outliers.sum()
        df.loc[mask, 'Total'] = vals.clip(lower=max(lower, 0), upper=upper)
    
    print(f"Outlier clipping: {clip_count} values clipped")
    return df

def fit_and_save_scalers(df):
    """Fit and save LabelEncoder for State, StandardScaler for Total."""
    le = LabelEncoder()
    le.fit(df['State'].unique())
    df['state_encoded'] = le.transform(df['State'])
    
    scaler = StandardScaler()
    scaler.fit(df[['Total']])
    
    joblib.dump(le, os.path.join(PREP_DIR, "label_encoder.pkl"))
    joblib.dump(scaler, os.path.join(PREP_DIR, "scaler.pkl"))
    
    # Save imputer config
    config = {
        "short_gap_limit": 3,
        "short_gap_method": "linear_interpolation",
        "seasonal_gap_method": "same_week_prior_year_mean",
        "outlier_method": "iqr_2x_clip_0.05_0.95",
        "weekly_anchor": "W-SUN",
        "states": sorted(df['State'].unique().tolist()),
        "n_states": int(df['State'].nunique()),
    }
    with open(os.path.join(PREP_DIR, "imputer_config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Saved: label_encoder.pkl, scaler.pkl, imputer_config.json")
    return df

def main():
    print("=" * 60)
    print("STEP 1: DATA PREPROCESSING")
    print("=" * 60)
    
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PREP_DIR, exist_ok=True)
    
    df = load_raw_data()
    df = normalize_to_weekly_sunday(df)
    df = fill_missing_weeks(df)
    df = impute_missing_values(df)
    df = handle_outliers(df)
    df = fit_and_save_scalers(df)
    
    # Save processed data
    df.to_parquet(os.path.join(DATA_DIR, "processed_data.parquet"), index=False)
    print(f"\nFinal dataset: {len(df)} rows, {df.columns.tolist()}")
    print(f"Saved to: {os.path.join(DATA_DIR, 'processed_data.parquet')}")
    print("PREPROCESSING COMPLETE")

if __name__ == "__main__":
    main()
