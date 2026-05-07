"""MySQL service for persistent storage of forecast results."""
import json
from datetime import datetime
from app.config import settings

# Will use direct pymysql for simplicity and reliability
_connection = None
_use_fallback = False
_memory_store = {"jobs": [], "results": []}

def connect():
    global _connection, _use_fallback
    try:
        import pymysql
        _connection = pymysql.connect(
            host=settings.MYSQL_HOST, port=settings.MYSQL_PORT,
            user=settings.MYSQL_USER, password=settings.MYSQL_PASSWORD,
            database=settings.MYSQL_DATABASE, autocommit=True,
            cursorclass=pymysql.cursors.DictCursor
        )
        _init_tables()
        print("[MySQL] Connected")
        return True
    except Exception as e:
        print(f"[MySQL] Connection failed: {e}. Using in-memory fallback.")
        _use_fallback = True
        return False

def _init_tables():
    global _connection
    with _connection.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS forecast_jobs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                job_id VARCHAR(64) UNIQUE NOT NULL,
                state_query VARCHAR(128),
                status VARCHAR(32),
                champion_model VARCHAR(64),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                completed_at DATETIME NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS forecast_results (
                id INT AUTO_INCREMENT PRIMARY KEY,
                job_id VARCHAR(64) NOT NULL,
                state VARCHAR(64),
                forecast_date DATE,
                predicted_value DOUBLE,
                confidence_lower DOUBLE NULL,
                confidence_upper DOUBLE NULL,
                model_name VARCHAR(64),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_job (job_id),
                INDEX idx_state (state)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS model_evaluations (
                id INT AUTO_INCREMENT PRIMARY KEY,
                job_id VARCHAR(64),
                model_name VARCHAR(64),
                wmape DOUBLE, rmse DOUBLE, mae DOUBLE,
                rank_position INT,
                is_champion BOOLEAN DEFAULT FALSE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

def save_job(job_id, state_query, status, champion_model=None):
    if _use_fallback:
        _memory_store["jobs"].append({"job_id": job_id, "state_query": state_query, "status": status})
        return
    try:
        with _connection.cursor() as cur:
            cur.execute(
                "INSERT INTO forecast_jobs (job_id, state_query, status, champion_model, completed_at) "
                "VALUES (%s, %s, %s, %s, %s) ON DUPLICATE KEY UPDATE status=%s, champion_model=%s, completed_at=%s",
                (job_id, state_query, status, champion_model, 
                 datetime.now() if status == "completed" else None,
                 status, champion_model, datetime.now() if status == "completed" else None)
            )
    except Exception as e:
        print(f"[MySQL] save_job error: {e}")

def save_forecast_results(job_id, forecasts, model_name):
    if _use_fallback:
        _memory_store["results"].extend(forecasts)
        return
    try:
        with _connection.cursor() as cur:
            for f in forecasts:
                cur.execute(
                    "INSERT INTO forecast_results (job_id, state, forecast_date, predicted_value, "
                    "confidence_lower, confidence_upper, model_name) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                    (job_id, f["state"], f["forecast_date"], f["predicted_value"],
                     f.get("confidence_lower"), f.get("confidence_upper"), model_name)
                )
    except Exception as e:
        print(f"[MySQL] save_results error: {e}")

def save_model_evaluation(job_id, model_name, metrics, rank, is_champion):
    if _use_fallback:
        return
    try:
        with _connection.cursor() as cur:
            cur.execute(
                "INSERT INTO model_evaluations (job_id, model_name, wmape, rmse, mae, rank_position, is_champion) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s)",
                (job_id, model_name, metrics.get("wMAPE",0), metrics.get("RMSE",0),
                 metrics.get("MAE",0), rank, is_champion)
            )
    except Exception as e:
        print(f"[MySQL] save_evaluation error: {e}")
