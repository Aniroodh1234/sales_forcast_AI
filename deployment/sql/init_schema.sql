CREATE DATABASE IF NOT EXISTS salescast_db;
USE salescast_db;

CREATE TABLE IF NOT EXISTS forecast_jobs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    job_id VARCHAR(64) UNIQUE NOT NULL,
    state_query VARCHAR(128),
    status VARCHAR(32),
    champion_model VARCHAR(64),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    completed_at DATETIME NULL
);

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
);

CREATE TABLE IF NOT EXISTS model_evaluations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    job_id VARCHAR(64),
    model_name VARCHAR(64),
    wmape DOUBLE,
    rmse DOUBLE,
    mae DOUBLE,
    rank_position INT,
    is_champion BOOLEAN DEFAULT FALSE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
