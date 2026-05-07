# SalesCast AI - Enterprise Sales Forecasting Platform

An end-to-end, highly optimized Artificial Intelligence platform for distributed sales forecasting. SalesCast AI deploys 6 cutting-edge machine learning models concurrently across 43 states to generate highly accurate, 8-week horizon forecasts, orchestrated by a high-performance backend and visualized through a premium Next.js dashboard.

---

## System Design & Architecture

The system is built on a decoupled, microservices-inspired architecture designed for low latency and high throughput.

### 1. Frontend (Next.js 15 + React)
- **Framework:** Next.js with App Router.
- **UI/UX:** Premium Glassmorphism design system using raw CSS, completely responsive.
- **Data Visualization:** `recharts` for interactive Area, Line, and Bar charts.
- **Real-Time Integration:** Consumes Server-Sent Events (SSE) from the backend for live AI reasoning and orchestration progress without polling overhead.

### 2. Backend (FastAPI + Python)
- **Framework:** FastAPI running on ASGI (Uvicorn).
- **Inference Orchestrator:** Asynchronous worker threads concurrently execute 6 different ML models.
- **Artifact Registry:** A singleton pattern registry that pre-loads all scalers, encoders, and model weights into memory at startup to eliminate disk I/O latency during inference.
- **Streaming Response:** Yields JSON packets directly to the frontend as each model finishes its forecast via `StreamingResponse`.

### 3. Databases & State
- **MySQL:** Stores permanent system metadata and logging.
- **Redis:** In-memory data structure store used for ultra-fast inter-process communication, locking, and temporary caching during complex multi-state orchestrations.

---

## AI Engine & Orchestration

The platform utilizes a **Champion-Challenger** machine learning architecture. Rather than relying on a single algorithm, the system evaluates 6 distinct models to find the optimal fit for the data.

### The Models
1. **SARIMA** (Seasonal ARIMA): Captures linear seasonality and trends.
2. **Prophet** (Facebook): Handles holiday effects and non-linear growth curves effectively.
3. **ETS** (Exponential Smoothing): Excels at capturing dynamic seasonal patterns.
4. **XGBoost** (Gradient Boosting): Powerful tree-based model for complex, non-linear feature interactions.
5. **LSTM** (Long Short-Term Memory): Deep learning recurrent neural network for long-term sequential dependencies.
6. **TFT** (Temporal Fusion Transformer): State-of-the-art attention-based deep learning for multi-horizon forecasting.

### AI Orchestration
When a forecast is requested, the backend orchestrator assigns the task to asynchronous workers. The system calculates the `wMAPE` (Weighted Mean Absolute Percentage Error) and `RMSE` for each model on validation data. The orchestrator automatically promotes the model with the lowest wMAPE to **Champion** status, using its predictions as the primary source of truth.

---

## Optimizations & Enhancements

### 1. Speed & Latency Reduction
- **In-Memory Artifacts:** The `ArtifactRegistry` caches all 6 ML models and massive `.parquet` baseline datasets into RAM at boot time. Inference takes milliseconds because there is zero disk I/O.
- **SSE Streams:** Avoids traditional HTTP request-response blocking. The UI receives incremental progress updates and logs *while* the models are calculating.
- **NGINX No-Buffering:** NGINX is configured with `proxy_buffering off;` and `chunked_transfer_encoding on;` ensuring AI inference streams hit the client instantly without network buffering.
- **Multi-Core Uvicorn:** The FastAPI backend runs with `--workers 4`, bypassing Python's GIL to handle high concurrent throughput.

### 2. Accuracy
- **Dynamic Ensembling:** By forcing 6 diverse algorithms (statistical, tree-based, and deep learning) to compete, the system guarantees the highest possible accuracy for the specific data shape.
- **Robust Preprocessing:** Uses `MinMaxScaler` and `LabelEncoder` pipelines preserved faithfully from training to inference.

### 3. Security
- **Strict CORS Policy:** FastAPI only accepts requests from trusted origins.
- **Environment Isolation:** Database credentials and API keys are strictly kept out of source code using `.env` files.
- **Hidden Artifacts:** Model weights and processed data are explicitly `.gitignore`d to prevent sensitive corporate data leaks.

---

## How to Run Locally (Start to End)

Because the heavy ML artifacts (`.pkl`, `.h5`, `.parquet`) are `.gitignore`d, you **must train the models first** to generate the `artifacts/` folder before running the backend.

### Prerequisites
- Python 3.10+
- Node.js 22.x+
- Redis Server (Running on `localhost:6379`)
- MySQL Server (Running on `localhost:3306`)

### Step 1: Train the Models (Generate Artifacts)
Run the training pipeline sequentially. This will process your raw data, engineer features, train all 6 models, evaluate them, and output everything into the root `artifacts/` directory.

```bash
cd training
# 1. Clean and preprocess raw data
python 01_preprocess.py
# 2. Engineer time-series features
python 02_features.py
# 3. Train SARIMA, Prophet, ETS, and XGBoost
python 03_train_models.py
# 4. Train Deep Learning models (LSTM & TFT)
python 04_fix_lstm_tft.py
cd ..
```
*Verify that an `artifacts/` folder now exists in your root directory containing `models`, `metrics`, `preprocessor`, and `data`.*

### Step 2: Start the Backend (FastAPI)
```bash
cd backend
python -m venv venv
# Windows: venv\Scripts\activate | Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Step 3: Start the Frontend (Next.js)
Open a new terminal window.
```bash
cd frontend
npm install
npm run dev
```
Navigate to `http://localhost:3000` in your browser. The orchestrator will automatically read your generated `artifacts/` folder and stream the live forecasts!

---

## Future Scope: Production Deployment

In the future, deploying this system to an AWS EC2 instance will be fully automated for massive scale. We have already prepared the infrastructure scripts inside the `deployment/` folder:

### 1. S3 Artifact Synchronization
Instead of committing gigabytes of model weights to GitHub, we will utilize an S3 bucket (e.g., `s3://sales-615645510621/artifacts/`). 
- **Locally:** We will run `upload_to_s3.bat` to push the locally trained `artifacts/` folder to the cloud.
- **On EC2:** The `deployment/s3_sync.sh` script will pull these models directly onto the EC2 NVMe drive for ultra-fast local read speeds.

### 2. One-Click EC2 Deployment
We have engineered a master `deploy_ec2.sh` script. When executed on a fresh Ubuntu EC2 instance, it will:
1. Install Node, Python, Redis, MySQL, and NGINX.
2. Clone this repository.
3. Securely pull the ML models from AWS S3.
4. Build the Next.js production bundle.
5. Create `systemd` daemon services to keep the backend and frontend running forever in the background, recovering instantly from crashes.

### 3. NGINX Reverse Proxy
NGINX will be configured to route traffic seamlessly on Port 80, bypassing standard caching to allow the Server-Sent Events (SSE) AI streams to flow directly to the end user with zero latency.
