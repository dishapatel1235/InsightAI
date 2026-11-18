# InsightAI — Full-Stack Analytics, Machine Learning & Deep Learning Platform

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI_0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![React](https://img.shields.io/badge/Frontend-React_19_TypeScript-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Bundler-Vite_8-646CFF?style=for-the-badge&logo=vite&logoColor=white)](https://vitejs.dev/)
[![PyTorch](https://img.shields.io/badge/Deep_Learning-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/Boosting-XGBoost-EB5424?style=for-the-badge)](https://xgboost.readthedocs.io/)

**InsightAI** is an enterprise-grade, full-stack intelligence platform that unifies exploratory data analysis, statistical moment profiling, supervised machine learning, unsupervised pattern discovery, and deep learning time-series forecasting into an interactive, real-time web application.

---

## System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│               Frontend: React 19 + TypeScript + Vite                   │
│                                                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  │
│  │ Statistics   │  │ Predictive   │  │ Pattern      │  │ TimeSeries │  │
│  │ & Profiling  │  │ ML & What-If │  │ Discovery    │  │ Forecast   │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └─────┬──────┘  │
│         │                 │                 │                │         │
│  ┌──────┴─────────────────┴─────────────────┴────────────────┴──────┐  │
│  │    Multi-Tab Export Hub (CSV Matrices, JSON, Markdown, PDF)      │  │
│  └─────────────────────────────────┬────────────────────────────────┘  │
└────────────────────────────────────┼───────────────────────────────────┘
                                     │ HTTP / REST APIs
┌────────────────────────────────────┼───────────────────────────────────┐
│               Backend: FastAPI (Python 3.11)                           │
│                                    ▼                                   │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Thread-Safe Ingestion & Multi-Encoding Loader (utf-8, latin1)    │  │
│  └──────┬─────────────────┬─────────────────┬────────────────┬──────┘  │
│         │                 │                 │                │         │
│  ┌──────▼───────┐  ┌──────▼───────┐  ┌──────▼───────┐  ┌─────▼──────┐  │
│  │ Descriptive  │  │ Predictive   │  │ Pattern      │  │ TimeSeries │  │
│  │ Statistics   │  │ Pipeline     │  │ Pipeline     │  │ Pipeline   │  │
│  │ (Skew/Kurt)  │  │ (RF/XGB/MLP) │  │ (KMeans/PCA) │  │ (LSTM/Aut) │  │
│  └──────────────┘  └──────┬───────┘  └──────────────┘  └────────────┘  │
│                           │                                            │
│                    ┌──────▼────────────────────────┐                   │
│                    │ In-Memory ModelRegistry (<5ms)│                   │
│                    └───────────────────────────────┘                   │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Zero-Dependency Pure NumPy/Pandas Analytical Fallback Engine     │  │
│  │ (Closed-Form Ridge, Vectorized Softmax, SVD PCA, Lloyd's KMeans) │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Key Features & Capabilities

### 1. Supervised Machine Learning & "What-If" Simulator
- **Dual Problem Support**: Supports both **Classification** (discrete labels) and **Regression** (continuous target variables).
- **Ensemble & Deep Models**:
  - **Random Forest**: Bagged decision tree ensembles (`RandomForestClassifier`, `RandomForestRegressor`).
  - **XGBoost / Gradient Boosting**: Sequential gradient-boosted trees for high-accuracy predictions.
  - **Deep Neural Network (MLP)**: Multi-Layer Perceptrons with dense hidden layers and early stopping.
- **Evaluation Dashboards**:
  - Classification: Test Accuracy, Training Accuracy, Weighted Precision, Weighted Recall, Weighted F1-Score, and an interactive **Confusion Matrix Heatmap**.
  - Regression: $R^2$ Goodness of Fit, Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), Mean Squared Error (MSE), MAPE %, and an **Actual vs. Predicted Scatter Plot with 45° reference line**.
- **Ranked Feature Importance**: Normalized contribution scores across all predictor features.
- **Interactive "What-If" Inference Simulator**: Real-time simulation powered by an in-memory `ModelRegistry`. Tweak individual feature sliders to trigger instantaneous ($< 5\text{ms}$) predictions without retraining.

### 2. Unsupervised Pattern Discovery & Clustering
- **K-Means Clustering**: Natural data segmentation ($K=2$ to $10$) with Silhouette Score evaluation and Inertia metrics.
- **2D PCA Projection Canvas**: Singular Value Decomposition (SVD) dimensionality reduction projecting multidimensional data points and cluster centroids onto a 2D interactive canvas.
- **Correlation Matrix Heatmap**: Full $N \times N$ Pearson correlation matrix ($r \in [-1.0, +1.0]$) highlighting top positively and negatively correlated feature pairs.
- **Anomaly Detection**: Unsupervised outlier isolation via **Isolation Forests** with fallback to Mahalanobis / standardized Z-score percentile thresholding.

### 3. Sequential Time-Series Forecasting
- **PyTorch LSTM Architecture**: Recurrent neural networks with input, forget, and output gates capturing temporal dependencies across sequential observations.
- **Sliding-Window Supervised Transformation**: Automated lag lookback windowing ($t-L$ to $t$).
- **Multi-Step Autoregressive Rollout**: Recursive prediction forecasting arbitrary future horizons.
- **Expanding 95% Confidence Intervals**: Uncertainty bands computed via residual variance:
  $$\text{Bounds}_{95\%} = \hat{y}_h \pm 1.96 \cdot \sigma_{\text{residual}} \cdot \sqrt{h}$$
- **Evaluation**: Validation MAE, RMSE, and MAPE % error metrics.

### 4. Descriptive Statistics & Distribution Shape Profiling
- **11 Core Statistical Metrics**: Mean, Median, Std Dev, Variance, Min, Max, Sum, Valid Count, Missing Values, Skewness, Kurtosis.
- **The 4 Statistical Moments**:
  - 1st Moment (Mean): Central tendency.
  - 2nd Moment (Variance): Dispersion.
  - 3rd Moment (Skewness): Asymmetry classification (Right-Skewed, Left-Skewed, Symmetric).
  - 4th Moment (Kurtosis): Tail weight and outlier propensity (Leptokurtic, Mesokurtic, Platykurtic).
- **Data Quality & Hygiene Audit**: Automated audit screening for missing entry rates, completeness scores, and extreme outlier vulnerabilities.

### 5. Multi-Tab Export & Reporting Engine
- **Per-Tab Downloads**: Direct CSV/JSON/Markdown exports directly within every module:
  - Statistics CSV Matrix (all 11 metrics per column)
  - Model Predictions & Residuals CSV
  - Ranked Feature Importance CSV
  - Cluster Profiles & Centroids CSV
  - Pairwise Correlation Matrix CSV
  - Detected Anomalies Table CSV
  - Forecast Timeline & 95% Confidence Bounds CSV
  - Data Quality Audit Matrix CSV
- **Unified Master Report**: Single-click generation of:
  - **Master Intelligence JSON**: Full multi-module machine-readable document.
  - **Executive Master Brief (Markdown)**: Multi-section report ready for documentation, GitHub, or Notion.
  - **Batch CSV Download**: Sequentially packages all available tabular outputs.
  - **Print / PDF Master View**: Print-ready CSS styling for executive presentations.

### 6. Zero-Dependency Pure NumPy/Pandas Fallback Engine
- Features an integrated **zero-dependency analytical engine** written in pure NumPy:
  - `PureNumpyClassifier`: Vectorized multi-class Softmax classifier with L2 regularization and cross-entropy gradient descent.
  - `PureNumpyRegressor`: Closed-form analytical Ridge regression solver ($\beta = (X^T X + \alpha I)^{-1} X^T y$).
  - `pure_numpy_kmeans`: Lloyd’s algorithm with K-means++ initialization.
  - `pure_numpy_pca`: Singular Value Decomposition (SVD) 2D projection.
  - Native metric solvers for Accuracy, F1, Confusion Matrix, $R^2$, RMSE, MAE, and MAPE.
- Guarantees 100% operational resilience in minimal environments without crashing.

---

## Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Frontend** | React 19, TypeScript, Vite 8, HTML5 Canvas, Custom CSS Design System |
| **Backend** | FastAPI, Uvicorn, Pydantic v2, Python 3.11, python-multipart, aiofiles |
| **Data Processing** | Pandas, NumPy, Chunked Stream Blob Slicing |
| **Machine Learning** | Scikit-Learn, XGBoost, PyTorch, Custom Pure NumPy Analytical Solvers |

---

## 1-Click Instant Sample Datasets

The dashboard includes 4 built-in datasets accessible via the **"Load Sample Data"** dropdown in the header:
1. `telecom_customer_churn.csv`: Classification dataset for Customer Churn modeling.
2. `housing_prices_regression.csv`: Real estate property valuation dataset.
3. `daily_retail_forecast_timeseries.csv`: 40-day revenue and demand sequence for LSTM forecasting.
4. `mall_customers_segmentation.csv`: Customer demographics and spending scores for K-Means & PCA.

---

## Getting Started

### Prerequisites
- **Python**: 3.10 or higher
- **Node.js**: v18.0.0 or higher
- **npm**: v9.0.0 or higher

---

### Step 1: Clone the Repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

---

### Step 2: Backend Setup (FastAPI)

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create and activate a Python virtual environment:
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # macOS / Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables (optional, defaults provided):
   ```bash
   # Copy the template to .env
   cp .env.example .env
   ```

5. Start the FastAPI development server:
   ```bash
   uvicorn app.main:app --reload
   ```
   - API Server: `http://127.0.0.1:8000`
   - Interactive Swagger API Docs: `http://127.0.0.1:8000/docs`

---

### Step 3: Frontend Setup (React + Vite)

1. Open a new terminal and navigate to the frontend directory:
   ```bash
   cd frontend/analyst-project
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Configure environment variables (optional):
   ```bash
   cp .env.example .env
   ```

4. Launch the Vite development server:
   ```bash
   npm run dev
   ```
   - Frontend Application: `http://localhost:5173`

---

## Project Structure

```
InsightAI/
├── backend/
│   ├── app/
│   │   ├── analytics/
│   │   │   └── statistics/
│   │   │       └── descriptive_statistics.py # Vectorized stats, sampling, skewness, kurtosis
│   │   ├── api/
│   │   │   └── routes/
│   │   │       ├── dataset_routes.py         # Ingestion, file listing, preview APIs
│   │   │       └── ml_routes.py              # Supervised, unsupervised, time-series endpoints
│   │   ├── ml/
│   │   │   ├── model_registry.py             # In-memory model caching for live What-If inference
│   │   │   ├── pattern_discovery.py          # K-Means, SVD PCA, correlation, anomaly detection
│   │   │   ├── predictive.py                 # RF, XGBoost, Neural Nets & Pure NumPy engine
│   │   │   └── time_series.py                # PyTorch LSTM, autoregressive rollout & CI bounds
│   │   ├── services/
│   │   │   └── dataset_service.py            # File persistence & sanitization
│   │   ├── utils/
│   │   │   └── dataframe_loader.py           # Multi-encoding (utf-8, latin1) DataFrame reader
│   │   └── main.py                           # FastAPI initialization & CORS configuration
│   ├── uploads/                              # Temporary dataset staging directory
│   └── requirements.txt                      # Python dependencies
├── frontend/
│   └── analyst-project/
│       ├── src/
│       │   ├── components/
│       │   │   ├── DataPreview.tsx           # Tabular dataset previewer
│       │   │   ├── DataQualityAudit.tsx      # Missing value & skewness screening
│       │   │   ├── ExportModal.tsx           # Central multi-tab export hub
│       │   │   ├── FileUpload.tsx            # Drag & drop upload & sample loader
│       │   │   ├── Header.tsx                # Brand header, server status & dataset reset
│       │   │   ├── PatternDiscovery.tsx      # Clustering, PCA scatter canvas & anomalies
│       │   │   ├── PredictiveAnalytics.tsx   # ML training, confusion matrix & What-If simulator
│       │   │   ├── StatisticsTable.tsx       # 11-metric statistical table with sort & filter
│       │   │   ├── SummaryCards.tsx          # Key performance indicator cards
│       │   │   ├── TimeSeriesForecasting.tsx # LSTM timeline visualizer with 95% confidence bands
│       │   │   └── VisualAnalytics.tsx       # Range spreads & distribution shape analysis
│       │   ├── services/
│       │   │   └── api.ts                    # Axios/fetch API clients with typed contracts
│       │   ├── types/
│       │   │   ├── analytics.ts              # TypeScript schemas for statistics & datasets
│       │   │   └── ml.ts                     # TypeScript schemas for ML, patterns & time-series
│       │   ├── utils/
│       │   │   ├── csvParser.ts              # 64KB chunked stream preview parser & sample datasets
│       │   │   └── exportUtils.ts            # CSV/JSON/MD generators & download managers
│       │   ├── App.tsx                       # Root view, state management & tab controller
│       │   └── App.css                       # Custom responsive dark-mode styling system
│       ├── package.json                      # Frontend dependencies & scripts
│       └── vite.config.ts                    # Vite build configuration
├── .gitignore                                # Excludes venv, node_modules, cache, and uploads
└── README.md                                 # Project documentation
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
