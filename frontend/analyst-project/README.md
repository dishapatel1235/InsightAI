# InsightAI — Full-Stack Analytics, Machine Learning & Deep Learning Platform

An enterprise-grade, responsive React + TypeScript analytics interface integrated with a modular FastAPI backend for **Descriptive Statistics**, **Predictive ML (Random Forest, XGBoost, Neural Networks)**, **Pattern Discovery (K-Means Clustering, 2D PCA, Correlation, Anomaly Detection)**, and **Deep Learning Time-Series Forecasting (LSTM)**.

---

## Capabilities & Architecture

### 1. Supervised Predictive Analytics
- **Task Types**: Classification (discrete category prediction) and Regression (continuous numeric prediction).
- **Supported Algorithms**:
  - **Random Forest**: Bagged decision tree ensemble (`RandomForestClassifier`, `RandomForestRegressor`).
  - **XGBoost / Gradient Boosting**: Sequential boosted decision trees with high gradient accuracy.
  - **Deep Neural Network (MLP)**: Multi-Layer Perceptron with dense layers, early stopping, and non-linear activations.
- **Automated Preprocessing**: Imputation of missing numeric & categorical values, label encoding, standard feature scaling.
- **Evaluation Dashboard**:
  - Classification: Test Accuracy, Training Accuracy, F1-Score (weighted), Precision, Recall, and an **Interactive Confusion Matrix Heatmap**.
  - Regression: $R^2$ Goodness of Fit, RMSE, MAE, MSE, MAPE %, and an **Actual vs Predicted Scatter Plot with 45° Ideal Line**.
  - **Feature Importance Ranking**: Relative influential weights of each variable.
- **Interactive "What-If" Live Simulator**:
  - Adjust feature inputs via sliders/inputs to trigger instantaneous real-time inference against the active model.

### 2. Unsupervised Pattern Discovery & Clustering
- **K-Means Clustering**: Discover natural data groupings ($k=2$ to $8$), compute Silhouette Scores, Inertia, and Segment Profiles.
- **2D PCA Projection Space**: Visual cluster scatter plot mapping multi-dimensional data into 2 Principal Components with centroid markers.
- **Correlation Matrix Heatmap**: Color-coded pairwise Pearson correlation coefficients ($r \in [-1.0, +1.0]$) with strongest feature pairs highlighted.
- **Isolation Forest Anomaly Detection**: Unsupervised anomaly scoring to isolate outlier records.

### 3. Deep Learning Time-Series Forecasting
- **LSTM (Long Short-Term Memory)**: Recurrent neural network capturing temporal dependencies across sequential observations.
- **Deep Neural Net (Autoregressive MLP)** & **Sequential Random Forest**: Multi-lag sliding window models.
- **Interactive Horizon Visualizer**:
  - Historical observed actuals (blue line).
  - In-sample fitted model predictions (green dashed line).
  - Out-of-sample future forecast horizon (purple line).
  - **95% Confidence Interval Band** (translucent purple envelope).
- **Accuracy Metrics**: MAE, RMSE, and MAPE % on validation test segments.

### 4. Descriptive Statistics & Data Profiling
- 11 statistical measures per numeric column: `Mean`, `Median`, `Std Dev`, `Variance`, `Min`, `Max`, `Sum`, `Count`, `Missing Values`, `Skewness`, `Kurtosis`.
- Range spreads, volatility bar charts, distribution symmetry meters, and data quality audits.

---

## 1-Click Instant Test Datasets
The application includes 4 pre-packaged datasets accessible via the **"Load Sample Data"** menu:
1. `telecom_customer_churn.csv`: Classification dataset for Customer Churn (Random Forest & XGBoost).
2. `housing_prices_regression.csv`: Real estate valuation dataset (Neural Networks & Gradient Boosting).
3. `daily_retail_forecast_timeseries.csv`: 40-day demand and revenue timeline (LSTM Time-Series Forecasting).
4. `mall_customers_segmentation.csv`: Customer demographics and spending (K-Means Clustering & PCA).

---

## Getting Started

### 1. Install Backend Dependencies & Start FastAPI
In your backend terminal:
```powershell
cd "backend"
pip install -r requirements.txt
uvicorn app.main:app --reload
```
*FastAPI starts at **`http://127.0.0.1:8000`** (Interactive Swagger API docs at `http://127.0.0.1:8000/docs`).*

### 2. Start Frontend Dev Server
In your frontend terminal:
```powershell
cd "frontend\analyst-project"
npm run dev
```
*Vite starts the dashboard at **`http://localhost:5173`**.*

---

## Backend API Endpoints
- `POST /datasets/upload`: Ingests CSV or Excel files, saves to `uploads/`, computes descriptive statistics and preview rows.
- `POST /ml/train-predictive`: Trains Random Forest, XGBoost, or Neural Network; returns metrics, confusion matrix, feature importance, and `model_id`.
- `POST /ml/pattern-discovery`: Runs K-Means clustering, 2D PCA projection, correlation matrices, and anomaly detection.
- `POST /ml/forecast-timeseries`: Fits sequential LSTM or neural autoregressive models and projects future horizon with confidence intervals.
- `POST /ml/predict`: Runs real-time inference on an active `model_id` using user-supplied feature values.
- `GET /ml/status`: Returns status of installed ML libraries (`scikit-learn`, `xgboost`, `torch`).


## Commands
activate virtual environment : .\venv\Scripts\Activate
install ml dependencies in venv : .\venv\Scripts\pip install scikit-learn xgboost torch
run backend : uvicorn app.main:app --reload 
run frontend : npm run dev
