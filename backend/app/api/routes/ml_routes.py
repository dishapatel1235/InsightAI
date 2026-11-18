import os
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import settings
from app.utils.dataframe_loader import DataFrameLoader
from app.ml.predictive import PredictivePipeline, SKLEARN_AVAILABLE, XGBOOST_AVAILABLE, TORCH_AVAILABLE
from app.ml.pattern_discovery import PatternDiscoveryPipeline
from app.ml.time_series import TimeSeriesPipeline

router = APIRouter(prefix="/ml", tags=["Machine Learning & AI Engine"])
UPLOAD_DIR = settings.UPLOAD_DIR


def _get_dataframe(file_name: str):
    file_path = os.path.join(UPLOAD_DIR, file_name)
    if not os.path.exists(file_path):
        # Fallback to any recent file in uploads if exact name not found
        if os.path.exists(UPLOAD_DIR):
            files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith((".csv", ".xlsx"))]
            if files:
                file_path = os.path.join(UPLOAD_DIR, files[-1])
            else:
                raise HTTPException(status_code=404, detail=f"Dataset '{file_name}' not found. Please upload a dataset first.")
        else:
            raise HTTPException(status_code=404, detail=f"No datasets found in uploads. Please upload a file first.")

    try:
        return DataFrameLoader.load_dataframe(file_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read dataset: {str(e)}")


class TrainPredictiveRequest(BaseModel):
    file_name: str
    target_column: str
    task_type: str = "classification"  # "classification" or "regression"
    model_name: str = "random_forest"  # "random_forest", "xgboost", "neural_network"
    feature_columns: Optional[List[str]] = None
    hyperparameters: Optional[Dict[str, Any]] = None


class PatternDiscoveryRequest(BaseModel):
    file_name: str
    method: str = "clustering"  # "clustering", "correlation", "anomaly", "all"
    feature_columns: Optional[List[str]] = None
    n_clusters: int = 3
    contamination: float = 0.05


class ForecastRequest(BaseModel):
    file_name: str
    target_column: str
    time_column: Optional[str] = None
    model_name: str = "lstm"  # "lstm", "neural_net", "random_forest"
    lookback: int = 5
    horizon: int = 10
    epochs: int = 50


class PredictRequest(BaseModel):
    model_id: str
    input_features: Dict[str, Any]


@router.get("/status")
def get_ml_status():
    """Returns availability of ML/DL packages."""
    return {
        "sklearn_available": SKLEARN_AVAILABLE,
        "xgboost_available": XGBOOST_AVAILABLE,
        "torch_available": TORCH_AVAILABLE,
        "supported_models": {
            "predictive": ["random_forest", "xgboost", "neural_network"],
            "pattern_discovery": ["clustering", "correlation", "anomaly_detection"],
            "time_series": ["lstm", "neural_net", "random_forest"],
        },
    }


@router.post("/train-predictive")
def train_predictive_model(req: TrainPredictiveRequest):
    """Trains a predictive machine learning or deep learning model."""
    df = _get_dataframe(req.file_name)
    try:
        result = PredictivePipeline.train_model(
            df=df,
            target_column=req.target_column,
            task_type=req.task_type,
            model_name=req.model_name,
            feature_columns=req.feature_columns,
            hyperparameters=req.hyperparameters,
        )
        return {
            "success": True,
            "data": result,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/pattern-discovery")
def analyze_patterns(req: PatternDiscoveryRequest):
    """Performs unsupervised clustering, correlation analysis, and anomaly detection."""
    df = _get_dataframe(req.file_name)
    try:
        result = PatternDiscoveryPipeline.analyze_patterns(
            df=df,
            method=req.method,
            feature_columns=req.feature_columns,
            n_clusters=req.n_clusters,
            contamination=req.contamination,
        )
        return {
            "success": True,
            "data": result,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/forecast-timeseries")
def forecast_timeseries(req: ForecastRequest):
    """Forecasts continuous time-series trajectories using LSTM or sequence models."""
    df = _get_dataframe(req.file_name)
    try:
        result = TimeSeriesPipeline.forecast(
            df=df,
            target_column=req.target_column,
            time_column=req.time_column,
            model_name=req.model_name,
            lookback=req.lookback,
            horizon=req.horizon,
            epochs=req.epochs,
        )
        return {
            "success": True,
            "data": result,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/predict")
def predict_single_instance(req: PredictRequest):
    """Simulates real-time inference using a trained model."""
    try:
        result = PredictivePipeline.predict_single(
            model_id=req.model_id,
            input_features=req.input_features,
        )
        return {
            "success": True,
            "data": result,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
