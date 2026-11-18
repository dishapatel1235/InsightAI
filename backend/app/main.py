from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api.routes.dataset_routes import router as dataset_router
from app.api.routes.ml_routes import router as ml_router


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(dataset_router)
app.include_router(ml_router)


@app.get("/")
def home():
    return {
        "message": f"{settings.PROJECT_NAME} Running",
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "features": [
            "Descriptive Statistics",
            "Predictive ML (RF, XGBoost, NN)",
            "Pattern Discovery (K-Means, PCA)",
            "Time-Series Forecasting (LSTM)",
        ],
    }
