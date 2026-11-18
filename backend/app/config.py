import os
from pathlib import Path
from typing import List

# Locate .env in the backend folder
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

if ENV_PATH.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(ENV_PATH)
    except ImportError:
        # Fallback pure-Python parser if python-dotenv is not installed
        with open(ENV_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    val = val.strip().strip("'\"")
                    os.environ.setdefault(key.strip(), val)


class Settings:
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "InsightAI — Machine Learning & Analytics Engine")
    VERSION: str = os.getenv("VERSION", "2.0.0")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", 8000))
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "uploads")

    raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
    if raw_origins == "*":
        ALLOWED_ORIGINS: List[str] = ["*"]
    else:
        ALLOWED_ORIGINS: List[str] = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]


settings = Settings()
