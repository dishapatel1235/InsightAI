import os
import shutil
import uuid
import numpy as np
import pandas as pd
from fastapi import UploadFile

from app.config import settings
from app.services.analytics_service import AnalyticsService
from app.utils.dataframe_loader import DataFrameLoader

UPLOAD_DIR = settings.UPLOAD_DIR


class DatasetService:
    def __init__(self):
        self.analytics_service = AnalyticsService()

    def list_uploaded_files(self):
        """Lists all dataset files currently stored on the server in uploads/."""
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        files_info = []
        for name in sorted(os.listdir(UPLOAD_DIR)):
            if name.lower().endswith((".csv", ".xlsx", ".xls")):
                path = os.path.join(UPLOAD_DIR, name)
                try:
                    size_bytes = os.path.getsize(path)
                    files_info.append({
                        "file_name": name,
                        "size_bytes": size_bytes,
                        "size_display": f"{size_bytes / (1024 * 1024):.1f} MB" if size_bytes >= 1024 * 1024 else f"{size_bytes / 1024:.1f} KB",
                    })
                except Exception:
                    files_info.append({"file_name": name, "size_bytes": 0, "size_display": "—"})
        return files_info

    def save_file(self, file: UploadFile) -> str:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        raw_name = os.path.basename(file.filename or "uploaded_dataset.csv")
        safe_name = "".join(c for c in raw_name if c.isalnum() or c in "._- ")
        if not safe_name:
            safe_name = f"dataset_{uuid.uuid4().hex[:8]}.csv"

        file_path = os.path.join(UPLOAD_DIR, safe_name)

        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except (PermissionError, OSError):
            unique_name = f"{uuid.uuid4().hex[:6]}_{safe_name}"
            file_path = os.path.join(UPLOAD_DIR, unique_name)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        return file_path

    def _build_dataset_payload(self, file_path: str, original_name: str):
        df = DataFrameLoader.load_dataframe(file_path)
        statistics = self.analytics_service.generate_statistics(df)

        preview_rows = []
        try:
            head_df = df.head(10)
            for _, row in head_df.iterrows():
                row_dict = {}
                for col in df.columns:
                    val = row[col]
                    if pd.isna(val) or val is None:
                        row_dict[str(col)] = None
                    elif isinstance(val, (int, np.integer)):
                        row_dict[str(col)] = int(val)
                    elif isinstance(val, (float, np.floating)):
                        if np.isnan(val) or np.isinf(val):
                            row_dict[str(col)] = None
                        else:
                            row_dict[str(col)] = round(float(val), 4)
                    elif isinstance(val, (pd.Timestamp, np.datetime64)):
                        row_dict[str(col)] = str(val)
                    else:
                        row_dict[str(col)] = str(val)
                preview_rows.append(row_dict)
        except Exception:
            preview_rows = []

        return {
            "file_name": original_name,
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "statistics": statistics,
            "columns_list": [str(c) for c in df.columns],
            "preview": preview_rows,
        }

    def process_dataset(self, file: UploadFile):
        file_path = self.save_file(file)
        return self._build_dataset_payload(file_path, os.path.basename(file_path))

    def process_existing_file(self, file_name: str):
        file_path = os.path.join(UPLOAD_DIR, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{file_name}' not found on server.")
        return self._build_dataset_payload(file_path, file_name)
