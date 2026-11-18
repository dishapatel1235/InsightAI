import uuid
from typing import Any, Dict, Optional

class ModelRegistry:
    """In-memory cache for trained models to enable live What-If predictions."""
    def __init__(self):
        self._models: Dict[str, Dict[str, Any]] = {}

    def register_model(
        self,
        model: Any,
        task_type: str,
        target_column: str,
        feature_columns: list,
        preprocessor: Any,
        model_name: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        model_id = str(uuid.uuid4())
        self._models[model_id] = {
            "model": model,
            "task_type": task_type,
            "target_column": target_column,
            "feature_columns": feature_columns,
            "preprocessor": preprocessor,
            "model_name": model_name,
            "meta": meta or {},
        }
        return model_id

    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        return self._models.get(model_id)

    def clear(self):
        self._models.clear()

registry = ModelRegistry()
