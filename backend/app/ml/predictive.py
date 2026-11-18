import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from app.ml.model_registry import registry

# Try importing sklearn and xgboost
SKLEARN_AVAILABLE = False
XGBOOST_AVAILABLE = False
TORCH_AVAILABLE = False

try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import (
        RandomForestClassifier,
        RandomForestRegressor,
        GradientBoostingClassifier,
        GradientBoostingRegressor,
    )
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        r2_score,
        mean_squared_error,
        mean_absolute_error,
    )
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ==========================================
# Pure NumPy Fallback Models & Metrics Engine
# ==========================================

def calc_accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def calc_confusion_matrix(y_true, y_pred, n_classes: int) -> List[List[int]]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    return cm.tolist()


def calc_precision_recall_f1(y_true, y_pred, n_classes: int):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if len(y_true) == 0:
        return 0.0, 0.0, 0.0

    precisions = []
    recalls = []
    f1s = []
    weights = []

    for c in range(n_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        support = np.sum(y_true == c)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        weights.append(support)

    total_weight = sum(weights) or 1
    weighted_prec = sum(p * w for p, w in zip(precisions, weights)) / total_weight
    weighted_rec = sum(r * w for r, w in zip(recalls, weights)) / total_weight
    weighted_f1 = sum(f * w for f, w in zip(f1s, weights)) / total_weight

    return float(weighted_prec), float(weighted_rec), float(weighted_f1)


def calc_r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))


def calc_mse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def calc_mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


class PureNumpyRegressor:
    """Zero-dependency Ridge + Linear Regressor with analytical solution."""
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.weights = None
        self.intercept = 0.0
        self.feature_importances_ = None
        self.mean_ = None
        self.std_ = None

    def fit(self, X, y):
        X_mat = np.asarray(X, dtype=float)
        y_vec = np.asarray(y, dtype=float)

        self.mean_ = np.nanmean(X_mat, axis=0)
        self.std_ = np.nanstd(X_mat, axis=0)
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)

        X_norm = (X_mat - self.mean_) / self.std_
        X_norm = np.nan_to_num(X_norm, nan=0.0)
        y_vec = np.nan_to_num(y_vec, nan=0.0)

        n_samples, n_features = X_norm.shape
        # Bias column
        X_bias = np.hstack([X_norm, np.ones((n_samples, 1))])

        reg = self.alpha * np.eye(n_features + 1)
        reg[-1, -1] = 0.0  # Do not regularize intercept

        try:
            beta = np.linalg.solve(X_bias.T @ X_bias + reg, X_bias.T @ y_vec)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(X_bias.T @ X_bias + reg) @ (X_bias.T @ y_vec)

        self.weights = beta[:-1]
        self.intercept = float(beta[-1])

        raw_imp = np.abs(self.weights)
        total_imp = np.sum(raw_imp)
        if total_imp > 0:
            self.feature_importances_ = raw_imp / total_imp
        else:
            self.feature_importances_ = np.ones(n_features) / n_features

        return self

    def predict(self, X):
        X_mat = np.asarray(X, dtype=float)
        if X_mat.ndim == 1:
            X_mat = X_mat.reshape(1, -1)
        X_norm = (X_mat - self.mean_) / self.std_
        X_norm = np.nan_to_num(X_norm, nan=0.0)
        return (X_norm @ self.weights) + self.intercept


class PureNumpyClassifier:
    """Zero-dependency Softmax Multi-class Classifier with L2 Regularization."""
    def __init__(self, lr: float = 0.05, max_iter: int = 150, l2_reg: float = 0.01):
        self.lr = lr
        self.max_iter = max_iter
        self.l2_reg = l2_reg
        self.W = None
        self.b = None
        self.n_classes = 2
        self.classes_ = None
        self.feature_importances_ = None
        self.mean_ = None
        self.std_ = None

    def _softmax(self, z):
        z_safe = z - np.max(z, axis=-1, keepdims=True)
        exp_z = np.exp(z_safe)
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    def fit(self, X, y):
        X_mat = np.asarray(X, dtype=float)
        y_vec = np.asarray(y, dtype=int)

        self.mean_ = np.nanmean(X_mat, axis=0)
        self.std_ = np.nanstd(X_mat, axis=0)
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)

        X_norm = (X_mat - self.mean_) / self.std_
        X_norm = np.nan_to_num(X_norm, nan=0.0)

        n_samples, n_features = X_norm.shape
        self.classes_ = np.unique(y_vec)
        self.n_classes = max(2, len(self.classes_))

        Y_onehot = np.zeros((n_samples, self.n_classes))
        for i, val in enumerate(y_vec):
            if val < self.n_classes:
                Y_onehot[i, val] = 1.0

        rng = np.random.RandomState(42)
        self.W = rng.randn(n_features, self.n_classes) * 0.01
        self.b = np.zeros(self.n_classes)

        for _ in range(self.max_iter):
            logits = X_norm @ self.W + self.b
            probs = self._softmax(logits)
            error = probs - Y_onehot

            grad_W = (X_norm.T @ error) / n_samples + self.l2_reg * self.W
            grad_b = np.mean(error, axis=0)

            self.W -= self.lr * grad_W
            self.b -= self.lr * grad_b

        raw_imp = np.sum(np.abs(self.W), axis=1)
        total_imp = np.sum(raw_imp)
        if total_imp > 0:
            self.feature_importances_ = raw_imp / total_imp
        else:
            self.feature_importances_ = np.ones(n_features) / n_features

        return self

    def predict_proba(self, X):
        X_mat = np.asarray(X, dtype=float)
        if X_mat.ndim == 1:
            X_mat = X_mat.reshape(1, -1)
        X_norm = (X_mat - self.mean_) / self.std_
        X_norm = np.nan_to_num(X_norm, nan=0.0)
        logits = X_norm @ self.W + self.b
        return self._softmax(logits)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


# ==========================================
# Predictive Pipeline
# ==========================================

class PredictivePipeline:
    """End-to-end predictive modeling for Classification & Regression."""

    @staticmethod
    def _prepare_data(
        df: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        task_type: str = "classification",
    ):
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset.")

        clean_df = df.copy()

        # If features not specified, use all columns except target
        if not feature_columns:
            feature_columns = [c for c in clean_df.columns if c != target_column]
        else:
            feature_columns = [c for c in feature_columns if c in clean_df.columns and c != target_column]

        if not feature_columns:
            raise ValueError("No valid feature columns provided.")

        # Drop rows where target is NaN
        clean_df = clean_df.dropna(subset=[target_column])

        # Separate X and y
        X_raw = clean_df[feature_columns].copy()
        y_raw = clean_df[target_column].copy()

        # Preprocess features: numeric impute with median, categorical with mode
        processed_features = {}
        categorical_encoders = {}
        target_encoder = None

        for col in feature_columns:
            series = X_raw[col]
            if pd.api.types.is_numeric_dtype(series):
                median_val = series.median()
                if pd.isna(median_val):
                    median_val = 0.0
                processed_features[col] = series.fillna(median_val).astype(float)
            else:
                mode_val = series.mode().iloc[0] if not series.mode().empty else "missing"
                filled = series.fillna(mode_val).astype(str)
                # Label encode strings
                if SKLEARN_AVAILABLE:
                    le = LabelEncoder()
                    processed_features[col] = le.fit_transform(filled)
                    categorical_encoders[col] = le
                else:
                    unique_vals = list(filled.unique())
                    mapping = {v: idx for idx, v in enumerate(unique_vals)}
                    processed_features[col] = filled.map(mapping)
                    categorical_encoders[col] = mapping

        X_mat = pd.DataFrame(processed_features)

        # Preprocess target
        classes_list = []
        if task_type == "classification":
            if SKLEARN_AVAILABLE:
                le_y = LabelEncoder()
                y_arr = le_y.fit_transform(y_raw.astype(str))
                classes_list = [str(c) for c in le_y.classes_]
                target_encoder = le_y
            else:
                unique_classes = list(y_raw.astype(str).unique())
                classes_list = [str(c) for c in unique_classes]
                mapping = {v: idx for idx, v in enumerate(unique_classes)}
                y_arr = np.array(y_raw.astype(str).map(mapping).fillna(0).astype(int))
                target_encoder = mapping
        else:
            # Regression: coerce to float
            y_arr = pd.to_numeric(y_raw, errors="coerce").fillna(0.0).to_numpy()

        return X_mat, y_arr, feature_columns, classes_list, categorical_encoders, target_encoder

    @classmethod
    def train_model(
        cls,
        df: pd.DataFrame,
        target_column: str,
        task_type: str = "classification",
        model_name: str = "random_forest",
        feature_columns: Optional[List[str]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        hyperparameters = hyperparameters or {}

        # 1. Prepare data
        X, y, features, classes_list, cat_encoders, target_encoder = cls._prepare_data(
            df=df,
            target_column=target_column,
            feature_columns=feature_columns,
            task_type=task_type,
        )

        n_samples = len(X)
        if n_samples < 5:
            raise ValueError("Dataset has too few samples (minimum 5 rows required for training).")

        # 2. Train/Test split
        test_size = float(hyperparameters.get("test_size", 0.2))
        test_size = max(0.1, min(0.4, test_size))

        if SKLEARN_AVAILABLE:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            # Pure numpy split
            split_idx = int(n_samples * (1 - test_size))
            perm = np.random.RandomState(42).permutation(n_samples)
            train_idx, test_idx = perm[:split_idx], perm[split_idx:]
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            mean_X = X_train.mean(axis=0)
            std_X = X_train.std(axis=0).replace(0, 1)
            X_train_scaled = ((X_train - mean_X) / std_X).to_numpy()
            X_test_scaled = ((X_test - mean_X) / std_X).to_numpy()
            scaler = {"mean": mean_X.to_dict(), "std": std_X.to_dict()}

        model = None
        feature_importances = {}
        effective_model_name = model_name

        # 3. Model instantiation & training
        if task_type == "classification":
            n_estimators = int(hyperparameters.get("n_estimators", 100))
            max_depth = hyperparameters.get("max_depth", None)
            if max_depth is not None:
                try:
                    max_depth = int(max_depth)
                except (ValueError, TypeError):
                    max_depth = None

            if model_name == "random_forest" and SKLEARN_AVAILABLE:
                model = RandomForestClassifier(
                    n_estimators=n_estimators, max_depth=max_depth, random_state=42
                )
                model.fit(X_train, y_train)
                importances = model.feature_importances_
                feature_importances = {f: round(float(imp), 4) for f, imp in zip(features, importances)}

            elif model_name == "xgboost":
                if XGBOOST_AVAILABLE:
                    model = xgb.XGBClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth or 6,
                        learning_rate=float(hyperparameters.get("learning_rate", 0.1)),
                        random_state=42,
                        eval_metric="logloss",
                    )
                    model.fit(X_train, y_train)
                    importances = model.feature_importances_
                    feature_importances = {f: round(float(imp), 4) for f, imp in zip(features, importances)}
                elif SKLEARN_AVAILABLE:
                    effective_model_name = "Gradient Boosting (XGBoost fallback)"
                    model = GradientBoostingClassifier(
                        n_estimators=n_estimators, max_depth=max_depth or 3, random_state=42
                    )
                    model.fit(X_train, y_train)
                    importances = model.feature_importances_
                    feature_importances = {f: round(float(imp), 4) for f, imp in zip(features, importances)}

            elif model_name == "neural_network" and SKLEARN_AVAILABLE:
                hidden_layers = hyperparameters.get("hidden_layers", (64, 32))
                epochs = int(hyperparameters.get("epochs", 150))
                model = MLPClassifier(
                    hidden_layer_sizes=hidden_layers,
                    max_iter=epochs,
                    random_state=42,
                    early_stopping=True,
                )
                model.fit(X_train_scaled, y_train)
                first_weights = np.abs(model.coefs_[0]).mean(axis=1)
                total_w = first_weights.sum() or 1.0
                feature_importances = {f: round(float(w / total_w), 4) for f, w in zip(features, first_weights)}

            # Fallback when scikit-learn is not installed or other models unavailable
            if model is None:
                if SKLEARN_AVAILABLE:
                    model = RandomForestClassifier(n_estimators=50, random_state=42)
                    model.fit(X_train, y_train)
                    importances = model.feature_importances_
                else:
                    effective_model_name = f"{model_name.replace('_', ' ').title()} (NumPy Engine)"
                    model = PureNumpyClassifier(max_iter=150)
                    model.fit(X_train, y_train)
                    importances = model.feature_importances_
                feature_importances = {f: round(float(imp), 4) for f, imp in zip(features, importances)}

            # 4. Evaluate Classification
            if model_name == "neural_network" and SKLEARN_AVAILABLE:
                y_pred = model.predict(X_test_scaled)
                y_train_pred = model.predict(X_train_scaled)
            else:
                y_pred = model.predict(X_test)
                y_train_pred = model.predict(X_train)

            y_pred = np.asarray(y_pred, dtype=int)
            y_train_pred = np.asarray(y_train_pred, dtype=int)
            y_test_arr = np.asarray(y_test, dtype=int)
            y_train_arr = np.asarray(y_train, dtype=int)

            n_cls = max(len(classes_list), int(np.max(y_test_arr) + 1) if len(y_test_arr) > 0 else 2)

            if SKLEARN_AVAILABLE:
                acc = float(accuracy_score(y_test_arr, y_pred))
                train_acc = float(accuracy_score(y_train_arr, y_train_pred))
                prec = float(precision_score(y_test_arr, y_pred, average="weighted", zero_division=0))
                rec = float(recall_score(y_test_arr, y_pred, average="weighted", zero_division=0))
                f1 = float(f1_score(y_test_arr, y_pred, average="weighted", zero_division=0))
                cm = confusion_matrix(y_test_arr, y_pred).tolist()
            else:
                acc = calc_accuracy(y_test_arr, y_pred)
                train_acc = calc_accuracy(y_train_arr, y_train_pred)
                prec, rec, f1 = calc_precision_recall_f1(y_test_arr, y_pred, n_cls)
                cm = calc_confusion_matrix(y_test_arr, y_pred, n_cls)

            # Format sample test predictions
            sample_predictions = []
            test_indices = list(range(min(20, len(y_test_arr))))
            for idx in test_indices:
                act_idx = int(y_test_arr[idx])
                pred_idx = int(y_pred[idx])
                act_str = classes_list[act_idx] if act_idx < len(classes_list) else str(act_idx)
                pred_str = classes_list[pred_idx] if pred_idx < len(classes_list) else str(pred_idx)
                sample_predictions.append({
                    "index": int(idx + 1),
                    "actual": act_str,
                    "predicted": pred_str,
                    "correct": bool(act_idx == pred_idx),
                })

            metrics = {
                "accuracy": round(acc, 4),
                "train_accuracy": round(train_acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "confusion_matrix": cm,
                "classes": classes_list if classes_list else [f"Class {i}" for i in range(n_cls)],
                "sample_predictions": sample_predictions,
            }

        else:
            # Regression Task
            n_estimators = int(hyperparameters.get("n_estimators", 100))
            max_depth = hyperparameters.get("max_depth", None)
            if max_depth is not None:
                try:
                    max_depth = int(max_depth)
                except (ValueError, TypeError):
                    max_depth = None

            if model_name == "random_forest" and SKLEARN_AVAILABLE:
                model = RandomForestRegressor(
                    n_estimators=n_estimators, max_depth=max_depth, random_state=42
                )
                model.fit(X_train, y_train)
                importances = model.feature_importances_
                feature_importances = {f: round(float(imp), 4) for f, imp in zip(features, importances)}

            elif model_name == "xgboost":
                if XGBOOST_AVAILABLE:
                    model = xgb.XGBRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth or 6,
                        learning_rate=float(hyperparameters.get("learning_rate", 0.1)),
                        random_state=42,
                    )
                    model.fit(X_train, y_train)
                    importances = model.feature_importances_
                    feature_importances = {f: round(float(imp), 4) for f, imp in zip(features, importances)}
                elif SKLEARN_AVAILABLE:
                    effective_model_name = "Gradient Boosting (XGBoost fallback)"
                    model = GradientBoostingRegressor(
                        n_estimators=n_estimators, max_depth=max_depth or 3, random_state=42
                    )
                    model.fit(X_train, y_train)
                    importances = model.feature_importances_
                    feature_importances = {f: round(float(imp), 4) for f, imp in zip(features, importances)}

            elif model_name == "neural_network" and SKLEARN_AVAILABLE:
                hidden_layers = hyperparameters.get("hidden_layers", (64, 32))
                epochs = int(hyperparameters.get("epochs", 150))
                model = MLPRegressor(
                    hidden_layer_sizes=hidden_layers,
                    max_iter=epochs,
                    random_state=42,
                    early_stopping=True,
                )
                model.fit(X_train_scaled, y_train)
                first_weights = np.abs(model.coefs_[0]).mean(axis=1)
                total_w = first_weights.sum() or 1.0
                feature_importances = {f: round(float(w / total_w), 4) for f, w in zip(features, first_weights)}

            # Fallback when scikit-learn is not installed or other models unavailable
            if model is None:
                if SKLEARN_AVAILABLE:
                    model = RandomForestRegressor(n_estimators=50, random_state=42)
                    model.fit(X_train, y_train)
                    importances = model.feature_importances_
                else:
                    effective_model_name = f"{model_name.replace('_', ' ').title()} (NumPy Engine)"
                    model = PureNumpyRegressor()
                    model.fit(X_train, y_train)
                    importances = model.feature_importances_
                feature_importances = {f: round(float(imp), 4) for f, imp in zip(features, importances)}

            # Evaluate Regression
            if model_name == "neural_network" and SKLEARN_AVAILABLE:
                y_pred = model.predict(X_test_scaled)
                y_train_pred = model.predict(X_train_scaled)
            else:
                y_pred = model.predict(X_test)
                y_train_pred = model.predict(X_train)

            y_pred = np.asarray(y_pred, dtype=float)
            y_train_pred = np.asarray(y_train_pred, dtype=float)
            y_test_arr = np.asarray(y_test, dtype=float)
            y_train_arr = np.asarray(y_train, dtype=float)

            if SKLEARN_AVAILABLE:
                r2 = float(r2_score(y_test_arr, y_pred))
                train_r2 = float(r2_score(y_train_arr, y_train_pred))
                mse = float(mean_squared_error(y_test_arr, y_pred))
                rmse = float(np.sqrt(mse))
                mae = float(mean_absolute_error(y_test_arr, y_pred))
            else:
                r2 = calc_r2(y_test_arr, y_pred)
                train_r2 = calc_r2(y_train_arr, y_train_pred)
                mse = calc_mse(y_test_arr, y_pred)
                rmse = float(np.sqrt(mse))
                mae = calc_mae(y_test_arr, y_pred)

            # MAPE (Mean Absolute Percentage Error)
            non_zero_mask = y_test_arr != 0
            if np.any(non_zero_mask):
                mape = float(np.mean(np.abs((y_test_arr[non_zero_mask] - y_pred[non_zero_mask]) / y_test_arr[non_zero_mask])) * 100)
            else:
                mape = 0.0

            # Sample test predictions for actual vs predicted scatter plot
            sample_predictions = []
            test_indices = list(range(min(30, len(y_test_arr))))
            for idx in test_indices:
                sample_predictions.append({
                    "index": int(idx + 1),
                    "actual": round(float(y_test_arr[idx]), 3),
                    "predicted": round(float(y_pred[idx]), 3),
                    "residual": round(float(y_test_arr[idx] - y_pred[idx]), 3),
                })

            metrics = {
                "r2_score": round(r2, 4),
                "train_r2_score": round(train_r2, 4),
                "rmse": round(rmse, 4),
                "mae": round(mae, 4),
                "mse": round(mse, 4),
                "mape_pct": round(mape, 2),
                "sample_predictions": sample_predictions,
            }

        # 5. Register model for live simulation/predict
        model_id = registry.register_model(
            model=model,
            task_type=task_type,
            target_column=target_column,
            feature_columns=features,
            preprocessor={
                "scaler": scaler,
                "cat_encoders": cat_encoders,
                "target_encoder": target_encoder,
                "classes_list": classes_list,
            },
            model_name=effective_model_name,
            meta={"metrics": metrics, "features": features},
        )

        # Sort feature importances descending
        sorted_importances = sorted(
            [{"feature": k, "importance": v} for k, v in feature_importances.items()],
            key=lambda x: x["importance"],
            reverse=True,
        )

        return {
            "model_id": model_id,
            "task_type": task_type,
            "model_name": effective_model_name,
            "target_column": target_column,
            "feature_columns": features,
            "metrics": metrics,
            "feature_importances": sorted_importances,
            "test_sample_count": len(y_test),
            "train_sample_count": len(y_train),
        }

    @classmethod
    def predict_single(cls, model_id: str, input_features: Dict[str, Any]) -> Dict[str, Any]:
        """Makes an instantaneous inference with a registered model."""
        entry = registry.get_model(model_id)
        if not entry:
            raise ValueError(f"Model ID '{model_id}' not found or session expired.")

        model = entry["model"]
        features = entry["feature_columns"]
        task_type = entry["task_type"]
        preprocessor = entry["preprocessor"]
        classes_list = preprocessor.get("classes_list", [])

        # Build feature vector
        vector = []
        for col in features:
            raw_val = input_features.get(col, 0.0)
            if col in preprocessor.get("cat_encoders", {}):
                encoder = preprocessor["cat_encoders"][col]
                if hasattr(encoder, "transform"):
                    try:
                        encoded = encoder.transform([str(raw_val)])[0]
                    except Exception:
                        encoded = 0
                elif isinstance(encoder, dict):
                    encoded = encoder.get(str(raw_val), 0)
                else:
                    encoded = 0
                vector.append(float(encoded))
            else:
                try:
                    vector.append(float(raw_val))
                except (ValueError, TypeError):
                    vector.append(0.0)

        df_input = pd.DataFrame([vector], columns=features)

        # Handle scaling for neural networks
        if "neural_network" in entry["model_name"].lower() and hasattr(preprocessor.get("scaler"), "transform"):
            input_mat = preprocessor["scaler"].transform(df_input)
            raw_pred = model.predict(input_mat)[0]
        else:
            raw_pred = model.predict(df_input)[0]

        if task_type == "classification":
            pred_class_idx = int(raw_pred)
            if classes_list and pred_class_idx < len(classes_list):
                formatted_pred = classes_list[pred_class_idx]
            else:
                formatted_pred = str(pred_class_idx)

            probabilities = None
            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(df_input)[0]
                    probabilities = {
                        classes_list[i] if i < len(classes_list) else f"Class_{i}": round(float(p), 4)
                        for i, p in enumerate(proba)
                    }
                except Exception:
                    pass

            return {
                "prediction": formatted_pred,
                "probabilities": probabilities,
                "task_type": "classification",
                "model_name": entry["model_name"],
            }
        else:
            return {
                "prediction": round(float(raw_pred), 4),
                "task_type": "regression",
                "model_name": entry["model_name"],
            }
