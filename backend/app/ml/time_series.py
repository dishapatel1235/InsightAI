import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

TORCH_AVAILABLE = False
SKLEARN_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.ensemble import RandomForestRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


if TORCH_AVAILABLE:
    class PyTorchLSTM(nn.Module):
        def __init__(self, input_size=1, hidden_size=32, num_layers=1):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            # x shape: (batch_size, seq_len, 1)
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])
            return out


class TimeSeriesPipeline:
    """Time-series forecasting with LSTM, Neural Network, and Sequential ensembles."""

    @classmethod
    def forecast(
        cls,
        df: pd.DataFrame,
        target_column: str,
        time_column: Optional[str] = None,
        model_name: str = "lstm",
        lookback: int = 5,
        horizon: int = 10,
        epochs: int = 50,
    ) -> Dict[str, Any]:
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found.")

        work_df = df.copy()

        # Handle chronological sorting
        if time_column and time_column in work_df.columns:
            try:
                work_df[time_column] = pd.to_datetime(work_df[time_column])
                work_df = work_df.sort_values(by=time_column).reset_index(drop=True)
                time_labels = work_df[time_column].dt.strftime("%Y-%m-%d").tolist()
            except Exception:
                time_labels = [str(x) for x in work_df[time_column].tolist()]
        else:
            time_labels = [f"t_{i+1}" for i in range(len(work_df))]

        # Extract numeric series
        series = pd.to_numeric(work_df[target_column], errors="coerce").dropna().values
        n_points = len(series)

        if n_points < 10:
            raise ValueError("Time series must have at least 10 observations for forecasting.")

        lookback = max(2, min(lookback, n_points // 3, 20))
        horizon = max(1, min(horizon, 60))

        # Scale series to [0, 1] for stable neural training
        min_val = float(np.min(series))
        max_val = float(np.max(series))
        denom = (max_val - min_val) if (max_val - min_val) != 0 else 1.0
        scaled_series = (series - min_val) / denom

        # Create sliding window sequences
        X_seq, y_seq = [], []
        for i in range(len(scaled_series) - lookback):
            X_seq.append(scaled_series[i : i + lookback])
            y_seq.append(scaled_series[i + lookback])

        X_mat = np.array(X_seq)
        y_mat = np.array(y_seq)

        # Train/validation split
        val_size = max(2, min(10, int(len(X_mat) * 0.2)))
        X_train, X_val = X_mat[:-val_size], X_mat[-val_size:]
        y_train, y_val = y_mat[:-val_size], y_mat[-val_size:]

        fitted_scaled = []
        val_preds_scaled = []

        # Train model
        if model_name == "lstm" and TORCH_AVAILABLE:
            torch.manual_seed(42)
            lstm_model = PyTorchLSTM(input_size=1, hidden_size=32)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.01)

            t_X = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
            t_y = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)

            lstm_model.train()
            for _ in range(min(epochs, 80)):
                optimizer.zero_grad()
                output = lstm_model(t_X)
                loss = criterion(output, t_y)
                loss.backward()
                optimizer.step()

            lstm_model.eval()
            with torch.no_grad():
                all_X = torch.tensor(X_mat, dtype=torch.float32).unsqueeze(-1)
                fitted_scaled = lstm_model(all_X).squeeze().numpy().tolist()
                if not isinstance(fitted_scaled, list):
                    fitted_scaled = [fitted_scaled]
                val_preds_scaled = fitted_scaled[-val_size:]

            # Autoregressive multi-step future forecasting
            curr_window = list(scaled_series[-lookback:])
            future_scaled = []
            with torch.no_grad():
                for _ in range(horizon):
                    in_t = torch.tensor([curr_window[-lookback:]], dtype=torch.float32).unsqueeze(-1)
                    pred_step = float(lstm_model(in_t).item())
                    future_scaled.append(pred_step)
                    curr_window.append(pred_step)

        else:
            # Scikit-learn MLP / Random Forest sequential forecaster
            regressor = None
            if model_name in ("lstm", "neural_net") and SKLEARN_AVAILABLE:
                regressor = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=epochs, random_state=42)
            elif SKLEARN_AVAILABLE:
                regressor = RandomForestRegressor(n_estimators=50, random_state=42)

            if regressor:
                regressor.fit(X_train, y_train)
                fitted_scaled = regressor.predict(X_mat).tolist()
                val_preds_scaled = fitted_scaled[-val_size:]

                # Autoregressive forecast
                curr_window = list(scaled_series[-lookback:])
                future_scaled = []
                for _ in range(horizon):
                    step_input = np.array([curr_window[-lookback:]])
                    next_val = float(regressor.predict(step_input)[0])
                    future_scaled.append(next_val)
                    curr_window.append(next_val)
            else:
                # Robust pure NumPy Autoregressive Ridge forecaster
                reg = 1e-3 * np.eye(lookback)
                try:
                    beta = np.linalg.solve(X_train.T @ X_train + reg, X_train.T @ y_train)
                except np.linalg.LinAlgError:
                    beta = np.linalg.pinv(X_train.T @ X_train + reg) @ (X_train.T @ y_train)

                fitted_scaled = (X_mat @ beta).tolist()
                val_preds_scaled = fitted_scaled[-val_size:]

                curr_window = list(scaled_series[-lookback:])
                future_scaled = []
                for _ in range(horizon):
                    step_vec = np.array(curr_window[-lookback:])
                    next_val = float(step_vec @ beta)
                    # Clip extreme explosions to reasonable scaled range
                    next_val = max(-0.5, min(1.5, next_val))
                    future_scaled.append(next_val)
                    curr_window.append(next_val)

        # Rescale back to original data units
        fitted_orig = [round(float(v * denom + min_val), 3) for v in fitted_scaled]
        val_preds_orig = np.array(val_preds_scaled) * denom + min_val
        val_actuals_orig = np.array(y_val) * denom + min_val

        # Forecasting accuracy metrics on validation split
        mae = float(np.mean(np.abs(val_actuals_orig - val_preds_orig)))
        rmse = float(np.sqrt(np.mean((val_actuals_orig - val_preds_orig) ** 2)))
        mape_denom = np.where(val_actuals_orig == 0, 1e-6, val_actuals_orig)
        mape = float(np.mean(np.abs((val_actuals_orig - val_preds_orig) / mape_denom)) * 100)

        # Future forecast points
        future_forecast = [round(float(v * denom + min_val), 3) for v in future_scaled]
        residual_std = max(float(np.std(val_actuals_orig - val_preds_orig)), 0.01 * denom)

        # 95% Confidence Bounds (expanding over horizon)
        bounds = []
        for step, val in enumerate(future_forecast):
            uncertainty = 1.96 * residual_std * np.sqrt(step + 1)
            bounds.append({
                "step": step + 1,
                "forecast": val,
                "lower_bound": round(float(val - uncertainty), 3),
                "upper_bound": round(float(val + uncertainty), 3),
            })

        # Future time labels
        future_labels = []
        try:
            last_date = pd.to_datetime(time_labels[-1])
            # Determine daily or general frequency
            for step in range(1, horizon + 1):
                future_date = last_date + pd.Timedelta(days=step)
                future_labels.append(future_date.strftime("%Y-%m-%d"))
        except Exception:
            future_labels = [f"t+{step+1}" for step in range(horizon)]

        # Prepare chart timeline format
        historical_chart_data = []
        for i in range(n_points):
            entry = {
                "label": time_labels[i],
                "actual": round(float(series[i]), 3),
                "fitted": fitted_orig[i - lookback] if i >= lookback and (i - lookback) < len(fitted_orig) else None,
                "forecast": None,
                "lower_bound": None,
                "upper_bound": None,
            }
            historical_chart_data.append(entry)

        # Append future horizon
        forecast_chart_data = []
        for i in range(horizon):
            entry = {
                "label": future_labels[i],
                "actual": None,
                "fitted": None,
                "forecast": future_forecast[i],
                "lower_bound": bounds[i]["lower_bound"],
                "upper_bound": bounds[i]["upper_bound"],
            }
            forecast_chart_data.append(entry)

        display_name = (
            "LSTM Neural Network"
            if model_name == "lstm" and TORCH_AVAILABLE
            else ("Neural Network (MLP)" if model_name in ("lstm", "neural_net") and SKLEARN_AVAILABLE else f"{model_name.replace('_', ' ').title()} Forecaster")
        )

        return {
            "model_name": display_name,
            "target_column": target_column,
            "lookback_window": lookback,
            "horizon": horizon,
            "metrics": {
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "mape_pct": round(mape, 2),
            },
            "timeline": historical_chart_data + forecast_chart_data,
            "future_forecast": bounds,
            "last_observed_value": round(float(series[-1]), 3),
        }
