from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.trainer import LSTMModel
from src.utils import get_device, get_logger, load_config


class RULPredictor:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.logger = get_logger(__name__)
        self.device = get_device()
        paths = self.config["paths"]
        self.model_save_dir = Path(paths["model_save_path"])
        self.scaler_path = Path(paths["scaler_path"])  # sibling naming: scaler_{subset}.pkl
        self.logs_dir = Path(paths["logs_path"])
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def load_lstm(self, subset: str) -> LSTMModel:
        path = self.model_save_dir / f"lstm_{subset}.pt"
        raw = torch.load(path, map_location=self.device, weights_only=False)

        if isinstance(raw, dict) and "state_dict" in raw:
            ckpt = raw
            input_size = int(ckpt["input_size"])
            hidden_size = int(ckpt["hidden_size"])
            num_layers = int(ckpt["num_layers"])
            dropout = float(ckpt["dropout"])
            state = ckpt["state_dict"]
        else:
            # legacy: plain state_dict only — infer input_size, rest from config
            state = raw
            w = state["lstm.weight_ih_l0"]
            input_size = int(w.shape[1])
            m = self.config["model"]
            hidden_size = int(m["hidden_size"])
            num_layers = int(m["num_layers"])
            dropout = float(m["dropout"])
            self.logger.warning(
                "Loaded legacy LSTM weights without checkpoint meta; using config for arch"
            )

        model = LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        ).to(self.device)
        model.load_state_dict(state)
        model.eval()
        return model

    def load_xgboost(self, subset: str):
        path = self.model_save_dir / f"xgb_{subset}.pkl"
        return joblib.load(path)

    def load_scaler(self, subset: str) -> dict:
        path = self.scaler_path.with_name(f"scaler_{subset}.pkl")
        return joblib.load(path)

    def predict_lstm(self, X_test: np.ndarray, subset: str) -> np.ndarray:
        model = self.load_lstm(subset)
        x = torch.tensor(X_test, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            out = model(x)
        return out.cpu().numpy().reshape(-1)

    def predict_xgboost(self, X_test: np.ndarray, subset: str) -> np.ndarray:
        model = self.load_xgboost(subset)
        n = X_test.shape[0]
        X_flat = X_test.reshape(n, -1)
        return np.asarray(model.predict(X_flat), dtype=np.float64).reshape(-1)

    def predict(
        self,
        X_test: np.ndarray,
        subset: str,
        model_type: str = "lstm",
    ) -> np.ndarray:
        model_type = model_type.lower().strip()
        if model_type == "lstm":
            y = self.predict_lstm(X_test, subset)
        elif model_type in ("xgboost", "xgb"):
            y = self.predict_xgboost(X_test, subset)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        y = np.clip(y, 0.0, None)
        self.logger.info(
            "Predictions (%s, %s): min=%.4f max=%.4f mean=%.4f",
            subset,
            model_type,
            float(np.min(y)),
            float(np.max(y)),
            float(np.mean(y)),
        )
        # TODO: optional batching for very large X_test on GPU
        return y

    def get_results_df(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        subset: str,
    ) -> pd.DataFrame:
        y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
        y_true = np.asarray(y_true, dtype=float).reshape(-1)
        if y_pred.shape[0] != y_true.shape[0]:
            raise ValueError("y_pred and y_true must have the same length")

        # assumes same order as test engines (sorted unit_id in pipeline)
        n = len(y_pred)
        df = pd.DataFrame(
            {
                "unit_id": np.arange(1, n + 1, dtype=int),
                "actual_RUL": y_true,
                "predicted_RUL": y_pred,
                "error": y_pred - y_true,
            }
        )
        self.logger.info("Results preview (%s):\n%s", subset, df.head(5).to_string(index=False))

        out_path = self.logs_dir / f"predictions_{subset}.csv"
        df.to_csv(out_path, index=False)
        self.logger.info("Predictions table saved to %s", out_path)
        # TODO: pass real unit_id column from ingestion if not 1..n order
        return df
