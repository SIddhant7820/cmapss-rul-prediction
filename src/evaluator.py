from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.utils import get_logger, load_config


class RULEvaluator:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.logger = get_logger(__name__)
        self.plots_dir = Path(self.config["paths"]["plots_path"])
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true, dtype=float).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    def mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true, dtype=float).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
        return float(np.mean(np.abs(y_true - y_pred)))

    def r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true, dtype=float).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            return 0.0
        return float(1.0 - (ss_res / ss_tot))

    def nasa_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true, dtype=float).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
        d = y_pred - y_true

        early_mask = d < 0
        late_mask = ~early_mask

        score_early = np.exp(-d[early_mask] / 13.0) - 1.0
        score_late = np.exp(d[late_mask] / 10.0) - 1.0
        return float(np.sum(score_early) + np.sum(score_late))

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, subset: str) -> dict[str, float]:
        y_true = np.asarray(y_true, dtype=float).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError("y_true and y_pred should have the same length")

        results = {
            "rmse": self.rmse(y_true, y_pred),
            "mae": self.mae(y_true, y_pred),
            "r2": self.r2(y_true, y_pred),
            "nasa_score": self.nasa_score(y_true, y_pred),
        }

        self.logger.info(
            "%s | RMSE=%.4f | MAE=%.4f | R2=%.4f | NASA=%.4f",
            subset,
            results["rmse"],
            results["mae"],
            results["r2"],
            results["nasa_score"],
        )
        return results

    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, subset: str) -> None:
        y_true = np.asarray(y_true, dtype=float).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError("y_true and y_pred should have the same length")

        sns.set_style("whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # scatter plot + perfect line
        sns.scatterplot(x=y_true, y=y_pred, s=35, alpha=0.7, ax=axes[0], color="#1f77b4")
        min_v = float(min(np.min(y_true), np.min(y_pred)))
        max_v = float(max(np.max(y_true), np.max(y_pred)))
        axes[0].plot([min_v, max_v], [min_v, max_v], "r--", lw=1.5)
        axes[0].set_title(f"Predicted vs Actual ({subset})")
        axes[0].set_xlabel("Actual RUL")
        axes[0].set_ylabel("Predicted RUL")

        # line plot sorted by actual RUL
        sort_idx = np.argsort(y_true)
        y_true_sorted = y_true[sort_idx]
        y_pred_sorted = y_pred[sort_idx]
        axes[1].plot(y_true_sorted, label="Actual RUL", lw=2)
        axes[1].plot(y_pred_sorted, label="Predicted RUL", lw=2)
        axes[1].set_title(f"RUL Curves Sorted by Actual ({subset})")
        axes[1].set_xlabel("Engine Index (sorted)")
        axes[1].set_ylabel("RUL")
        axes[1].legend()

        save_path = self.plots_dir / f"predictions_{subset}.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        plt.close(fig)
        self.logger.info("Prediction plot saved to %s", save_path)

    def print_report(self, results: dict[str, float], subset: str) -> None:
        line = "─" * 29
        print(line)
        print(f"Evaluation Report — {subset}")
        print(line)
        print(f"RMSE       : {results['rmse']:8.2f}")
        print(f"MAE        : {results['mae']:8.2f}")
        print(f"R2 Score   : {results['r2']:8.2f}")
        print(f"NASA Score : {results['nasa_score']:8.2f}")
        print(line)
