from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

from src.utils import get_device, get_logger, load_config, set_seed


class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        output = self.fc(last_hidden)
        return output


class RULTrainer:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.logger = get_logger(__name__)

        model_cfg = self.config["model"]
        self.hidden_size = model_cfg["hidden_size"]
        self.num_layers = model_cfg["num_layers"]
        self.dropout = model_cfg["dropout"]
        self.learning_rate = model_cfg["learning_rate"]
        self.batch_size = model_cfg["batch_size"]
        self.epochs = model_cfg["epochs"]
        self.early_stopping_patience = model_cfg["early_stopping_patience"]
        self.train_val_split = model_cfg["train_val_split"]
        self.seed = model_cfg["seed"]

        xgb_cfg = self.config["xgboost"]
        self.xgb_params = {
            "n_estimators": xgb_cfg["n_estimators"],
            "max_depth": xgb_cfg["max_depth"],
            "learning_rate": xgb_cfg["learning_rate"],
            "subsample": xgb_cfg["subsample"],
            "colsample_bytree": xgb_cfg["colsample_bytree"],
            "random_state": xgb_cfg["random_state"],
            "objective": "reg:squarederror",
        }

        paths_cfg = self.config["paths"]
        self.model_save_dir = Path(paths_cfg["model_save_path"])
        self.plots_dir = Path(paths_cfg["plots_path"])
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        set_seed(self.seed)
        self.device = get_device()

    def build_model(self, input_size: int) -> LSTMModel:
        model = LSTMModel(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        total_params = sum(p.numel() for p in model.parameters())
        self.logger.info("LSTM built with %d total parameters", total_params)
        return model

    def get_dataloaders(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> tuple[DataLoader, DataLoader]:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train,
            y_train,
            test_size=self.train_val_split,
            random_state=self.seed,
            shuffle=True,
        )

        train_ds = TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32).view(-1, 1),
        )
        val_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32).view(-1, 1),
        )

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader

    def train(
        self, X_train: np.ndarray, y_train: np.ndarray, subset: str
    ) -> tuple[LSTMModel, list[float], list[float]]:
        if X_train.ndim != 3:
            raise ValueError("X_train should be 3D: (samples, window, features)")

        input_size = X_train.shape[2]
        model = self.build_model(input_size)
        train_loader, val_loader = self.get_dataloaders(X_train, y_train)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
        )

        best_val = float("inf")
        best_state = None
        patience_counter = 0
        train_losses: list[float] = []
        val_losses: list[float] = []

        for epoch in range(1, self.epochs + 1):
            model.train()
            batch_train_losses: list[float] = []

            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()

                batch_train_losses.append(loss.item())

            avg_train = float(np.mean(batch_train_losses))
            train_losses.append(avg_train)

            model.eval()
            batch_val_losses: list[float] = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    preds = model(xb)
                    v_loss = criterion(preds, yb)
                    batch_val_losses.append(v_loss.item())

            avg_val = float(np.mean(batch_val_losses))
            val_losses.append(avg_val)
            scheduler.step(avg_val)

            self.logger.info(
                "Epoch %d/%d | train_loss=%.6f | val_loss=%.6f",
                epoch,
                self.epochs,
                avg_train,
                avg_val,
            )

            if avg_val < best_val:
                best_val = avg_val
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.early_stopping_patience:
                self.logger.info("Early stopping at epoch %d", epoch)
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        save_path = self.model_save_dir / f"lstm_{subset}.pt"
        torch.save(model.state_dict(), save_path)
        self.logger.info("Best LSTM model saved to %s", save_path)

        # TODO: store full checkpoint (optimizer/scheduler) for easy resume
        return model, train_losses, val_losses

    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray, subset: str) -> XGBRegressor:
        if X_train.ndim != 3:
            raise ValueError("X_train should be 3D: (samples, window, features)")

        n_samples = X_train.shape[0]
        X_flat = X_train.reshape(n_samples, -1)  # flatten (window, features)

        model = XGBRegressor(**self.xgb_params)
        model.fit(X_flat, y_train)

        save_path = self.model_save_dir / f"xgb_{subset}.pkl"
        joblib.dump(model, save_path)
        self.logger.info("XGBoost model saved to %s", save_path)

        # TODO: add a quick CV sweep for xgb params if needed
        return model

    def plot_losses(self, train_losses: list[float], val_losses: list[float], subset: str) -> None:
        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title(f"LSTM Training Curves ({subset})")
        plt.legend()
        plt.grid(alpha=0.3)

        save_path = self.plots_dir / f"loss_{subset}.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        plt.close()

        self.logger.info("Loss plot saved to %s", save_path)
