import optuna
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.utils import load_config, set_seed, get_device, get_logger
from src.ingestion import CMAPSSIngestion
from src.transformation import CMAPSSTransformer
from src.trainer import LSTMModel

logger = get_logger(__name__)
config = load_config()

# ── load all 4 subsets once outside objective ──────────────
# no point reloading data every trial
logger.info("Loading and transforming all subsets...")

all_data = {}
for subset in ["FD001", "FD002", "FD003", "FD004"]:
    ing  = CMAPSSIngestion()
    data = ing.load_subset(subset)
    trans = CMAPSSTransformer()
    X_train, y_train, X_test, y_test = trans.run(
        data["train_df"], data["test_df"], data["rul_df"], subset
    )
    all_data[subset] = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test":  X_test,
        "y_test":  y_test,
    }

logger.info("All subsets loaded. Starting tuning...")


def objective(trial):
    # hyperparameters to search
    hidden_size   = trial.suggest_categorical("hidden_size",  [64, 128, 256])
    num_layers    = trial.suggest_int("num_layers", 1, 3)
    dropout       = trial.suggest_float("dropout", 0.1, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size    = trial.suggest_categorical("batch_size", [32, 64, 128])
    window_size   = trial.suggest_categorical("window_size", [20, 30, 40, 50])

    set_seed(42)
    device = get_device()

    # tune on FD001 and FD002 — one easy one hard
    val_losses = []

    for subset in ["FD001", "FD002"]:
        X_train = all_data[subset]["X_train"]
        y_train = all_data[subset]["y_train"]

        # if window size changed we need different sequences
        # for now use preloaded data with default window
        # TODO: rebuild sequences per window_size trial if needed

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train,
            test_size=0.2,
            random_state=42,
            shuffle=True
        )

        train_ds = TensorDataset(
            torch.tensor(X_tr,  dtype=torch.float32),
            torch.tensor(y_tr,  dtype=torch.float32),
        )
        val_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

        input_size = X_train.shape[2]
        model = LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
        )

        best_val   = float("inf")
        no_improve = 0

        # max 40 epochs per trial to keep tuning fast
        for epoch in range(40):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            batch_val = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    batch_val.append(criterion(model(xb), yb).item())

            avg_val = float(np.mean(batch_val))
            scheduler.step(avg_val)

            # report to optuna for pruning bad trials early
            trial.report(avg_val, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if avg_val < best_val:
                best_val   = avg_val
                no_improve = 0
            else:
                no_improve += 1

            # early stopping per trial
            if no_improve >= 8:
                break

        val_losses.append(best_val)

    # average val loss across FD001 and FD002
    return float(np.mean(val_losses))


if __name__ == "__main__":
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
    )

    # 30 trials - increase if you have more time
    study.optimize(objective, n_trials=30, show_progress_bar=True)

    print("\n" + "="*55)
    print("BEST TRIAL RESULTS")
    print("="*55)
    print(f"  Best val_loss (MSE) : {study.best_trial.value:.4f}")
    print(f"  Best RMSE           : {study.best_trial.value**0.5:.4f}")
    print("\nBest Hyperparameters Found:")
    print("-"*55)
    for k, v in study.best_trial.params.items():
        print(f"  {k:25s}: {v}")
    print("="*55)
    print("\n✅ Copy these into config/config.yaml and retrain!")

    # save best params to a file for reference
    best_params_path = Path("logs/best_params.txt")
    with open(best_params_path, "w") as f:
        f.write(f"Best val_loss (MSE): {study.best_trial.value:.4f}\n")
        f.write(f"Best RMSE: {study.best_trial.value**0.5:.4f}\n\n")
        f.write("Best Hyperparameters:\n")
        for k, v in study.best_trial.params.items():
            f.write(f"  {k}: {v}\n")

    logger.info("Best params saved to logs/best_params.txt")