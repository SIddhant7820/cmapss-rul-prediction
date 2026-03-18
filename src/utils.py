from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def _require_mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Expected '{name}' to be a mapping in config/config.yaml.")
    return value  # type: ignore[return-value]


def _require_str(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected '{name}' to be a non-empty string in config/config.yaml.")
    return value


def load_config(config_path: str = "config/config.yaml") -> dict[str, Any]:
    """
    Load the YAML configuration file and return it as a dictionary.

    Args:
        config_path: Path to the YAML config file, relative to project root by default.

    Returns:
        Parsed configuration as a dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config file is empty or cannot be parsed as a mapping.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found at '{path.as_posix()}'. "
            "Expected a YAML file (e.g., 'config/config.yaml')."
        )

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or not data:
        raise ValueError(
            f"Config at '{path.as_posix()}' is empty or invalid. "
            "Expected a non-empty YAML mapping."
        )
    return data


def get_logger(name: str) -> logging.Logger:
    """
    Create (or retrieve) a configured logger.

    The logger reads `logging.log_file` and `logging.level` from `config/config.yaml`,
    logs to both console and file, and uses this format:
    `[TIMESTAMP] [LEVEL] [name] - message`

    Args:
        name: Logger name (usually `__name__`).

    Returns:
        A configured `logging.Logger`.
    """
    cfg = load_config()
    logging_cfg = _require_mapping(cfg.get("logging"), name="logging")

    level_name = _require_str(logging_cfg.get("level"), name="logging.level").upper()
    level = logging.getLevelName(level_name)
    if not isinstance(level, int):
        raise ValueError(
            f"Invalid logging level '{level_name}' in config/config.yaml (logging.level)."
        )

    log_file = Path(_require_str(logging_cfg.get("log_file"), name="logging.log_file"))
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for full reproducibility.

    This seeds Python's `random`, NumPy, and PyTorch, and configures deterministic
    behavior where supported.

    Args:
        seed: Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        # Some ops/devices may not support full determinism; still keep best-effort.
        pass


def get_device() -> str:
    """
    Determine the compute device to use.

    Returns:
        'cuda' if a compatible GPU is available, otherwise 'cpu'.
    """
    logger = get_logger(__name__)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)
    return device


def ensure_dirs() -> None:
    """
    Ensure required project directories exist.

    Reads all relevant paths from `config/config.yaml` and creates directories if missing:
    - dataset.raw_path (e.g., data/raw)
    - dataset.processed_path (e.g., data/processed)
    - paths.model_save_path (e.g., models/saved)
    - paths.logs_path (e.g., logs)
    - paths.plots_path (e.g., logs/plots)
    """
    cfg = load_config()

    dataset_cfg = cfg.get("dataset", {})
    paths_cfg = cfg.get("paths", {})

    raw_path = Path(str(dataset_cfg.get("raw_path", "")))
    processed_path = Path(str(dataset_cfg.get("processed_path", "")))

    model_save_path = Path(str(paths_cfg.get("model_save_path", "")))
    logs_path = Path(str(paths_cfg.get("logs_path", "")))
    plots_path = Path(str(paths_cfg.get("plots_path", "")))
    scaler_path = Path(str(paths_cfg.get("scaler_path", "")))

    to_create = [
        raw_path,
        processed_path,
        model_save_path,
        logs_path,
        plots_path,
        scaler_path.parent if scaler_path.name else scaler_path,
    ]

    for p in to_create:
        if str(p).strip():
            p.mkdir(parents=True, exist_ok=True)
