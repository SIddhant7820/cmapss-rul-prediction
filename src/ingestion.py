from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils import get_logger, load_config


class CMAPSSIngestion:
    """
    Data ingestion utilities for the NASA C-MAPSS dataset.

    This class loads the raw space-separated `.txt` files for a given subset (FD001-FD004),
    assigns column names, drops the trailing NaN columns, validates data, and can save
    train/test splits to Parquet in the configured raw data directory.
    """

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        """
        Initialize the ingestion pipeline.

        Args:
            config_path: Path to `config.yaml`.
        """
        self.config_path = config_path
        self.config: dict[str, Any] = load_config(config_path)
        self.logger = get_logger(self.__class__.__name__)

        dataset_cfg = self.config.get("dataset", {})
        if not isinstance(dataset_cfg, dict):
            raise ValueError("Expected 'dataset' section to be a mapping in config/config.yaml.")

        raw_path = dataset_cfg.get("raw_path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError("Expected 'dataset.raw_path' to be a non-empty string in config/config.yaml.")
        self.raw_dir = Path(raw_path)

        column_names = dataset_cfg.get("column_names")
        if not isinstance(column_names, list) or not all(isinstance(x, str) for x in column_names):
            raise ValueError(
                "Expected 'dataset.column_names' to be a list of strings in config/config.yaml."
            )
        self.column_names: list[str] = list(column_names)

        subsets = dataset_cfg.get("subsets")
        if not isinstance(subsets, list) or not all(isinstance(x, str) for x in subsets):
            raise ValueError("Expected 'dataset.subsets' to be a list of strings in config/config.yaml.")
        self.subsets: list[str] = list(subsets)

    def load_subset(self, subset: str) -> dict[str, pd.DataFrame]:
        """
        Load a single C-MAPSS subset from raw `.txt` files.

        Expects the following files to exist inside `dataset.raw_path`:
        - `train_{subset}.txt`
        - `test_{subset}.txt`
        - `RUL_{subset}.txt`

        The train/test files are space-separated with no header and include two trailing
        empty columns that must be dropped.

        Args:
            subset: One of `FD001`, `FD002`, `FD003`, `FD004`.

        Returns:
            Dictionary with keys: `train_df`, `test_df`, `rul_df`.

        Raises:
            FileNotFoundError: If any required file is missing.
            ValueError: If the loaded files do not match expected shapes.
        """
        subset = str(subset).strip()
        train_path = self.raw_dir / f"train_{subset}.txt"
        test_path = self.raw_dir / f"test_{subset}.txt"
        rul_path = self.raw_dir / f"RUL_{subset}.txt"

        missing = [p for p in (train_path, test_path, rul_path) if not p.exists()]
        if missing:
            missing_list = ", ".join(f"'{p.as_posix()}'" for p in missing)
            raise FileNotFoundError(
                f"Missing required C-MAPSS file(s) for subset '{subset}': {missing_list}"
            )

        train_df = self._read_cmapss_txt(train_path)
        test_df = self._read_cmapss_txt(test_path)
        rul_df = self._read_rul_txt(rul_path)

        data = {"train_df": train_df, "test_df": test_df, "rul_df": rul_df}
        return data

    def load_all(self) -> dict[str, dict[str, pd.DataFrame]]:
        """
        Load all configured C-MAPSS subsets.

        Returns:
            Dictionary keyed by subset name (e.g., `FD001`) where each value contains
            `train_df`, `test_df`, and `rul_df`.
        """
        out: dict[str, dict[str, pd.DataFrame]] = {}
        for subset in self.subsets:
            out[subset] = self.load_subset(subset)
        return out

    def save_raw(self, data: dict[str, pd.DataFrame], subset: str) -> None:
        """
        Save raw train/test dataframes to Parquet inside `dataset.raw_path`.

        Args:
            data: Dictionary containing `train_df` and `test_df`.
            subset: Subset identifier (e.g., `FD001`).
        """
        subset = str(subset).strip()
        train_df = data.get("train_df")
        test_df = data.get("test_df")
        if not isinstance(train_df, pd.DataFrame) or not isinstance(test_df, pd.DataFrame):
            raise ValueError("Expected 'data' to contain pandas DataFrames for 'train_df' and 'test_df'.")

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        train_out = self.raw_dir / f"train_{subset}.parquet"
        test_out = self.raw_dir / f"test_{subset}.parquet"

        train_df.to_parquet(train_out, index=False)
        self.logger.info("Saved %s", train_out.as_posix())

        test_df.to_parquet(test_out, index=False)
        self.logger.info("Saved %s", test_out.as_posix())

    def validate(self, data: dict[str, pd.DataFrame], subset: str) -> None:
        """
        Validate loaded subset data.

        Checks:
        - `train_df` has 26 columns (unit_id, cycle, op1-op3, s1-s21)
        - no fully empty columns in `train_df`
        - `unit_id` and `cycle` columns exist
        - `rul_df` is not empty

        Args:
            data: Dictionary containing `train_df`, `test_df`, and `rul_df`.
            subset: Subset identifier for logging.
        """
        subset = str(subset).strip()
        train_df = data.get("train_df")
        rul_df = data.get("rul_df")
        if not isinstance(train_df, pd.DataFrame) or not isinstance(rul_df, pd.DataFrame):
            raise ValueError("Expected 'data' to contain pandas DataFrames for 'train_df' and 'rul_df'.")

        expected_cols = 26
        if train_df.shape[1] != expected_cols:
            raise ValueError(
                f"Validation failed for subset '{subset}': expected train_df to have "
                f"{expected_cols} columns, got {train_df.shape[1]}."
            )

        empty_cols = [c for c in train_df.columns if train_df[c].isna().all()]
        if empty_cols:
            raise ValueError(
                f"Validation failed for subset '{subset}': fully empty columns found in train_df: "
                f"{empty_cols}"
            )

        for required in ("unit_id", "cycle"):
            if required not in train_df.columns:
                raise ValueError(
                    f"Validation failed for subset '{subset}': missing required column '{required}'."
                )

        if rul_df.empty:
            raise ValueError(f"Validation failed for subset '{subset}': rul_df is empty.")

        self.logger.info("Validation passed for subset %s", subset)

    def _read_cmapss_txt(self, path: Path) -> pd.DataFrame:
        """
        Read a C-MAPSS train/test `.txt` file (space separated, no header).

        Drops the trailing 2 NaN columns and assigns `dataset.column_names`.

        Args:
            path: File path to read.

        Returns:
            Parsed dataframe with standardized column names.
        """
        try:
            df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
        except Exception as e:
            raise ValueError(f"Failed to read C-MAPSS file '{path.as_posix()}': {e}") from e

        if df.shape[1] < len(self.column_names):
            raise ValueError(
                f"Unexpected column count in '{path.as_posix()}': got {df.shape[1]}, "
                f"expected at least {len(self.column_names)}."
            )

        # C-MAPSS files include two trailing empty columns; drop them.
        if df.shape[1] >= len(self.column_names) + 2:
            df = df.iloc[:, : len(self.column_names)]
        else:
            # If the file already has the expected number of columns, keep as-is.
            df = df.iloc[:, : len(self.column_names)]

        df.columns = self.column_names
        return df

    def _read_rul_txt(self, path: Path) -> pd.DataFrame:
        """
        Read a C-MAPSS RUL `.txt` file (single column, no header).

        Args:
            path: File path to read.

        Returns:
            Dataframe with a single column named `RUL`.
        """
        try:
            df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
        except Exception as e:
            raise ValueError(f"Failed to read RUL file '{path.as_posix()}': {e}") from e

        if df.shape[1] != 1:
            # Some downloads may include trailing whitespace columns; keep the first.
            df = df.iloc[:, [0]]

        df.columns = ["RUL"]
        return df
