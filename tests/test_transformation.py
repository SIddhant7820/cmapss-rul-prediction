import pytest
import numpy as np
from src.ingestion import CMAPSSIngestion
from src.transformation import CMAPSSTransformer


ing = CMAPSSIngestion()
data = ing.load_subset("FD001")
trans = CMAPSSTransformer()


def test_compute_rul_adds_column():
    df = trans.compute_rul(data["train_df"])
    assert "RUL" in df.columns


def test_rul_capped_at_max():
    df = trans.compute_rul(data["train_df"])
    assert df["RUL"].max() <= trans.max_rul


def test_rul_non_negative():
    df = trans.compute_rul(data["train_df"])
    assert df["RUL"].min() >= 0


def test_drop_sensors_removes_columns():
    df = trans.compute_rul(data["train_df"])
    df = trans.drop_low_variance_sensors(df)
    for sensor in trans.drop_sensors:
        assert sensor not in df.columns


def test_sequence_shape():
    X_train, y_train, X_test, y_test = trans.run(
        data["train_df"], data["test_df"], data["rul_df"], "FD001"
    )
    # X should be 3D
    assert X_train.ndim == 3
    assert X_test.ndim == 3
    # y should be 1D
    assert y_train.ndim == 1
    assert y_test.ndim == 1


def test_window_size_correct():
    X_train, y_train, X_test, y_test = trans.run(
        data["train_df"], data["test_df"], data["rul_df"], "FD001"
    )
    assert X_train.shape[1] == trans.window_size_default
    assert X_test.shape[1] == trans.window_size_default


def test_no_nan_in_sequences():
    X_train, y_train, X_test, y_test = trans.run(
        data["train_df"], data["test_df"], data["rul_df"], "FD001"
    )
    assert not np.isnan(X_train).any()
    assert not np.isnan(y_train).any()