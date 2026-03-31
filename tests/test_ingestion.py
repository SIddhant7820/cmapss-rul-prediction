import pytest
import pandas as pd
from src.ingestion import CMAPSSIngestion


ing = CMAPSSIngestion()


def test_load_subset_returns_dict():
    data = ing.load_subset("FD001")
    assert isinstance(data, dict)
    assert "train_df" in data
    assert "test_df" in data
    assert "rul_df" in data


def test_column_count():
    data = ing.load_subset("FD001")
    # 26 columns: unit_id, cycle, op1-op3, s1-s21
    assert data["train_df"].shape[1] == 26
    assert data["test_df"].shape[1] == 26


def test_no_fully_empty_columns():
    data = ing.load_subset("FD001")
    assert not data["train_df"].isnull().all().any()
    assert not data["test_df"].isnull().all().any()


def test_unit_id_and_cycle_exist():
    data = ing.load_subset("FD001")
    assert "unit_id" in data["train_df"].columns
    assert "cycle" in data["train_df"].columns


def test_rul_df_not_empty():
    data = ing.load_subset("FD001")
    assert len(data["rul_df"]) > 0


def test_train_df_has_rows():
    data = ing.load_subset("FD001")
    assert data["train_df"].shape[0] > 0


def test_all_subsets_load():
    for subset in ["FD001", "FD002", "FD003", "FD004"]:
        data = ing.load_subset(subset)
        assert data["train_df"].shape[0] > 0