import pytest
import torch
import numpy as np
from src.trainer import LSTMModel, RULTrainer


def test_lstm_forward_pass():
    model = LSTMModel(input_size=14, hidden_size=64, num_layers=2, dropout=0.2)
    x = torch.randn(8, 30, 14)
    out = model(x)
    # output should be (batch,)
    assert out.shape == (8,)


def test_lstm_different_inputs_different_outputs():
    model = LSTMModel(input_size=14, hidden_size=64, num_layers=2, dropout=0.2)
    model.eval()
    x1 = torch.randn(1, 30, 14)
    x2 = torch.randn(1, 30, 14)
    with torch.no_grad():
        out1 = model(x1).item()
        out2 = model(x2).item()
    assert out1 != out2


def test_trainer_builds_model():
    trainer = RULTrainer()
    model = trainer.build_model(input_size=14)
    assert model is not None
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 0


def test_trainer_dataloaders():
    trainer = RULTrainer()
    X = np.random.randn(200, 30, 14).astype("float32")
    y = np.random.rand(200).astype("float32") * 125
    train_loader, val_loader = trainer.get_dataloaders(X, y)
    assert train_loader is not None
    assert val_loader is not None


def test_model_output_shape():
    model = LSTMModel(input_size=14, hidden_size=64, num_layers=2, dropout=0.2)
    x = torch.randn(32, 30, 14)
    out = model(x)
    assert out.shape == (32,)