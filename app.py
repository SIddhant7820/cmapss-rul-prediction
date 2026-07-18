"""
Interactive demo for the NASA C-MAPSS Remaining Useful Life (RUL) predictor.
Run locally with:  streamlit run app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.evaluator import RULEvaluator
from src.predictor import RULPredictor
from src.utils import load_config

st.set_page_config(page_title="C-MAPSS RUL Predictor", page_icon="tools", layout="wide")

CONFIG_PATH = "config/config.yaml"
config = load_config(CONFIG_PATH)
PROCESSED = Path("data/processed")
PLOTS_DIR = Path(config["paths"]["plots_path"])
MODEL_DIR = Path(config["paths"]["model_save_path"])

st.title("Turbofan Engine — Remaining Useful Life (RUL) Predictor")
st.caption(
    "NASA C-MAPSS dataset - LSTM & XGBoost models - trained on run-to-failure "
    "sensor sequences from simulated turbofan engines."
)

st.sidebar.header("Configuration")
subset = st.sidebar.selectbox(
    "Dataset subset",
    ["FD001", "FD002", "FD003", "FD004"],
    help="FD001/FD003: single operating condition. FD002/FD004: six operating "
         "conditions (harder). FD003/FD004 also include two fault modes.",
)
model_type = st.sidebar.radio("Model", ["lstm", "xgboost"], horizontal=True)

lstm_path = MODEL_DIR / f"lstm_{subset}.pt"
xgb_path = MODEL_DIR / f"xgb_{subset}.pkl"
model_path = lstm_path if model_type == "lstm" else xgb_path

if not model_path.exists() or model_path.stat().st_size == 0:
    st.error(
        f"No valid trained {model_type.upper()} checkpoint found for {subset} "
        f"(expected `{model_path}`, found "
        f"{'0-byte file' if model_path.exists() else 'nothing'}). Train it first with:\n\n"
        f"`python main.py --subset {subset} --model {model_type} "
        f"--config config/config_fast.yaml`"
    )
    st.stop()


@st.cache_resource
def get_predictor():
    return RULPredictor(CONFIG_PATH)


@st.cache_data
def load_test_data(subset: str):
    X_test = np.load(PROCESSED / f"X_test_{subset}.npy")
    y_test = np.load(PROCESSED / f"y_test_{subset}.npy")
    return X_test, y_test


@st.cache_data
def run_predictions(subset: str, model_type: str):
    predictor = get_predictor()
    X_test, y_test = load_test_data(subset)
    y_pred = predictor.predict(X_test, subset, model_type=model_type)
    return y_test, y_pred


y_test, y_pred = run_predictions(subset, model_type)

evaluator = RULEvaluator(CONFIG_PATH)
metrics = evaluator.evaluate(y_test, y_pred, subset)

c1, c2, c3, c4 = st.columns(4)
c1.metric("RMSE (cycles)", f"{metrics['rmse']:.2f}")
c2.metric("MAE (cycles)", f"{metrics['mae']:.2f}")
c3.metric("R2", f"{metrics['r2']:.3f}")
c4.metric("NASA Score", f"{metrics['nasa_score']:.1f}", help=(
    "Asymmetric scoring function from the NASA C-MAPSS benchmark - "
    "penalizes late predictions (missed failures) far more than early ones."
))

st.divider()

left, right = st.columns([2, 1])

with left:
    st.subheader("Predicted vs Actual RUL - all test engines")
    df = pd.DataFrame({
        "engine": np.arange(1, len(y_test) + 1),
        "actual_RUL": y_test,
        "predicted_RUL": y_pred,
    })
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["engine"], y=df["actual_RUL"],
                              mode="lines+markers", name="Actual RUL",
                              line=dict(color="#2563eb")))
    fig.add_trace(go.Scatter(x=df["engine"], y=df["predicted_RUL"],
                              mode="lines+markers", name="Predicted RUL",
                              line=dict(color="#f97316")))
    fig.update_layout(xaxis_title="Test engine unit", yaxis_title="RUL (cycles)",
                       height=420, margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Pick one engine")
    engine_id = st.selectbox("Engine unit", df["engine"].tolist())
    row = df[df["engine"] == engine_id].iloc[0]
    st.metric("Actual RUL", f"{row['actual_RUL']:.0f} cycles")
    st.metric("Predicted RUL", f"{row['predicted_RUL']:.0f} cycles",
              delta=f"{row['predicted_RUL'] - row['actual_RUL']:+.1f}")
    if row["predicted_RUL"] < row["actual_RUL"]:
        st.info("Conservative prediction - flags failure earlier than it occurs.")
    else:
        st.warning("Optimistic prediction - flags failure later than it occurs.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Prediction error distribution")
    err = df["predicted_RUL"] - df["actual_RUL"]
    fig2 = px.histogram(err, nbins=30, labels={"value": "Error (predicted - actual)"})
    fig2.update_layout(showlegend=False, height=350, margin=dict(t=10))
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    st.subheader("LSTM training curve")
    loss_plot = PLOTS_DIR / f"loss_{subset}.png"
    if model_type == "lstm" and loss_plot.exists():
        st.image(str(loss_plot), use_container_width=True)
    else:
        st.caption("No training curve to show (XGBoost selected, or plot not generated yet).")

st.divider()
st.caption(
    "Built on the NASA C-MAPSS turbofan degradation simulation dataset. "
    "Source: A. Saxena and K. Goebel, NASA Ames Prognostics Data Repository."
)