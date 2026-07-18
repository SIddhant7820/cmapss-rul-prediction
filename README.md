# 🚀 Remaining Useful Life (RUL) Prediction using Deep Learning

Predicting the **Remaining Useful Life (RUL)** of aircraft turbofan engines using the **NASA C-MAPSS dataset** with an end-to-end machine learning pipeline built in Python.

This project implements a modular workflow for data ingestion, preprocessing, model training, prediction, and evaluation using **LSTM** and **XGBoost** models.

---

## 📌 Project Overview

Predictive Maintenance is one of the most important applications of Artificial Intelligence in manufacturing and aviation.

Instead of waiting for equipment to fail, predictive maintenance estimates how many cycles remain before failure, allowing maintenance to be scheduled proactively.

This project predicts the Remaining Useful Life (RUL) of aircraft engines using sensor measurements collected from the NASA C-MAPSS benchmark dataset.

---

## 🎯 Objectives

- Predict Remaining Useful Life (RUL) of aircraft engines
- Reduce unexpected engine failures
- Compare Deep Learning and Machine Learning approaches
- Build a reusable end-to-end ML pipeline
- Support all four NASA C-MAPSS subsets (FD001–FD004)

---

# 📂 Dataset

Dataset:
NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset

Each engine contains:

- Operational settings
- 21 sensor measurements
- Engine cycles

Training data contains complete degradation until failure.

Test data contains partial engine life, and the goal is to predict the remaining cycles.

Supported subsets:

| Dataset | Operating Conditions | Fault Modes |
|----------|---------------------|-------------|
| FD001 | 1 | 1 |
| FD002 | 6 | 1 |
| FD003 | 1 | 2 |
| FD004 | 6 | 2 |

---

# 🛠 Project Structure

```
cursor_project/

│── app.py
│── main.py
│── config/
│     config.yaml
│
├── data/
│     ├── raw/
│     └── processed/
│
├── models/
│     └── saved/
│
├── logs/
│     ├── plots/
│     └── predictions/
│
├── notebooks/
│
├── src/
│     ingestion.py
│     transformation.py
│     trainer.py
│     predictor.py
│     evaluator.py
│     utils.py
│
├── requirements.txt
└── README.md
```

---

# ⚙️ Features

✔ End-to-End Pipeline

✔ Modular Architecture

✔ YAML Configuration

✔ Automatic Logging

✔ Data Validation

✔ Feature Engineering

✔ LSTM Deep Learning Model

✔ XGBoost Baseline

✔ Prediction Visualization

✔ Evaluation Metrics

✔ Streamlit Application

✔ Supports FD001–FD004

---

# 🧠 Machine Learning Pipeline

```
Raw Dataset
      │
      ▼
Data Ingestion
      │
      ▼
Data Validation
      │
      ▼
RUL Generation
      │
      ▼
Sensor Selection
      │
      ▼
Cluster-based Normalization
      │
      ▼
Sliding Window Sequence Generation
      │
      ▼
LSTM / XGBoost Training
      │
      ▼
Prediction
      │
      ▼
Evaluation
```

---

# 🔍 Feature Engineering

The following preprocessing techniques were applied:

- RUL clipping (Maximum RUL = 125)
- Removal of noisy sensors
- Missing value handling
- Standard Scaling
- Sliding Window Sequence Generation
- Cluster-wise normalization using K-Means
- Train/Test preprocessing consistency

Removed noisy sensors:

```
s1
s5
s6
s10
s16
s18
s19
```

Remaining sensors:

```
14 Sensors
```

Window Size:

```
30 Cycles
```

---

# 🤖 Models

## LSTM

- 2 LSTM Layers
- Hidden Size = 128
- Dropout = 0.2
- Adam Optimizer
- Early Stopping

---

## XGBoost

- Gradient Boosting Trees
- Used as baseline model
- Hyperparameter tuned

---

# 📈 Evaluation Metrics

The project evaluates predictions using:

- RMSE
- MAE
- R² Score
- NASA Scoring Function

---

# 📊 Results

| Dataset | RMSE | MAE | R² |
|----------|------|------|------|
| FD001 | 13.48 | 10.23 | 0.895 |
| FD002 | 26.72 | 17.82 | 0.753 |
| FD003 | 13.26 | 9.34 | 0.897 |
| FD004 | *(Update after training completes)* | *(Update)* | *(Update)* |

---

# 📷 Sample Output

Prediction Plot

```
logs/plots/predictions_FD001.png
```

Prediction CSV

```
logs/predictions_FD001.csv
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Remaining-Useful-Life-Prediction.git
```

Move into the project

```bash
cd Remaining-Useful-Life-Prediction
```

Create environment

```bash
conda create -n rul python=3.11
conda activate rul
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Usage

Train model

```bash
python main.py --subset FD001 --model lstm
```

Predict

```bash
python main.py --subset FD001 --model lstm --skip-train
```

Run Streamlit

```bash
streamlit run app.py
```

---

# 📦 Technologies Used

- Python
- PyTorch
- XGBoost
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Streamlit
- YAML
- Git

---

# 🔮 Future Improvements

- Transformer-based RUL prediction
- Attention-based LSTM
- Hyperparameter Optimization
- Docker Deployment
- REST API
- Cloud Deployment

---

# 👨‍💻 Author

**Siddhant Ghogare**

Artificial Intelligence & Data Science Student

GitHub:
https://github.com/SIddhant7820

LinkedIn:
https://www.linkedin.com/in/siddhant-ghogare

---

# ⭐ If you found this project useful, consider giving it a star!