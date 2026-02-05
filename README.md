# Practical Lab 1 — Streaming Predictive Maintenance with Linear Regression Alerts

## Project Summary
This project extends a streaming industrial current pipeline by adding **univariate linear regression models**
(Time → Axis currents #1–#8) and a **residual-based anomaly detection** system to generate **Alerts** and **Errors**
for predictive maintenance.

Training data is pulled from a **cloud PostgreSQL database (Neon.tech)**. Synthetic test data is generated using
training metadata (time step, mean, std), then normalized/standardized with respect to the training distribution.

## Tech Stack
- Python (pandas, numpy, matplotlib)
- PostgreSQL (Neon.tech)
- Regression + residual analysis
- Threshold-based alerting with persistence (continuous time)

---

## Setup Instructions

### 1) Create a virtual environment and install requirements
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Configure Neon database credentials
Create a `.env` file in the repo root:
```
DB_HOST=ep-square-brook-ah6zvg12-pooler.c-3.us-east-1.aws.neon.tech
DB_PORT=5432
DB_NAME=neondb
DB_USER=neondb_owner
DB_PASSWORD=npg_CsRhPogOx95F

RAW_TABLE=robot_currents_raw
EVENTS_TABLE=pm_events

```

### 3) Run the pipeline
```bash
python -m src.run_pipeline
```

Outputs:
- `outputs/models/linreg_models.json`
- `outputs/models/scalers.json`
- `outputs/logs/events.csv`
- figures in `outputs/figures/` (generated from notebook)

---

## Regression + Residual Logic

### Regression Model (per axis)
For each axis (1–8), a univariate linear regression is fit:
**Axis(t) = intercept + slope * time**

### Residuals
Residuals are computed as:
**residual = observed - predicted**

Only **positive residuals** are used for alerts/errors (above regression line).

---

## Threshold Discovery (MinC, MaxC, T)
Thresholds are not fixed. They are derived from training residuals.
A strong approach is to set:
- **MinC** = ~95th percentile of **positive residuals**
- **MaxC** = ~99th percentile of **positive residuals**
- **T** = persistence window (seconds) selected by testing multiple values until false positives decrease
  while sustained deviations remain detectable.

Evidence is shown in `notebooks/01_regression_threshold_discovery.ipynb`:
- regression plots per axis
- residual distribution plots
- quantile tables and event counts for multiple T values

---

## Alert & Error Rules
- **ALERT**: deviation ≥ MinC above regression line for ≥ T seconds continuously
- **ERROR**: deviation ≥ MaxC above regression line for ≥ T seconds continuously

Detected events are:
- logged to `outputs/logs/events.csv`
- stored in Neon table `pm_events`

---

## Repository Contents
- `src/` : database, preprocessing, regression, detection, streaming simulation
- `notebooks/` : threshold discovery and justification
- `data/` : synthetic test csv + (optional) training exports
- `outputs/` : models, logs, figures

---

## Notes
- If your database schema uses different column names than `time_s` and `axis_1..axis_8`, update `src/config.py`.
- Before training, ensure you have inserted/streamed training data into Neon (table `RAW_TABLE`).
