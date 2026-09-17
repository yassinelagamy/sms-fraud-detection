# SMS Fraud Detection — Rule-Based → ML Migration

## Context
Orange Egypt currently flags SMS fraud with a hard-coded rule: if a user sends
more than N messages of similar length (with some variance) to multiple
numbers within a short time window, the line is shut down automatically.

This repository provides a machine learning baseline for SMS fraud detection.
It includes synthetic data generation, event-time features, model training,
backtesting, and mock real-time scoring. Production integration is still needed.

## What's already here
- `data/generate_synthetic_data.py` — generates fake SMS event logs that mimic
  real telecom CDR/SMS metadata (sender, recipient, timestamp, message length,
  etc.) so you can develop and test without needing production data access.
- `features/feature_engineering.py` — turns raw per-message events into
  trailing, event-time features after every SMS (message count, recipient
  diversity, length entropy, inter-arrival time stats, new-number ratio).
  This is the ML equivalent of your current if-condition logic.
- `models/train_isolation_forest.py` — unsupervised baseline. Scores each
  user/window as an anomaly without needing labeled fraud data.
- `models/train_xgboost_baseline.py` — supervised baseline for once you have
  labeled fraud/legitimate cases (e.g. from past manual reviews).
- `scoring/score_realtime.py` — mock scoring function showing how a live SMS
  gateway would call the trained model per event, producing a risk score
  instead of a binary shutdown decision.
- `tests/test_features.py` — a couple of sanity checks on the feature pipeline.

`backtest/backtest_harness.py` also compares the legacy rule, Isolation Forest,
and XGBoost with sender-disjoint splits and line-level intervention metrics.

## Planned improvements
- Real data ingestion from your CDR/SMS gateway logs (format currently mocked)
- A proper feature store / streaming pipeline (Spark, Flink, or Redis-backed)
- Model serving/API layer for production scoring
- SHAP-based explainability for compliance sign-off on shutdown decisions
- Threshold/tiering logic (soft flag → throttle → manual review → shutdown)
  instead of a single binary cutoff
- CI, logging, config management, and proper packaging

## Quick start
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python data/generate_synthetic_data.py
python features/feature_engineering.py
python models/train_isolation_forest.py
python backtest/backtest_harness.py
```

## Safety defaults

Feature rows use a trailing 15-minute event-time window, preventing a burst
from evading detection merely by crossing a clock boundary. Backtest thresholds
are selected against a line-level false-positive budget, not window-level F1.
Until scores are calibrated with real outcomes, the default policy routes
critical cases to manual review and does not automatically suspend a line.
