"""
Regression tests for score scaling and operating-threshold calibration:

1. Isolation Forest scores must preserve ranking in the anomaly tail
   (clipping at 1.0 once collapsed all extreme anomalies to the same value,
   making the FPR-constrained threshold degenerate to ">1" -- zero flags).
2. The XGBoost operating threshold must satisfy the line-level FPR budget on
   the SAME fitted model it was selected for (a refit once silently voided
   the guarantee), and the persisted bundle must carry that threshold into
   real-time scoring.

Run with: pytest tests/test_calibration.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
import config
from models.train_isolation_forest import FEATURE_COLUMNS
from models.train_isolation_forest import score as score_iso
from models.train_isolation_forest import train as train_iso
from models.train_xgboost_baseline import fit_and_calibrate
from scoring.score_realtime import load_artifacts, score_user_window


def _typical_rows(n, rng, sender_prefix="legit", label=0):
    """Feature rows shaped like normal human traffic."""
    return pd.DataFrame({
        "sender_id": [f"{sender_prefix}_{i}" for i in range(n)],
        "label_fraud": label,
        "message_count": rng.integers(1, 8, n),
        "unique_recipients": rng.integers(1, 5, n),
        "new_recipient_ratio": rng.uniform(0, 0.2, n),
        "length_mean": rng.uniform(40, 100, n),
        "length_std": rng.uniform(10, 30, n),
        "length_cv": rng.uniform(0.15, 0.5, n),
        "interarrival_mean_sec": rng.uniform(300, 3000, n),
        "interarrival_std_sec": rng.uniform(100, 1500, n),
        "messages_per_recipient": rng.uniform(1.0, 2.5, n),
    })


def _burst_row(sender, message_count):
    """A fraud-burst-shaped row; larger message_count = more extreme."""
    return {
        "sender_id": sender, "label_fraud": 1,
        "message_count": message_count,
        "unique_recipients": message_count,
        "new_recipient_ratio": 1.0,
        "length_mean": 100.0, "length_std": 2.0, "length_cv": 0.02,
        "interarrival_mean_sec": 5.0, "interarrival_std_sec": 2.0,
        "messages_per_recipient": 1.0,
    }


def test_isolation_forest_preserves_anomaly_tail_ranking():
    rng = np.random.default_rng(0)
    train_df = _typical_rows(400, rng)
    model, scaler = train_iso(train_df)

    # A mildly anomalous row: past the training 99th-percentile reference
    # (so its risk exceeds 1) but far less extreme than a full fraud burst.
    # Note the burst itself sits at Isolation Forest's own decision_function
    # floor (maximally isolated), so the discriminating pair is
    # moderate-vs-extreme, both of which the old clip collapsed to 1.0.
    moderate_row = {
        "sender_id": "moderate", "label_fraud": 1,
        "message_count": 10, "unique_recipients": 6,
        "new_recipient_ratio": 0.3,
        "length_mean": 100.0, "length_std": 10.0, "length_cv": 0.1,
        "interarrival_mean_sec": 200.0, "interarrival_std_sec": 100.0,
        "messages_per_recipient": 10 / 6,
    }
    to_score = pd.DataFrame([
        _typical_rows(1, rng).iloc[0].to_dict(),
        moderate_row,
        _burst_row("extreme", 500),
    ])
    risk = score_iso(model, scaler, to_score)

    typical, moderate, extreme = risk.iloc[0], risk.iloc[1], risk.iloc[2]
    # Ranking must survive into the tail: under the old 0-1 clip, moderate
    # and extreme (both past 1.0) saturated at exactly 1.0 and became
    # indistinguishable, which degenerated the FPR threshold to ">1".
    assert typical < moderate < extreme
    assert moderate > 1.0 and extreme > 1.0
    assert (risk >= 0).all()  # but never negative (risk bands assume >= 0)


def _xgb_training_frames(seed=0):
    rng = np.random.default_rng(seed)
    legit = _typical_rows(120, rng)
    fraud = pd.DataFrame([
        _burst_row(f"fraud_{i}", int(c))
        for i, c in enumerate(rng.integers(40, 150, 12))
    ])
    frame = pd.concat([legit, fraud], ignore_index=True)
    senders = list(frame["sender_id"].unique())
    rng.shuffle(senders)
    val_senders = set(senders[: len(senders) // 4])
    val_rows = frame[frame["sender_id"].isin(val_senders)].reset_index(drop=True)
    fit_rows = frame[~frame["sender_id"].isin(val_senders)].reset_index(drop=True)
    # both splits need both classes for a meaningful fit/calibration
    assert fit_rows["label_fraud"].nunique() == 2
    assert val_rows["label_fraud"].nunique() == 2
    return fit_rows, val_rows


def test_operating_threshold_meets_line_fpr_budget_on_same_model():
    fit_rows, val_rows = _xgb_training_frames()
    model, threshold = fit_and_calibrate(fit_rows, val_rows)

    # The guarantee the threshold exists to provide: on the sender-disjoint
    # validation split, scored by THE SAME fitted model, line-level FPR is
    # within the approved budget.
    val_prob = model.predict_proba(val_rows[FEATURE_COLUMNS].fillna(0))[:, 1]
    lines = val_rows[["sender_id", "label_fraud"]].copy()
    lines["score"] = val_prob
    per_line = lines.groupby("sender_id").agg(
        label=("label_fraud", "max"), score=("score", "max"))
    legit = per_line[per_line["label"] == 0]
    line_fpr = float((legit["score"] >= threshold).mean())
    assert line_fpr <= config.TARGET_LINE_FALSE_POSITIVE_RATE


def test_realtime_scorer_applies_bundled_threshold(tmp_path, monkeypatch):
    fit_rows, val_rows = _xgb_training_frames()
    model, threshold = fit_and_calibrate(fit_rows, val_rows)

    joblib.dump(
        {"model": model, "operating_threshold": float(threshold),
         "target_line_fpr": config.TARGET_LINE_FALSE_POSITIVE_RATE,
         "feature_columns": list(FEATURE_COLUMNS)},
        tmp_path / "xgboost_baseline.joblib",
    )
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(config, "DEPLOYMENT_MODEL_KIND", "xgboost")

    artifacts = load_artifacts()
    assert artifacts["operating_threshold"] == float(threshold)

    burst = {k: v for k, v in _burst_row("x", 85).items()
             if k in FEATURE_COLUMNS}
    rng = np.random.default_rng(1)
    normal = {k: v for k, v in _typical_rows(1, rng).iloc[0].to_dict().items()
              if k in FEATURE_COLUMNS}

    burst_out = score_user_window(artifacts, burst)
    normal_out = score_user_window(artifacts, normal)
    # The deployment decision signal must be the bundled validated threshold,
    # not a hardcoded cutoff.
    assert burst_out["exceeds_operating_threshold"] is (burst_out["risk_score"] >= threshold)
    assert normal_out["exceeds_operating_threshold"] is (normal_out["risk_score"] >= threshold)
    assert burst_out["exceeds_operating_threshold"] is True
    assert normal_out["exceeds_operating_threshold"] is False


def test_legacy_bare_model_artifact_reports_no_threshold(tmp_path, monkeypatch):
    fit_rows, val_rows = _xgb_training_frames()
    model, _ = fit_and_calibrate(fit_rows, val_rows)
    joblib.dump(model, tmp_path / "xgboost_baseline.joblib")  # pre-bundle format
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(config, "DEPLOYMENT_MODEL_KIND", "xgboost")

    artifacts = load_artifacts()
    assert artifacts["operating_threshold"] is None
    out = score_user_window(
        artifacts, {k: v for k, v in _burst_row("x", 85).items() if k in FEATURE_COLUMNS})
    # None, not False: "no validated guarantee" must be distinguishable from
    # "below threshold".
    assert out["exceeds_operating_threshold"] is None
