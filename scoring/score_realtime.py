"""
Mock real-time scoring entry point.

This is what your SMS gateway would call per-event (or per-window on a
timer) in production, instead of evaluating the current if-condition rule.

Real implementation would need:
  - A feature store maintaining rolling per-user stats in near-real-time
    (e.g. Redis, or a Flink/Spark Streaming job) instead of recomputing
    from a full CSV each time.
  - The trained model + scaler loaded once at service startup, not per call.
  - Proper logging/audit trail for every decision (required for compliance
    review of any shutdown action).

This file only shows the shape of the integration, using the batch
artifacts produced by train_isolation_forest.py for demonstration.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
import config
from models.train_isolation_forest import FEATURE_COLUMNS, score as score_isolation_forest
from scoring.risk_bands import band_for


def load_artifacts() -> dict:
    """
    Load the one model approved by the deployment policy, as a dict:
      kind                : config.DEPLOYMENT_MODEL_KIND at load time
      model, scaler       : fitted artifacts (scaler only for isolation_forest)
      operating_threshold : validation-calibrated line-FPR threshold, or None
                            if the artifact predates threshold bundling
                            (deployment then has NO validated FPR guarantee)
    """
    if config.DEPLOYMENT_MODEL_KIND == "isolation_forest":
        model = joblib.load(config.MODEL_DIR / "isolation_forest.joblib")
        scaler = joblib.load(config.MODEL_DIR / "scaler.joblib")
        return {"kind": "isolation_forest", "model": model, "scaler": scaler,
                "operating_threshold": None}
    if config.DEPLOYMENT_MODEL_KIND == "xgboost":
        artifact = joblib.load(config.MODEL_DIR / "xgboost_baseline.joblib")
        if isinstance(artifact, dict):  # bundle: model + calibrated threshold
            return {"kind": "xgboost", "model": artifact["model"], "scaler": None,
                    "operating_threshold": artifact["operating_threshold"]}
        # Legacy bare-model artifact (pre-bundling): usable for scoring, but
        # carries no validated operating threshold.
        return {"kind": "xgboost", "model": artifact, "scaler": None,
                "operating_threshold": None}
    raise ValueError(f"Unknown deployment model: {config.DEPLOYMENT_MODEL_KIND}")


def score_user_window(artifacts: dict, feature_row: dict) -> dict:
    """
    feature_row: dict with keys matching FEATURE_COLUMNS, computed by
    feature_engineering.py for a single (user, window).

    The returned risk_score is RELATIVE (higher = riskier), not a calibrated
    probability -- band/action mapping lives in scoring/risk_bands.py, the
    single score->action chokepoint where probability calibration will plug
    in once real labeled data exists.

    exceeds_operating_threshold is the validated decision signal: True means
    this window's score crosses the threshold calibrated (on sender-disjoint
    validation data, for this exact fitted model) to the approved line-level
    FPR budget. None means the loaded artifact carries no such threshold and
    deployment decisions have no validated FPR guarantee.
    """
    df = pd.DataFrame([feature_row])[FEATURE_COLUMNS]
    if artifacts["kind"] == "isolation_forest":
        risk = float(score_isolation_forest(artifacts["model"], artifacts["scaler"], df).iloc[0])
    elif artifacts["kind"] == "xgboost":
        risk = float(artifacts["model"].predict_proba(df)[:, 1][0])
    else:
        raise ValueError(f"Unknown deployment model: {artifacts['kind']}")
    if not np.isfinite(risk):
        raise ValueError("Model returned a non-finite risk score")

    threshold = artifacts.get("operating_threshold")
    exceeds = bool(risk >= threshold) if threshold is not None else None
    return {"risk_score": risk, "exceeds_operating_threshold": exceeds, **band_for(risk)}


if __name__ == "__main__":
    artifacts = load_artifacts()

    # Example: a burst-like fraud pattern vs. a normal pattern
    example_fraud_like = {
        "message_count": 85,
        "unique_recipients": 80,
        "new_recipient_ratio": 0.95,
        "length_mean": 100,
        "length_std": 2.0,
        "length_cv": 0.02,
        "interarrival_mean_sec": 8,
        "interarrival_std_sec": 3,
        "messages_per_recipient": 1.06,
    }
    example_normal = {
        "message_count": 6,
        "unique_recipients": 4,
        "new_recipient_ratio": 0.1,
        "length_mean": 75,
        "length_std": 20,
        "length_cv": 0.27,
        "interarrival_mean_sec": 900,
        "interarrival_std_sec": 500,
        "messages_per_recipient": 1.5,
    }

    print("Fraud-like pattern:", score_user_window(artifacts, example_fraud_like))
    print("Normal pattern:    ", score_user_window(artifacts, example_normal))
