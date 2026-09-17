"""
Unsupervised anomaly detection baseline using Isolation Forest.

Use this when you don't yet have reliable fraud labels — it scores each
(user, window) by how "unusual" its behavior is relative to the overall
population, without needing to know which historical cases were fraud.

Once you accumulate labeled cases (confirmed fraud / confirmed false
positives from manual review), switch to or combine with
train_xgboost_baseline.py for a supervised model with better precision.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).parent.parent))
import config

FEATURE_COLUMNS = [
    "message_count",
    "unique_recipients",
    "new_recipient_ratio",
    "length_mean",
    "length_std",
    "length_cv",
    "interarrival_mean_sec",
    "interarrival_std_sec",
    "messages_per_recipient",
]


def train(features: pd.DataFrame):
    X = features[FEATURE_COLUMNS].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=config.ISO_FOREST_N_ESTIMATORS,
        contamination=config.ISO_FOREST_CONTAMINATION,
        random_state=config.RANDOM_STATE,
    )
    model.fit(X_scaled)

    # Train-referenced calibration for the 0-1 risk score: store reference
    # quantiles of the TRAINING decision_function so scoring never depends
    # on the composition of the batch being scored (which would make batch
    # and single-row scores inconsistent, and let test-batch statistics
    # influence test scores). Follows sklearn's fitted-attribute convention;
    # joblib persists it with the model.
    raw_train = model.decision_function(X_scaled)
    model.risk_calibration_ = (
        float(np.quantile(raw_train, 0.01)),
        float(np.quantile(raw_train, 0.99)),
    )

    return model, scaler


def score(model, scaler, features: pd.DataFrame) -> pd.Series:
    """
    Returns a non-negative risk score per row, higher = more anomalous.
    Typical traffic lands in ~[0, 1]; values above 1 are more anomalous than
    the training set's 99th-percentile reference.

    Normalized against reference quantiles of the training set's
    decision_function (model.risk_calibration_), so the same feature vector
    always gets the same risk score whether scored in a batch or alone —
    a requirement for consistent real-time scoring and honest evaluation.
    Falls back to a fixed range for artifacts trained before calibration
    existed.

    Only the LOWER end is clipped (a negative score must not fall through
    the risk bands). The upper end is deliberately NOT clipped: capping at 1
    collapses every extreme anomaly to the same value, so per-line max
    scores tie at exactly 1.0 for many legit lines, and an FPR-constrained
    threshold degenerates to ">1" — flagging nothing despite good ranking.
    Downstream, scoring/risk_bands.band_for maps any score at/above the top
    band into "critical", so >1 scores are handled.
    """
    X = features[FEATURE_COLUMNS].fillna(0)
    X_scaled = scaler.transform(X)
    raw = model.decision_function(X_scaled)

    ref_low, ref_high = getattr(model, "risk_calibration_", (-0.2, 0.2))
    risk = np.maximum((ref_high - raw) / (ref_high - ref_low), 0)

    return pd.Series(risk, index=features.index)


def evaluate(features: pd.DataFrame, risk_scores: pd.Series):
    """Only works if label_fraud ground truth is available (e.g. synthetic data)."""
    if "label_fraud" not in features or features["label_fraud"].isna().all():
        print("No ground truth labels available — skipping evaluation.")
        return

    y_true = features["label_fraud"]
    y_pred = (risk_scores >= config.RISK_THRESHOLDS["manual_review"]).astype(int)

    print("AUC:", roc_auc_score(y_true, risk_scores))
    print(classification_report(y_true, y_pred, digits=3))


def main():
    features = pd.read_csv(config.FEATURES_PATH)

    model, scaler = train(features)
    risk_scores = score(model, scaler, features)

    features["risk_score"] = risk_scores
    features["risk_tier"] = pd.cut(
        risk_scores,
        bins=[-np.inf] + list(config.RISK_THRESHOLDS.values()) + [np.inf],
        labels=["normal", "soft_flag", "throttle", "manual_review", "shutdown"],
    )

    evaluate(features, risk_scores)

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODEL_DIR / "isolation_forest.joblib")
    joblib.dump(scaler, config.MODEL_DIR / "scaler.joblib")

    out_path = config.DATA_DIR / "scored_features.csv"
    features.to_csv(out_path, index=False)
    print(f"Saved scored features -> {out_path}")
    print(features["risk_tier"].value_counts())


if __name__ == "__main__":
    main()
