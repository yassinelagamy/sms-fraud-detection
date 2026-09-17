"""
Supervised fraud classifier using XGBoost.

Use this once you have a reasonable number of labeled cases (confirmed fraud
shutdowns + confirmed false positives from manual review). Compared to the
Isolation Forest baseline, this gives you:
  - feature importances you can show to compliance/ops as justification
  - generally better precision once labels are available
  - a natural place to add SHAP explainability (see bottom of file)

This script trains on the synthetic labels for demonstration. Replace
label_fraud with your real historical labels when available.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).parent.parent))
import config
from models.train_isolation_forest import FEATURE_COLUMNS
from scoring.thresholds import select_threshold_at_line_fpr

XGB_ARTIFACT_PATH_NAME = "xgboost_baseline.joblib"


def train(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        n_estimators=config.XGB_N_ESTIMATORS,
        max_depth=config.XGB_MAX_DEPTH,
        learning_rate=config.XGB_LEARNING_RATE,
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
        random_state=config.RANDOM_STATE,
        eval_metric="aucpr",
        tree_method="hist",
    )
    model.fit(X_train, y_train)
    return model


def _split_senders(rows: pd.DataFrame, holdout_size: float, seed: int):
    """Sender-disjoint, fraud-stratified split of feature rows."""
    sender_labels = rows.groupby("sender_id")["label_fraud"].max()
    kept, held_out = train_test_split(
        sender_labels.index,
        test_size=holdout_size,
        stratify=sender_labels.to_numpy(),
        random_state=seed,
    )
    return (
        rows[rows["sender_id"].isin(kept)],
        rows[rows["sender_id"].isin(held_out)],
    )


def fit_and_calibrate(fit_rows: pd.DataFrame, val_rows: pd.DataFrame):
    """
    Fit on fit_rows, then select the line-level-FPR operating threshold from
    THIS model's scores on the sender-disjoint val_rows. Returns
    (model, threshold). The threshold is only valid for this exact fitted
    model — never refit and reuse it (score scales differ across fits).
    """
    model = train(fit_rows[FEATURE_COLUMNS].fillna(0), fit_rows["label_fraud"].astype(int))
    val_prob = model.predict_proba(val_rows[FEATURE_COLUMNS].fillna(0))[:, 1]
    threshold = select_threshold_at_line_fpr(val_rows, val_prob)
    return model, threshold


def main():
    features = pd.read_csv(config.FEATURES_PATH)
    features = features.dropna(subset=["label_fraud"])

    # Split by sender, not feature row.  Windows from one line are highly
    # correlated; putting them in train and test would leak a line's behavior
    # into its own evaluation set.
    train_rows, test_rows = _split_senders(features, 0.25, config.RANDOM_STATE)
    # Threshold-calibration carve-out OF THE TRAIN SPLIT: the deployed
    # operating threshold must come from senders the model never fit on, and
    # must be applied to the same fitted model it was selected for.
    fit_rows, val_rows = _split_senders(
        train_rows, config.BACKTEST_VAL_SIZE, config.RANDOM_STATE + 1000)

    model, threshold = fit_and_calibrate(fit_rows, val_rows)

    X_test = test_rows[FEATURE_COLUMNS].fillna(0)
    y_test = test_rows["label_fraud"].astype(int)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    print("AUC:", roc_auc_score(y_test, y_prob))
    print(f"Operating threshold (target line FPR "
          f"{config.TARGET_LINE_FALSE_POSITIVE_RATE}): {threshold:.6f}")
    print(classification_report(y_test, y_pred, digits=3))

    per_line = test_rows[["sender_id", "label_fraud"]].copy()
    per_line["pred"] = y_pred
    lines = per_line.groupby("sender_id").max()
    fraud = lines[lines["label_fraud"] == 1]
    legit = lines[lines["label_fraud"] == 0]
    print(f"Held-out line-level recall: {fraud['pred'].mean():.3f} "
          f"({len(fraud)} fraud lines), line-level FPR: {legit['pred'].mean():.4f} "
          f"({int(legit['pred'].sum())}/{len(legit)} legit lines flagged)")

    print("\nFeature importances:")
    for feat, imp in sorted(
        zip(FEATURE_COLUMNS, model.feature_importances_),
        key=lambda x: -x[1],
    ):
        print(f"  {feat:25s} {imp:.3f}")

    # Persist model + operating threshold as ONE artifact so deployment can
    # never pair the threshold with a different fit than the one it was
    # calibrated for.
    bundle = {
        "model": model,
        "operating_threshold": float(threshold),
        "target_line_fpr": config.TARGET_LINE_FALSE_POSITIVE_RATE,
        "feature_columns": list(FEATURE_COLUMNS),
    }
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, config.MODEL_DIR / XGB_ARTIFACT_PATH_NAME)
    print(f"\nSaved model bundle -> {config.MODEL_DIR / XGB_ARTIFACT_PATH_NAME}")

    # --- TODO: add SHAP explainability ---
    # import shap
    # explainer = shap.TreeExplainer(model)
    # shap_values = explainer.shap_values(X_test)
    # shap.summary_plot(shap_values, X_test, feature_names=FEATURE_COLUMNS)


if __name__ == "__main__":
    main()
