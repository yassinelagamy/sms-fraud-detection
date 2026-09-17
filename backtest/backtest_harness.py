"""
Multi-seed backtest comparing the legacy if-condition rule against the ML
baselines (Isolation Forest, XGBoost) on independently regenerated synthetic
populations.

Decision units reported:
  window-level : per (sender, window) row -- how the models are trained/scored
  line-level   : per sender -- the production decision unit (lines get shut
                 down), including time-to-detection (median messages a fraud
                 line gets through before its first flagged window closes)

Evaluation discipline:
  - sender-disjoint train/test split, stratified on the sender-level label
    (asserted at runtime; no line appears in both sides)
  - ML decision thresholds are selected on a sender-disjoint validation
    carve-out of the TRAINING split to meet the line-level FPR budget, never
    on test
  - every metric is aggregated over config.BACKTEST_SEEDS independent
    populations and reported as mean +/- std -- single-seed numbers on a few
    dozen fraud lines are anecdotes, not estimates
  - per-persona breakdown joins the simulator diagnostics sidecar
    (data/_sim_metadata.csv), which is NEVER visible to features or models

Run with: python backtest/backtest_harness.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).parent.parent))
import config
from data.generate_synthetic_data import generate_dataset
from features.feature_engineering import compute_window_features
from models.train_isolation_forest import FEATURE_COLUMNS
from models.train_isolation_forest import score as score_iso
from models.train_isolation_forest import train as train_iso
from models.train_xgboost_baseline import train as train_xgb
from scoring.risk_bands import band_for
from scoring.thresholds import select_threshold_at_line_fpr

# Backward-compatible alias; the implementation moved to scoring/thresholds.py
# so the deployment training script calibrates with the identical logic.
_select_threshold_at_line_fpr = select_threshold_at_line_fpr


def apply_legacy_rule(features: pd.DataFrame) -> pd.Series:
    """Binary flag per (user, window) using the current production if-condition."""
    if config.OLD_RULE_MESSAGE_COMPARATOR == ">":
        message_threshold = features["message_count"] > config.OLD_RULE_MIN_MESSAGES
    elif config.OLD_RULE_MESSAGE_COMPARATOR == ">=":
        message_threshold = features["message_count"] >= config.OLD_RULE_MIN_MESSAGES
    else:
        raise ValueError("OLD_RULE_MESSAGE_COMPARATOR must be '>' or '>='")

    flagged = (
        message_threshold
        & (features["length_cv"] <= config.OLD_RULE_LENGTH_CV_MAX)
        & (features["unique_recipients"] >= config.OLD_RULE_MIN_RECIPIENTS)
    )
    return flagged.astype(int)


def compute_metrics(name: str, y_true: pd.Series, y_pred, y_score=None) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    row = {
        "method": name,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "false_positive_rate": fpr,
        "flagged": int(tp + fp),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }
    if y_score is not None and pd.Series(y_true).nunique() > 1:
        row["auc"] = roc_auc_score(y_true, y_score)
    else:
        row["auc"] = np.nan
    return row


def _split_by_sender(features: pd.DataFrame, test_size: float, seed: int):
    """
    Sender-disjoint, fraud-stratified split (see Step 1 rationale: window-level
    splits leak per-line signatures a production system never sees).
    """
    sender_labels = features.groupby("sender_id")["label_fraud"].max()
    train_senders, test_senders = train_test_split(
        sender_labels.index, test_size=test_size,
        stratify=sender_labels.values, random_state=seed,
    )
    train_df = features[features["sender_id"].isin(train_senders)].reset_index(drop=True)
    test_df = features[features["sender_id"].isin(test_senders)].reset_index(drop=True)

    overlap = set(train_df["sender_id"]) & set(test_df["sender_id"])
    assert not overlap, f"Sender leakage: {len(overlap)} sender(s) in both splits"
    return train_df, test_df


def _time_to_detection(test_df: pd.DataFrame, preds, events: pd.DataFrame) -> dict:
    """
    Per caught fraud line: how many of its messages were already sent by the
    time its first flagged window CLOSED (decision available at window end) --
    i.e. the damage that got through before any action could trigger.
    """
    df = test_df[["sender_id", "window_start", "label_fraud"]].copy()
    df["pred"] = np.asarray(preds)
    flagged = df[(df["label_fraud"] == 1) & (df["pred"] == 1)]
    if flagged.empty:
        return {}

    decision_time_col = "window_end" if "window_end" in flagged else "window_start"
    first_flag = flagged.groupby("sender_id")[decision_time_col].min()
    ttd = {}
    for sender, decision_time in first_flag.items():
        # Sliding feature rows are scored when the final event in their
        # trailing window arrives, not when a clock-aligned bucket closes.
        cutoff = pd.Timestamp(decision_time)
        if decision_time_col == "window_start":
            cutoff += pd.Timedelta(minutes=config.WINDOW_MINUTES)
        sender_events = events[events["sender_id"] == sender]
        ttd[sender] = int((sender_events["timestamp"] < cutoff).sum())
    return ttd


def _feature_overlap(features: pd.DataFrame) -> dict:
    """
    Step 4 certification diagnostic: fraction of fraud windows whose feature
    value falls inside the legit population's [p1, p99] band, per feature.
    High overlap = the feature alone cannot separate; low = strong signal.
    """
    legit = features[features["label_fraud"] == 0]
    fraud = features[features["label_fraud"] == 1]
    out = {}
    for col in FEATURE_COLUMNS:
        lo, hi = legit[col].quantile([0.01, 0.99])
        out[col] = float(((fraud[col] >= lo) & (fraud[col] <= hi)).mean())
    return out


def run_backtest(seed: int) -> dict:
    """One full cycle: regenerate population -> features -> split -> train ->
    calibrate thresholds train-side -> evaluate on held-out senders."""
    events, metadata = generate_dataset(seed)
    events["timestamp"] = pd.to_datetime(events["timestamp"])

    features = compute_window_features(events)
    features = features.dropna(subset=["label_fraud"]).reset_index(drop=True)

    train_df, test_df = _split_by_sender(features, config.BACKTEST_TEST_SIZE, seed)
    # Threshold-calibration carve-out: sender-disjoint subset OF THE TRAIN SPLIT.
    fit_df, val_df = _split_by_sender(train_df, config.BACKTEST_VAL_SIZE, seed + 1000)

    y_test = test_df["label_fraud"].astype(int)
    persona_map = metadata[metadata["group"] == "fraud"].set_index("sender_id")["persona"]

    methods = {}

    # 1. Legacy rule: fixed production thresholds, nothing to calibrate.
    methods["legacy_rule"] = {
        "pred": apply_legacy_rule(test_df).to_numpy(), "score": None, "threshold": np.nan,
    }

    # 2. Isolation Forest: unsupervised fit on the full train split (labels
    # unused in fitting); operating threshold is selected at the approved
    # line-level FPR on validation, never on test.
    iso_model, iso_scaler = train_iso(train_df)
    iso_thr = _select_threshold_at_line_fpr(val_df, score_iso(iso_model, iso_scaler, val_df))
    iso_risk_test = score_iso(iso_model, iso_scaler, test_df).to_numpy()
    methods["isolation_forest"] = {
        "pred": (iso_risk_test >= iso_thr).astype(int), "score": iso_risk_test, "threshold": iso_thr,
    }

    # 3. XGBoost: fit on fit_df, choose a line-level-FPR threshold from
    # validation predictions, and score test with THAT SAME fitted model.
    # A threshold is only meaningful on the score scale it was selected on:
    # refitting on the full train split (fit_df + val_df) produces a model
    # whose probability scale can differ, which silently voids the validation
    # FPR guarantee. Forgoing the validation senders in the final fit is the
    # price of a threshold that actually means something.
    xgb_model = train_xgb(fit_df[FEATURE_COLUMNS].fillna(0), fit_df["label_fraud"].astype(int))
    val_prob = xgb_model.predict_proba(val_df[FEATURE_COLUMNS].fillna(0))[:, 1]
    xgb_thr = _select_threshold_at_line_fpr(val_df, val_prob)
    xgb_prob_test = xgb_model.predict_proba(test_df[FEATURE_COLUMNS].fillna(0))[:, 1]
    methods["xgboost"] = {
        "pred": (xgb_prob_test >= xgb_thr).astype(int), "score": xgb_prob_test, "threshold": xgb_thr,
    }

    window_rows, line_rows, persona_rows, tier_rows = [], [], [], []
    for name, m in methods.items():
        w = compute_metrics(name, y_test, m["pred"], m["score"])
        w.update({"seed": seed, "threshold": m["threshold"]})
        window_rows.append(w)

        line_df = test_df[["sender_id", "label_fraud"]].copy()
        line_df["pred"] = m["pred"]
        per_line = line_df.groupby("sender_id").max()
        fraud_lines = per_line[per_line["label_fraud"] == 1]
        legit_lines = per_line[per_line["label_fraud"] == 0]
        ttd = _time_to_detection(test_df, m["pred"], events)
        line_rows.append({
            "method": name, "seed": seed,
            "line_recall": fraud_lines["pred"].mean() if len(fraud_lines) else np.nan,
            "line_fpr": legit_lines["pred"].mean() if len(legit_lines) else np.nan,
            "fraud_lines_in_test": len(fraud_lines),
            "legit_lines_flagged": int(legit_lines["pred"].sum()),
            "median_msgs_before_detection": float(np.median(list(ttd.values()))) if ttd else np.nan,
        })

        caught = fraud_lines.copy()
        caught["persona"] = caught.index.map(persona_map)
        for persona, grp in caught.groupby("persona"):
            grp_ttd = [ttd[s] for s in grp.index if s in ttd]
            persona_rows.append({
                "method": name, "seed": seed, "persona": persona,
                "lines": len(grp), "line_recall": grp["pred"].mean(),
                "median_msgs_before_detection": float(np.median(grp_ttd)) if grp_ttd else np.nan,
            })

        if m["score"] is not None:
            for tier, thr in config.RISK_THRESHOLDS.items():
                t = compute_metrics(name, y_test, (m["score"] >= thr).astype(int))
                tier_rows.append({
                    "method": name, "tier": tier, "tier_threshold": thr, "seed": seed,
                    "precision": t["precision"], "recall": t["recall"],
                    "false_positive_rate": t["false_positive_rate"], "flagged": t["flagged"],
                })

    # Risk-band composition for the same model selected for real-time scoring:
    # how many
    # test lines land in each operational band, and how many of them are
    # actually fraud. These are OBSERVED COUNTS on synthetic data -- volume
    # sizing for ops queues, not probability claims (scores are relative;
    # calibration is deferred until real labeled data exists).
    band_rows = []
    deployment_method = config.DEPLOYMENT_MODEL_KIND
    if deployment_method not in methods:
        raise ValueError(f"Unknown deployment model: {deployment_method}")
    line_scores = test_df[["sender_id", "label_fraud"]].copy()
    line_scores["score"] = methods[deployment_method]["score"]
    per_line = line_scores.groupby("sender_id").agg(
        label=("label_fraud", "max"), line_score=("score", "max"),
    )
    per_line["band"] = per_line["line_score"].map(lambda s: band_for(s)["band"])
    for band_cfg in config.RISK_BANDS:
        grp = per_line[per_line["band"] == band_cfg["band"]]
        band_rows.append({
            "seed": seed, "method": deployment_method,
            "band": band_cfg["band"], "action": band_cfg["action"],
            "n_lines": len(grp), "n_fraud_lines": int(grp["label"].sum()),
        })

    return {
        "window": window_rows, "line": line_rows, "persona": persona_rows,
        "tiers": tier_rows, "bands": band_rows, "overlap": _feature_overlap(features),
        "n_test_fraud_windows": int(y_test.sum()),
    }


def main():
    seeds = config.BACKTEST_SEEDS
    window_rows, line_rows, persona_rows, tier_rows, band_rows, overlap_rows = [], [], [], [], [], []

    for seed in seeds:
        print(f"--- seed {seed} ---")
        result = run_backtest(seed)
        window_rows.extend(result["window"])
        line_rows.extend(result["line"])
        persona_rows.extend(result["persona"])
        tier_rows.extend(result["tiers"])
        band_rows.extend(result["bands"])
        overlap_rows.append(result["overlap"])
        print(f"    test fraud windows: {result['n_test_fraud_windows']}")

    window_df = pd.DataFrame(window_rows)
    line_df = pd.DataFrame(line_rows)
    persona_df = pd.DataFrame(persona_rows)
    tier_df = pd.DataFrame(tier_rows)

    fmt = lambda x: f"{x:.3f}"

    print(f"\n===== WINDOW-LEVEL (mean +/- std over {len(seeds)} seeds) =====")
    w_agg = window_df.groupby("method")[
        ["precision", "recall", "false_positive_rate", "auc", "threshold"]
    ].agg(["mean", "std"]).round(3)
    print(w_agg.to_string())

    print("\n===== LINE-LEVEL (production decision unit) =====")
    l_agg = line_df.groupby("method")[
        ["line_recall", "line_fpr", "legit_lines_flagged", "median_msgs_before_detection"]
    ].agg(["mean", "std"]).round(3)
    print(l_agg.to_string())

    print("\n===== PER-PERSONA LINE RECALL / TIME-TO-DETECTION =====")
    p_agg = persona_df.groupby(["method", "persona"])[
        ["line_recall", "median_msgs_before_detection"]
    ].mean().round(3)
    print(p_agg.to_string())

    print("\n===== RISK-TIER SWEEP (test metrics at each config.RISK_THRESHOLDS cutoff) =====")
    t_agg = tier_df.groupby(["method", "tier"])[
        ["precision", "recall", "false_positive_rate", "flagged"]
    ].mean().round(3)
    print(t_agg.to_string())

    print("\n===== RISK-BAND COMPOSITION (deployment-model line-level max score; observed counts, scores are RELATIVE, not probabilities) =====")
    band_df = pd.DataFrame(band_rows)
    band_order = [b["band"] for b in config.RISK_BANDS]
    b_agg = band_df.groupby(["band", "action"]).agg(
        lines_per_seed=("n_lines", "mean"),
        fraud_lines_total=("n_fraud_lines", "sum"),
        lines_total=("n_lines", "sum"),
    )
    b_agg["lines_per_seed"] = b_agg["lines_per_seed"].round(1)
    b_agg["observed_fraud_rate"] = (b_agg["fraud_lines_total"] / b_agg["lines_total"]).round(3)
    b_agg = b_agg.reindex([  # preserve escalation order
        idx for band in band_order for idx in b_agg.index if idx[0] == band
    ])
    print(b_agg.to_string())

    print("\n===== FEATURE OVERLAP (fraction of fraud windows inside legit p1-p99, mean over seeds) =====")
    overlap_df = pd.DataFrame(overlap_rows)
    print(overlap_df.mean().round(3).sort_values().to_string())

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    window_df.to_csv(config.DATA_DIR / "backtest_report.csv", index=False)
    line_df.to_csv(config.DATA_DIR / "backtest_line_level.csv", index=False)
    persona_df.to_csv(config.DATA_DIR / "backtest_per_persona.csv", index=False)
    tier_df.to_csv(config.DATA_DIR / "backtest_tier_sweep.csv", index=False)
    band_df.to_csv(config.DATA_DIR / "backtest_risk_bands.csv", index=False)
    print(f"\nSaved per-seed reports -> {config.DATA_DIR / 'backtest_*.csv'}")


if __name__ == "__main__":
    main()
