"""
Operating-threshold selection shared by the backtest harness and the
deployment training script.

A decision threshold is only meaningful on the score scale of the exact
fitted model that produced the calibration scores — selecting a threshold
from one model's validation scores and applying it to a refit model silently
voids the FPR guarantee. Both callers therefore follow the same contract:
fit a model, score a sender-disjoint validation set with THAT model, select
the threshold here, and deploy/evaluate the same model with that threshold.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
import config


def select_threshold_at_line_fpr(
    validation_df: pd.DataFrame, scores, target_fpr: float | None = None,
) -> float:
    """
    Choose the most sensitive validation threshold that respects the approved
    false-positive budget at the production decision unit: one sender/line.

    validation_df needs sender_id and label_fraud columns; scores are the
    window-level scores (same row order) from the model being calibrated.
    """
    target_fpr = config.TARGET_LINE_FALSE_POSITIVE_RATE if target_fpr is None else target_fpr
    if not 0 <= target_fpr < 1:
        raise ValueError("target_fpr must be in [0, 1)")

    validation = validation_df[["sender_id", "label_fraud"]].copy()
    validation["score"] = np.asarray(scores)
    lines = validation.groupby("sender_id").agg(
        label_fraud=("label_fraud", "max"), score=("score", "max"),
    )
    legit_scores = lines.loc[lines["label_fraud"] == 0, "score"].to_numpy()
    if legit_scores.size == 0:
        return 0.5

    # Include a threshold above every score so a zero-FPR policy remains
    # achievable even when the validation sample is small.
    candidates = np.r_[np.nextafter(lines["score"].max(), np.inf), np.unique(lines["score"])]
    best_threshold = float(candidates[0])
    best_recall = -1.0
    for threshold in candidates:
        predicted = lines["score"] >= threshold
        line_fpr = float(predicted[lines["label_fraud"] == 0].mean())
        line_recall = float(predicted[lines["label_fraud"] == 1].mean())
        if line_fpr <= target_fpr and (
            line_recall > best_recall
            or (line_recall == best_recall and threshold < best_threshold)
        ):
            best_threshold, best_recall = float(threshold), line_recall
    return best_threshold
