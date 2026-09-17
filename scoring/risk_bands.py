"""
Risk-band mapping: turns a model's RELATIVE risk score into an operational
action tier (config.RISK_BANDS).

Deliberately NOT calibrated probabilities. Raw model scores rank risk, but
their absolute values carry no validated statistical meaning -- in this
project's own synthetic backtests show that score distributions vary by model
and seed, so reading a raw 0.75 as "75% confident" would be a lie. Until real
Orange labeled data exists, a score is only a position on a relative scale,
and the bands are operational cutoffs sized against backtest volumes, not
probability statements.

--- Calibration slot (deferred) ---
When real labeled data arrives, probability calibration (Platt scaling or
isotonic regression, fitted on a train-side validation set and verified with
a per-band reliability report: observed fraud rate per band ~= band range)
plugs in HERE, upstream of band_for(). Band boundaries then get re-fit
against observed fraud rates and ops review capacity. Nothing else in the
pipeline needs to change -- that is the point of keeping this module as the
single score->action chokepoint.

Line-level aggregation: production decisions apply to LINES, not 15-minute
windows. A line's risk is the max score across its scored windows so far --
one critical window is enough to act on, and the max is trivially auditable
("this window triggered it"). Persistence (how many windows flagged) is
review-queue context, not part of the score.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
import config

# Actions that never affect a customer's service: logging, watchlisting,
# case records, and routing to a human. Everything else counts as automated
# enforcement and is gated behind config.ALLOW_AUTOMATED_ENFORCEMENT.
NON_ENFORCEMENT_ACTIONS = {"none", "watchlist", "soft_flag", "manual_review"}


def band_for(risk_score: float) -> dict:
    """
    Map a relative risk score (non-negative, higher = riskier) to its band
    and action. Scores at/above the top band's upper edge (e.g. Isolation
    Forest scores beyond the training 99th-percentile reference, which
    exceed 1) map to the top band.
    """
    result = None
    for band in config.RISK_BANDS:
        if band["low"] <= risk_score < band["high"]:
            result = {"band": band["band"], "action": band["action"]}
            break
    if result is None:
        # Score exactly at/above the last band's high (e.g. 1.0 with float edges).
        top = config.RISK_BANDS[-1]
        result = {"band": top["band"], "action": top["action"]}

    # A configuration mistake must not turn an uncalibrated starter model into
    # an automated customer-impacting action. This is an ALLOWLIST, not a
    # blocklist: anything that touches the customer's service (throttle,
    # suspend, shutdown, or a future action added to RISK_BANDS) is
    # enforcement and gets downgraded to human review unless automated
    # enforcement has been explicitly approved. A blocklist here once let
    # automatic throttling through despite ALLOW_AUTOMATED_ENFORCEMENT=False.
    if not config.ALLOW_AUTOMATED_ENFORCEMENT and result["action"] not in NON_ENFORCEMENT_ACTIONS:
        result["action"] = "manual_review"
    return result


def line_risk(window_scores) -> float:
    """
    A line's risk is the max score across its scored windows: one critical
    window is enough to act on, and the trigger window stays auditable.
    """
    arr = np.asarray(list(window_scores), dtype=float)
    if arr.size == 0:
        return 0.0
    return float(arr.max())
