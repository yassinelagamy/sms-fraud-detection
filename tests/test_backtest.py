"""
Sanity checks for the legacy if-condition rule re-implementation in
backtest_harness.py.
Run with: pytest tests/test_backtest.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from backtest.backtest_harness import apply_legacy_rule
from scoring.thresholds import select_threshold_at_line_fpr


def _make_features(rows):
    return pd.DataFrame(rows)


def test_burst_pattern_is_flagged():
    features = _make_features([
        {"message_count": 50, "length_cv": 0.02, "unique_recipients": 45},
    ])
    flagged = apply_legacy_rule(features)
    assert flagged.iloc[0] == 1


def test_exactly_n_messages_is_not_flagged_when_rule_says_more_than_n():
    features = _make_features([
        {"message_count": 30, "length_cv": 0.02, "unique_recipients": 30},
    ])
    flagged = apply_legacy_rule(features)
    assert flagged.iloc[0] == 0


def test_normal_pattern_is_not_flagged():
    features = _make_features([
        {"message_count": 6, "length_cv": 0.27, "unique_recipients": 4},
    ])
    flagged = apply_legacy_rule(features)
    assert flagged.iloc[0] == 0


def test_high_volume_but_high_variance_is_not_flagged():
    """Many messages, but lengths vary a lot -> not the burst pattern the rule targets."""
    features = _make_features([
        {"message_count": 50, "length_cv": 0.5, "unique_recipients": 45},
    ])
    flagged = apply_legacy_rule(features)
    assert flagged.iloc[0] == 0


def test_threshold_selection_uses_line_level_false_positive_budget():
    validation = pd.DataFrame([
        {"sender_id": "legit_low", "label_fraud": 0},
        {"sender_id": "legit_high", "label_fraud": 0},
        {"sender_id": "fraud", "label_fraud": 1},
    ])
    threshold = select_threshold_at_line_fpr(
        validation, scores=[0.1, 0.9, 0.8], target_fpr=0.5,
    )
    # The threshold catches the fraud line while allowing only one of two
    # legitimate lines through the 50% test budget.
    assert threshold == 0.8
