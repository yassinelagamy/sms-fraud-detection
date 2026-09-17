"""
Basic sanity checks for feature_engineering.py.
Run with: pytest tests/test_features.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from features.feature_engineering import compute_window_features


def _make_events(rows):
    return pd.DataFrame(rows)


def test_single_user_trailing_window_counts():
    events = _make_events([
        {"sender_id": "u1", "recipient_id": "r1", "timestamp": "2026-01-01 10:00:00",
         "message_length": 100, "is_new_recipient": True, "label_fraud": 0},
        {"sender_id": "u1", "recipient_id": "r2", "timestamp": "2026-01-01 10:02:00",
         "message_length": 102, "is_new_recipient": True, "label_fraud": 0},
    ])
    features = compute_window_features(events)

    assert len(features) == 2  # one snapshot after each incoming SMS
    assert features.iloc[-1]["message_count"] == 2
    assert features.iloc[-1]["unique_recipients"] == 2
    assert features.iloc[-1]["new_recipient_ratio"] == 1.0


def test_fraud_burst_has_low_length_variance_and_high_new_ratio():
    rows = []
    for i in range(50):
        # all 50 events within the same 15-minute window (10:00:00-10:00:49)
        rows.append({
            "sender_id": "fraud_u",
            "recipient_id": f"r{i}",
            "timestamp": f"2026-01-01 10:00:{i % 60:02d}",
            "message_length": 100 + (i % 3),  # near-identical length
            "is_new_recipient": True,
            "label_fraud": 1,
        })
    events = _make_events(rows)
    features = compute_window_features(events)

    row = features.iloc[-1]
    assert row["message_count"] == 50
    assert row["new_recipient_ratio"] == 1.0
    assert row["length_cv"] < 0.05  # low variance relative to mean
    assert row["label_fraud"] == 1


def test_empty_events_returns_empty_features():
    events = _make_events([])
    events = events.assign(sender_id=[], recipient_id=[], timestamp=[],
                            message_length=[], is_new_recipient=[], label_fraud=[])
    features = compute_window_features(events)
    assert features.empty


def test_trailing_window_catches_a_burst_across_clock_boundary():
    rows = []
    for i in range(31):
        rows.append({
            "sender_id": "u1", "recipient_id": f"r{i}",
            # Fifteen sends immediately before and sixteen immediately after
            # 10:15:00 must still be visible together in a trailing window.
            "timestamp": f"2026-01-01 10:{14 if i < 15 else 15}:{i % 15:02d}",
            "message_length": 100, "is_new_recipient": True, "label_fraud": 1,
        })
    features = compute_window_features(_make_events(rows))
    assert features.iloc[-1]["message_count"] == 31
    assert features.iloc[-1]["unique_recipients"] == 31
