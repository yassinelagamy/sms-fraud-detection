"""
Regression tests for onset-based fraud labels (data/generate_synthetic_data.py).

The original generator labelled camouflage warm-up events fraud even though
they occur BEFORE the campaign starts -- target-timing leakage that let the
model train on retrospective line outcomes and faked "early detection".
Labels must follow fraud onset (campaign_start), never line identity alone.
Run with: pytest tests/test_onset_labels.py
"""

import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from data.generate_synthetic_data import generate_fraud_user_events


def test_fraud_labels_follow_campaign_onset():
    start = datetime(2026, 1, 1)
    camouflage_lines_checked = 0

    for s in range(120):
        random.seed(s)
        np.random.seed(s)
        events, meta = generate_fraud_user_events(f"user_{s}", start)
        onset = meta["campaign_start"]

        # The invariant itself: label 1 iff the event is at/after fraud onset.
        for e in events:
            assert e["label_fraud"] == int(e["timestamp"] >= onset), (
                f"seed {s}: event at {e['timestamp']} labelled "
                f"{e['label_fraud']} with onset {onset}"
            )

        # A fraud line must keep at least one positive event, so line-level
        # metrics (max over windows) still see it as a fraud line.
        assert any(e["label_fraud"] == 1 for e in events), f"seed {s}: no positive events"

        if meta["camouflage_used"]:
            pre_onset = [e for e in events if e["timestamp"] < onset]
            if pre_onset:
                camouflage_lines_checked += 1
                assert all(e["label_fraud"] == 0 for e in pre_onset)

    # The test must actually have exercised the camouflage path
    # (config.FRAUD_CAMOUFLAGE prob is 0.3, so 120 draws make this certain
    # in practice; if this trips, the camouflage config changed).
    assert camouflage_lines_checked > 0
