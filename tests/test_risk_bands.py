"""
Sanity checks for the risk-band mapping (scoring/risk_bands.py).
Run with: pytest tests/test_risk_bands.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import config
from scoring.risk_bands import band_for, line_risk


def test_band_boundaries():
    assert band_for(0.0) == {"band": "minimal", "action": "none"}
    assert band_for(0.49) == {"band": "minimal", "action": "none"}
    assert band_for(0.50) == {"band": "low", "action": "watchlist"}
    assert band_for(0.75) == {"band": "guarded", "action": "soft_flag"}
    # elevated maps to throttle in config, but throttling is automated
    # enforcement and ALLOW_AUTOMATED_ENFORCEMENT is off in this project.
    assert band_for(0.90) == {"band": "elevated", "action": "manual_review"}
    assert band_for(0.96) == {"band": "high", "action": "manual_review"}
    assert band_for(0.99) == {"band": "critical", "action": "manual_review"}
    assert band_for(1.0) == {"band": "critical", "action": "manual_review"}


def test_scores_above_one_map_to_top_band():
    """Isolation Forest scores beyond the training reference can exceed 1."""
    assert band_for(1.7) == {"band": "critical", "action": "manual_review"}


def test_enforcement_gate_downgrades_all_customer_impacting_actions(monkeypatch):
    """
    With automated enforcement disabled (the shipped default), NO band may
    return an action outside the non-enforcement allowlist -- throttle
    included, not just suspension-like actions (regression: a blocklist once
    let automatic throttling through).
    """
    monkeypatch.setattr(config, "ALLOW_AUTOMATED_ENFORCEMENT", False)
    from scoring.risk_bands import NON_ENFORCEMENT_ACTIONS
    for s in [i / 100 for i in range(0, 105)]:
        assert band_for(s)["action"] in NON_ENFORCEMENT_ACTIONS

    # With enforcement explicitly approved, the configured action passes through.
    monkeypatch.setattr(config, "ALLOW_AUTOMATED_ENFORCEMENT", True)
    assert band_for(0.90)["action"] == "throttle"


def test_bands_cover_unit_interval_without_gaps():
    bands = config.RISK_BANDS
    assert bands[0]["low"] == 0.0
    assert bands[-1]["high"] >= 1.0
    for prev, nxt in zip(bands, bands[1:]):
        assert prev["high"] == nxt["low"], f"gap/overlap at {prev['band']} -> {nxt['band']}"


def test_escalation_is_monotonic():
    """A higher score must never map to an earlier (less severe) band."""
    order = [b["band"] for b in config.RISK_BANDS]
    scores = [i / 100 for i in range(0, 101)]
    ranks = [order.index(band_for(s)["band"]) for s in scores]
    assert ranks == sorted(ranks)


def test_line_risk_is_max_of_windows():
    assert line_risk([0.1, 0.9, 0.4]) == 0.9
    assert line_risk([0.2]) == 0.2
    assert line_risk([]) == 0.0
