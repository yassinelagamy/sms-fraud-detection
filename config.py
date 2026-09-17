"""
Central configuration for the SMS fraud ML pipeline.
Replace these values once real data / infra is wired up.
"""

from pathlib import Path

# --- Paths ---
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
RAW_EVENTS_PATH = DATA_DIR / "sms_events.csv"
FEATURES_PATH = DATA_DIR / "sms_features.csv"
MODEL_DIR = ROOT_DIR / "models" / "artifacts"

# --- Feature window settings ---
# A feature row is emitted after every SMS and summarizes the *trailing* window
# ending at that event.  Do not silently replace this with clock-aligned
# buckets: a campaign can otherwise split sends across a bucket boundary.
WINDOW_MINUTES = 15

# --- Synthetic data generation ---
# 2% fraud-line prevalence is still far above true production rates (<0.1%)
# but statistically workable for model development; document prevalence
# sensitivity before quoting precision numbers to stakeholders.
# Population sized for benchmark statistical power: ~60 fraud lines per
# held-out test split per seed (vs ~12 at N=2000), so recall/FPR estimates
# stop being hostage to single-line sampling noise. Prevalence unchanged.
N_USERS = 10_000
N_FRAUD_USERS = 200         # subset injected with fraud campaign behavior (~2%)
SIM_PERIOD_DAYS = 14
SIM_PERIOD_HOURS = SIM_PERIOD_DAYS * 24

# --- Account lifecycle (legit users only for now — fraud tenure is Step 3) ---
# Most subscribers are established lines with a contact history that predates
# the observation window. A minority are lines newly activated *during* the
# window — the genuine hard-negative case (new legit SIM, high novelty, no
# prior history) that must be distinguished from a newly activated fraud SIM.
# Activation dates are simulator-internal state: they shape behaviour but are
# NOT exported in the dataset schema.
NEW_LINE_FRACTION = 0.12          # calibration knob: replace with real activation-rate data
ESTABLISHED_TENURE_DAYS = (30, 730)  # established lines activated 1mo-2yr before window start

# --- Legitimate-user personas ---
# Every legit line draws ONE persona, then draws its own per-line parameters
# from that persona's ranges, so no two lines share an identical statistical
# fingerprint. Weights and ranges are calibration knobs — replace with real
# Orange subscriber statistics when production data is available.
#
#   weight            : share of the legit population
#   events_per_day    : per-line mean SMS volume range
#   contacts          : per-line contact-book size range
#   new_contact_prob  : per-message chance of texting a never-contacted number
#   length_median     : per-line median message length range (chars)
#   length_sigma      : per-line lognormal sigma range (message-length spread)
#   diurnal           : which hourly activity profile the line follows
#   broadcast_prob    : chance the line sends a festival-greeting broadcast
LEGIT_PERSONAS = {
    "quiet": {          # light users: a few close contacts, sparse messaging
        "weight": 0.45, "events_per_day": (0.2, 1.5), "contacts": (2, 6),
        "new_contact_prob": 0.01, "length_median": (30, 80),
        "length_sigma": (0.35, 0.6), "diurnal": "day_evening", "broadcast_prob": 0.02,
    },
    "regular": {        # typical socially active subscriber
        "weight": 0.30, "events_per_day": (1.0, 4.0), "contacts": (5, 18),
        "new_contact_prob": 0.02, "length_median": (40, 100),
        "length_sigma": (0.4, 0.7), "diurnal": "day_evening", "broadcast_prob": 0.05,
    },
    "social": {         # highly connected: group coordinators, big circles
        "weight": 0.15, "events_per_day": (3.0, 8.0), "contacts": (15, 35),
        "new_contact_prob": 0.04, "length_median": (35, 90),
        "length_sigma": (0.5, 0.8), "diurnal": "all_day", "broadcast_prob": 0.10,
    },
    "power": {          # genuine high-volume humans (organizers, chatty lines)
        "weight": 0.08, "events_per_day": (6.0, 15.0), "contacts": (25, 60),
        "new_contact_prob": 0.03, "length_median": (30, 80),
        "length_sigma": (0.5, 0.9), "diurnal": "all_day", "broadcast_prob": 0.08,
    },
    "night_owl": {      # shift workers / atypical hours
        "weight": 0.02, "events_per_day": (1.0, 6.0), "contacts": (4, 20),
        "new_contact_prob": 0.03, "length_median": (40, 100),
        "length_sigma": (0.4, 0.7), "diurnal": "night", "broadcast_prob": 0.03,
    },
}

# Relative activity weight per hour of day (0-23). Not probabilities —
# normalized at sample time.
DIURNAL_PROFILES = {
    "day_evening": [0.2, 0.1, 0.05, 0.05, 0.05, 0.1, 0.4, 0.8, 1.0, 1.2, 1.3, 1.4,
                    1.5, 1.4, 1.3, 1.4, 1.6, 1.9, 2.2, 2.5, 2.6, 2.3, 1.5, 0.6],
    "all_day":     [0.3, 0.15, 0.1, 0.1, 0.1, 0.2, 0.5, 1.0, 1.4, 1.6, 1.7, 1.8,
                    1.8, 1.7, 1.7, 1.8, 1.9, 2.0, 2.1, 2.1, 2.0, 1.7, 1.2, 0.6],
    "night":       [2.0, 2.2, 2.0, 1.6, 1.0, 0.6, 0.4, 0.3, 0.3, 0.4, 0.5, 0.6,
                    0.8, 0.8, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.2],
}

# --- Fraud personas ---
# Fraud lines are newly activated (production fact) and primarily target
# previously unseen recipients -- that signal is NOT varied here. Diversity
# instead comes from pacing, volume, timing, message-length variation,
# recipient reuse, and campaign evolution, mirroring real evasion tactics
# against the legacy rule. Weights/ranges are calibration knobs -- replace
# with Orange's confirmed-fraud investigation statistics when available.
FRAUD_PERSONAS = {
    "aggressive": {
        # Today's original template: fast, high-volume, near-identical
        # length. Kept as the baseline the legacy rule was built to catch.
        "weight": 0.30,
        "n_events": (40, 150),
        "length_range": (60, 160), "length_jitter": 3,
        "burst_minutes": (10, 30),
        "activation_lead_hours": (1, 24),
    },
    "threshold_evasion": {
        # Probes the rule's hard cutoff: stays just under
        # OLD_RULE_MIN_MESSAGES per window, repeated across several windows.
        "weight": 0.15,
        "n_windows": (3, 6),
        "msgs_per_window_frac": (0.65, 0.93),  # fraction of OLD_RULE_MIN_MESSAGES
        "length_range": (60, 160), "length_jitter": 3,
        "activation_lead_hours": (2, 48),
    },
    "stealthy": {
        # Defeats length_cv via text-spinning (wide length range instead of a
        # fixed template) and paces sends over hours so no window bursts.
        "weight": 0.15,
        "n_events": (60, 200),
        "length_range": (40, 160),
        "spread_hours": (3, 10),
        "activation_lead_hours": (2, 72),
    },
    "slow_drip": {
        # No single window ever resembles a burst -- only a multi-day
        # aggregate view would catch it. Expected to depress window-level
        # recall; that's an honest gap, not a bug (motivates future
        # multi-window/daily aggregate features).
        "weight": 0.15,
        "n_events": (100, 300),
        "spread_days": (5, 10),
        "length_range": (40, 160), "length_jitter": 20,
        "activation_lead_hours": (6, 48),
    },
    "mixed_recipient": {
        # Purchased/duplicated recipient lists: novelty and fan-out are high
        # but not 1.0, so no model can learn "novelty == 1.0 exactly" as an
        # artifact of the simulator.
        "weight": 0.10,
        "n_events": (50, 150),
        "target_list_size": (15, 40),
        "reuse_prob": (0.4, 0.7),
        "length_range": (50, 150), "length_jitter": 15,
        "burst_minutes": (15, 60),
        "activation_lead_hours": (2, 48),
    },
    "evolving": {
        # Multi-phase campaign: camouflage-like warm-up -> escalation -> full
        # blast. The legacy rule only fires on the final phase, after most
        # damage is done -- sets up a future time-to-detection comparison.
        "weight": 0.15,
        "activation_lead_hours": (12, 72),
        "phases": [
            {"n_events": (5, 15), "length_jitter": 25, "spread_hours": (24, 48)},
            {"n_events": (20, 40), "length_jitter": 10, "spread_hours": (2, 6)},
            {"n_events": (50, 100), "length_jitter": 3, "spread_minutes": (10, 30)},
        ],
    },
}

# Optional camouflage prelude for non-evolving personas: a handful of
# normal-looking messages before the campaign, mimicking a new SIM "warming
# up" to look human. These events are label_fraud=0: labels follow fraud
# ONSET (campaign_start), because at the decision time of a pre-onset event
# no fraud is observable yet — labelling them 1 would leak the line's future
# outcome into training and inflate early-detection metrics. The line still
# counts as a fraud line (its campaign events are label 1; line-level metrics
# take the max over windows).
FRAUD_CAMOUFLAGE = {
    "prob": 0.3,
    "n_events": (2, 8),
    "length_range": (40, 160),
    "distinct_recipients": (2, 4),
    "lead_hours_before_campaign": (6, 48),
}

# Shared synthetic MSISDN space fraud recipients are drawn from -- the same
# space legitimate users' new-contact draws use (see
# generate_normal_user_events), so high recipient novelty emerges from
# behaviour (few repeats) rather than a guaranteed-unique ID scheme.
MSISDN_SPACE = (1, 50_000)

# --- Event-driven legitimate broadcasts (festival greetings to KNOWN contacts) ---
# One calendar event inside the sim window (think Eid / New Year). Lines active
# that day may blast a templated greeting to most of their contact book within
# minutes: high count + near-identical length + many recipients — the exact
# shape the legacy rule flags — but recipient novelty stays ~0 because these
# are existing contacts. Calibrate broadcast_prob against the operator's real
# false-shutdown complaint rate.
EVENT_BROADCAST = {
    "event_day_offset": 6,      # festival lands on day 7 of the 14-day window
    "start_hour_range": (9, 22),  # greetings go out during waking hours
    "spread_minutes": (5, 30),  # whole broadcast sent within this span
    "length_jitter": 5,         # +/- chars around the greeting template
    "coverage": (0.7, 1.0),     # fraction of the contact book greeted
}

# --- Isolation Forest baseline ---
ISO_FOREST_CONTAMINATION = 0.05   # rough prior on fraud rate; tune with real data
ISO_FOREST_N_ESTIMATORS = 200
RANDOM_STATE = 42

# Keep the supervised baseline inexpensive enough for the multi-seed backtest.
# These are baseline settings, not production-tuned hyperparameters.
XGB_N_ESTIMATORS = 100
XGB_MAX_DEPTH = 3
XGB_LEARNING_RATE = 0.05

# --- Risk tiering (replaces single binary shutdown) ---
# Retained for the uncalibrated tier sweep in the backtest; the canonical
# action mapping for deployment is CONFIDENCE_BANDS below, which operates on
# CALIBRATED confidence (see scoring/confidence.py).
RISK_THRESHOLDS = {
    "soft_flag": 0.5,     # log for review, no action
    "throttle": 0.7,      # rate-limit the line
    "manual_review": 0.85,  # queue for human review
    "shutdown": 0.95,     # auto-shutdown, same as current rule's action
}

# --- Deployment and risk policy ---
# Keep this aligned with the model whose output feeds score_realtime.py and the
# backtest risk-band report.  A model change requires a fresh policy review.
# XGBoost is the evidence-backed candidate in the current 10k re-baseline:
# it satisfies the line-FPR budget while retaining high recall, whereas
# Isolation Forest cannot flag any line within that budget. RISK_BANDS below
# were sized from XGBoost's score distribution; pointing this at another model
# without re-sizing the bands would make the band report misleading.
DEPLOYMENT_MODEL_KIND = "xgboost"  # "isolation_forest" or "xgboost"

# Scores in this starter project are not calibrated on real outcomes.  Until
# they are, no model score may trigger ANY automated customer-impacting
# action -- throttling included, not just suspension (scoring/risk_bands.py
# enforces this with an allowlist of non-enforcement actions).  Such cases
# are routed to manual review; a production rollout can enable a separately
# approved automated action only after calibration and shadow-mode
# validation on real data.
ALLOW_AUTOMATED_ENFORCEMENT = False

# Thresholds are selected on a sender-disjoint validation set to satisfy this
# maximum *line-level* false-positive rate, rather than maximizing a
# window-level classification metric.  Refit with real historical outcomes and
# the approved customer-impact budget.
TARGET_LINE_FALSE_POSITIVE_RATE = 0.005

# --- Risk bands (relative risk score -> operational action) ---
# The model's output is treated as a RELATIVE risk score (higher = riskier),
# NOT a calibrated probability -- raw scores rank risk but their absolute
# values carry no validated statistical meaning on synthetic data. Band names
# are deliberately qualitative so nothing reads as a percentage claim.
#
# Probability calibration (Platt/isotonic + reliability validation) is
# DEFERRED until real Orange labeled data exists; it plugs into
# scoring/risk_bands.py at that point, and these boundaries get re-fit
# against observed fraud rates and ops queue capacity.
#
# Escalation logic: cheap/reversible actions for mid bands (where false
# positives concentrate), customer-impacting actions only at the top.
#
# Boundaries are sized from the backtest's line-level band-volume report
# (max window score per line): legit lines' spurious high windows smear up
# to ~0.99 while fraud lines concentrate above it, so the suspend boundary
# sits at 0.99 -- an even 0-1 spacing put ~40% of legit lines in the
# suspend band. Re-size against real volumes when Orange data arrives.
RISK_BANDS = [
    {"low": 0.00, "high": 0.50, "band": "minimal",  "action": "none"},           # log score only
    {"low": 0.50, "high": 0.70, "band": "low",      "action": "watchlist"},      # auto-recheck next windows
    {"low": 0.70, "high": 0.85, "band": "guarded",  "action": "soft_flag"},      # persistent case record
    {"low": 0.85, "high": 0.95, "band": "elevated", "action": "throttle"},       # rate-limit outbound SMS (gated: manual_review until ALLOW_AUTOMATED_ENFORCEMENT)
    {"low": 0.95, "high": 0.99, "band": "high",     "action": "manual_review"},  # human review queue
    {"low": 0.99, "high": 1.01, "band": "critical", "action": "manual_review"},  # enforcement is gated above
]

# --- Legacy if-condition rule (re-implemented here for backtest comparison) ---
# "more than N messages of similar length (with some variance) to multiple
# numbers within a short time window" -> auto-shutdown.
OLD_RULE_MIN_MESSAGES = 30       # N: message count threshold in-window
# The stated company rule is "more than N"; retain this explicitly so an
# equality case cannot drift unnoticed during future refactors.
OLD_RULE_MESSAGE_COMPARATOR = ">"
OLD_RULE_LENGTH_CV_MAX = 0.10    # "similar length": coefficient of variation ceiling
OLD_RULE_MIN_RECIPIENTS = 10     # "multiple numbers": distinct recipients in-window

# --- Backtest ---
BACKTEST_TEST_SIZE = 0.3
BACKTEST_VAL_SIZE = 0.25   # sender-disjoint carve-out OF THE TRAIN SPLIT, used
                           # only for ML threshold calibration (never test data)
BACKTEST_SEEDS = [42, 43, 44, 45, 46]  # independent populations per seed;
                                       # metrics reported as mean +/- std
