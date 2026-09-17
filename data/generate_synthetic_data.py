"""
Generates synthetic SMS event logs that mimic telecom CDR/SMS metadata.

Schema (one row per SMS sent):
    event_id        : unique event id
    sender_id       : subscriber/line id
    recipient_id    : destination number (hashed/anonymized in real life)
    timestamp       : datetime of the message
    message_length  : character count of the message body
    is_new_recipient: whether sender has never messaged this recipient before
                      (precomputed here for convenience; real pipeline would
                      derive this from history)
    label_fraud     : ground truth (1 = fraud), ONLY present because this is
                      synthetic. Real data usually won't have this — it's
                      what you'd backfill from confirmed fraud/manual review
                      cases.

Replace this file with a loader for your actual CDR/SMS gateway export once
you know the real schema. Keep the output columns compatible so
feature_engineering.py doesn't need to change.
"""

import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import config


def _random_timestamp(start: datetime, hours: int) -> datetime:
    offset_minutes = random.uniform(0, hours * 60)
    return start + timedelta(minutes=offset_minutes)


def _sample_persona() -> str:
    names = list(config.LEGIT_PERSONAS)
    weights = [config.LEGIT_PERSONAS[n]["weight"] for n in names]
    return random.choices(names, weights=weights)[0]


def _sample_message_length(median: float, sigma: float) -> int:
    """Right-skewed per-user message length (real SMS is short-heavy)."""
    return int(np.clip(np.random.lognormal(np.log(median), sigma), 2, 320))


def _diurnal_timestamp(
    active_start: datetime, active_end: datetime, hourly_weights: list[float]
) -> datetime:
    """
    Sample a timestamp inside [active_start, active_end] following an
    hour-of-day activity profile (humans sleep; uniform-over-window doesn't).
    Rejection-samples to respect activation bounds, with a uniform fallback.
    """
    base = active_start.replace(hour=0, minute=0, second=0, microsecond=0)
    n_days = (active_end - base).days + 1
    hours = range(24)
    for _ in range(12):
        ts = base + timedelta(
            days=random.randint(0, n_days - 1),
            hours=random.choices(hours, weights=hourly_weights)[0],
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59),
        )
        if active_start <= ts <= active_end:
            return ts
    span = max((active_end - active_start).total_seconds(), 1.0)
    return active_start + timedelta(seconds=random.uniform(0, span))


def _generate_event_broadcast(
    user_id: str, contacts: list[str], active_start: datetime,
    active_end: datetime, sim_start: datetime,
) -> list[dict]:
    """
    Festival-greeting broadcast to EXISTING contacts (e.g. Eid): a templated,
    near-identical message blasted to most of the contact book within minutes.
    Looks exactly like the legacy rule's fraud shape on count/length-variance/
    fan-out — but recipient novelty stays ~0, which is the signal that
    separates it from an actual fraud campaign.
    """
    cfg = config.EVENT_BROADCAST
    event_date = sim_start + timedelta(days=cfg["event_day_offset"])
    if not (active_start <= event_date <= active_end):
        return []  # line wasn't active on the festival day

    burst_start = event_date + timedelta(hours=random.uniform(*cfg["start_hour_range"]))
    greeted = random.sample(contacts, max(1, int(len(contacts) * random.uniform(*cfg["coverage"]))))
    template_length = random.randint(60, 160)
    spread_sec = random.uniform(*cfg["spread_minutes"]) * 60

    return [{
        "sender_id": user_id,
        "recipient_id": recipient,
        "timestamp": burst_start + timedelta(seconds=random.uniform(0, spread_sec)),
        "message_length": template_length + random.randint(-cfg["length_jitter"], cfg["length_jitter"]),
        "label_fraud": 0,
    } for recipient in greeted]


def generate_normal_user_events(
    user_id: str, start: datetime
) -> tuple[list[dict], set[tuple[str, str]]]:
    """
    Legitimate human subscriber. Behaviour is persona-driven (see
    config.LEGIT_PERSONAS): each line draws a persona, then its own volume,
    contact book, messaging style, daily rhythm, and (optionally) one
    festival-greeting broadcast — so legit traffic is heterogeneous instead
    of one shared statistical distribution.

    Lifecycle: established lines (majority) have a contact history that
    predates the observation window — their pre-existing contacts are
    returned separately so the novelty pass doesn't misread a first *logged*
    message to a long-time contact as "new". Newly activated lines start with
    no history and cannot send before their own (simulator-internal)
    activation timestamp.
    """
    persona = config.LEGIT_PERSONAS[_sample_persona()]
    window_end = start + timedelta(hours=config.SIM_PERIOD_HOURS)

    is_new_line = random.random() < config.NEW_LINE_FRACTION
    if is_new_line:
        active_start = _random_timestamp(start, config.SIM_PERIOD_HOURS)
        # a brand-new line is still building its contact book
        pool_size = max(2, random.randint(*persona["contacts"]) // 3)
        contacts = [f"num_{random.randint(1, 2000)}" for _ in range(pool_size)]
        preexisting_pairs: set[tuple[str, str]] = set()
    else:
        active_start = start
        pool_size = random.randint(*persona["contacts"])
        contacts = [f"num_{random.randint(1, 2000)}" for _ in range(pool_size)]
        preexisting_pairs = {(user_id, r) for r in contacts}

    # Zipf-like reuse: earlier contacts are closer ties and get texted most.
    reuse_weights = [1.0 / (rank + 1) for rank in range(len(contacts))]

    # Per-line messaging style: personal length median/spread, daily rhythm.
    length_median = random.uniform(*persona["length_median"])
    length_sigma = random.uniform(*persona["length_sigma"])
    hourly_weights = config.DIURNAL_PROFILES[persona["diurnal"]]

    active_days = max((window_end - active_start).total_seconds() / 86400.0, 0.05)
    n_events = max(1, int(
        random.uniform(*persona["events_per_day"]) * active_days * random.uniform(0.7, 1.3)
    ))

    events = []
    for _ in range(n_events):
        if random.random() < persona["new_contact_prob"]:
            recipient = f"num_{random.randint(1, 50000)}"  # genuinely new number
            contacts.append(recipient)
            reuse_weights.append(1.0 / len(contacts))
        else:
            recipient = random.choices(contacts, weights=reuse_weights)[0]
        events.append({
            "sender_id": user_id,
            "recipient_id": recipient,
            "timestamp": _diurnal_timestamp(active_start, window_end, hourly_weights),
            "message_length": _sample_message_length(length_median, length_sigma),
            "label_fraud": 0,
        })

    if random.random() < persona["broadcast_prob"]:
        events.extend(_generate_event_broadcast(
            user_id, contacts, active_start, window_end, start))

    return events, preexisting_pairs


def _fraud_recipient(target_list: list[str] | None = None) -> str:
    """
    Draw a fraud recipient. With no target list, draws fresh from the shared
    MSISDN space (config.MSISDN_SPACE) -- the same space legit new-contact
    draws use -- so high novelty emerges from behaviour, not from a
    guaranteed-unique ID scheme. With a target list (mixed_recipient
    persona), reuses it to simulate a purchased/duplicated recipient list.
    """
    if target_list is not None:
        return random.choice(target_list)
    return f"num_{random.randint(*config.MSISDN_SPACE)}"


def _fraud_event(user_id: str, recipient: str, ts: datetime, length: int) -> dict:
    return {
        "sender_id": user_id,
        "recipient_id": recipient,
        "timestamp": ts,
        "message_length": max(2, length),
        "label_fraud": 1,
    }


def _generate_camouflage(user_id: str, campaign_start: datetime, sim_start: datetime) -> list[dict]:
    """
    Optional prelude: a handful of normal-looking messages to a couple of
    repeated recipients before the campaign starts, mimicking a new SIM
    "warming up" to look human. These events end up label_fraud=0 — the
    onset relabel in generate_fraud_user_events zeroes everything before
    campaign_start (see config.FRAUD_CAMOUFLAGE).
    """
    cfg = config.FRAUD_CAMOUFLAGE
    if random.random() >= cfg["prob"]:
        return []
    lead_hours = random.uniform(*cfg["lead_hours_before_campaign"])
    window_start = max(sim_start, campaign_start - timedelta(hours=lead_hours))
    span = max((campaign_start - window_start).total_seconds(), 1.0)
    contacts = [_fraud_recipient() for _ in range(random.randint(*cfg["distinct_recipients"]))]

    events = []
    for _ in range(random.randint(*cfg["n_events"])):
        ts = window_start + timedelta(seconds=random.uniform(0, span))
        length = random.randint(*cfg["length_range"])
        events.append(_fraud_event(user_id, random.choice(contacts), ts, length))
    return events


def _gen_aggressive(user_id: str, campaign_start: datetime, p: dict) -> list[dict]:
    """Fast, high-volume, near-identical length -- the legacy rule's original target."""
    fixed_length = random.randint(*p["length_range"])
    spread_sec = random.uniform(*p["burst_minutes"]) * 60
    return [
        _fraud_event(
            user_id, _fraud_recipient(),
            campaign_start + timedelta(seconds=random.uniform(0, spread_sec)),
            fixed_length + random.randint(-p["length_jitter"], p["length_jitter"]),
        )
        for _ in range(random.randint(*p["n_events"]))
    ]


def _gen_threshold_evasion(user_id: str, campaign_start: datetime, p: dict) -> list[dict]:
    """Stays just under OLD_RULE_MIN_MESSAGES per window, repeated across windows."""
    window_td = timedelta(minutes=config.WINDOW_MINUTES)
    fixed_length = random.randint(*p["length_range"])
    events = []
    for w in range(random.randint(*p["n_windows"])):
        w_start = campaign_start + w * window_td
        n_msgs = max(1, int(config.OLD_RULE_MIN_MESSAGES * random.uniform(*p["msgs_per_window_frac"])))
        for _ in range(n_msgs):
            ts = w_start + timedelta(seconds=random.uniform(0, window_td.total_seconds()))
            length = fixed_length + random.randint(-p["length_jitter"], p["length_jitter"])
            events.append(_fraud_event(user_id, _fraud_recipient(), ts, length))
    return events


def _gen_stealthy(user_id: str, campaign_start: datetime, p: dict) -> list[dict]:
    """Text-spun length (wide random range defeats length_cv), paced over hours."""
    spread_sec = random.uniform(*p["spread_hours"]) * 3600
    events = []
    for _ in range(random.randint(*p["n_events"])):
        ts = campaign_start + timedelta(seconds=random.uniform(0, spread_sec))
        length = random.randint(*p["length_range"])
        events.append(_fraud_event(user_id, _fraud_recipient(), ts, length))
    return events


def _gen_slow_drip(user_id: str, campaign_start: datetime, p: dict) -> list[dict]:
    """Spread over days -- no single window ever looks like a burst."""
    spread_sec = random.uniform(*p["spread_days"]) * 86400
    fixed_length = random.randint(*p["length_range"])
    events = []
    for _ in range(random.randint(*p["n_events"])):
        ts = campaign_start + timedelta(seconds=random.uniform(0, spread_sec))
        length = fixed_length + random.randint(-p["length_jitter"], p["length_jitter"])
        events.append(_fraud_event(user_id, _fraud_recipient(), ts, length))
    return events


def _gen_mixed_recipient(user_id: str, campaign_start: datetime, p: dict) -> list[dict]:
    """Purchased/duplicated recipient list -- novelty and fan-out high but not 1.0."""
    target_list = [_fraud_recipient() for _ in range(random.randint(*p["target_list_size"]))]
    reuse_prob = random.uniform(*p["reuse_prob"])
    fixed_length = random.randint(*p["length_range"])
    spread_sec = random.uniform(*p["burst_minutes"]) * 60
    events = []
    for _ in range(random.randint(*p["n_events"])):
        ts = campaign_start + timedelta(seconds=random.uniform(0, spread_sec))
        length = fixed_length + random.randint(-p["length_jitter"], p["length_jitter"])
        recipient = _fraud_recipient(target_list) if random.random() < reuse_prob else _fraud_recipient()
        events.append(_fraud_event(user_id, recipient, ts, length))
    return events


def _gen_evolving(user_id: str, campaign_start: datetime, p: dict) -> list[dict]:
    """Multi-phase: camouflage-like warm-up -> escalation -> full blast."""
    events = []
    cursor = campaign_start
    for phase in p["phases"]:
        span = (random.uniform(*phase["spread_hours"]) * 3600 if "spread_hours" in phase
                else random.uniform(*phase["spread_minutes"]) * 60)
        base_length = random.randint(60, 160)
        for _ in range(random.randint(*phase["n_events"])):
            ts = cursor + timedelta(seconds=random.uniform(0, span))
            length = base_length + random.randint(-phase["length_jitter"], phase["length_jitter"])
            events.append(_fraud_event(user_id, _fraud_recipient(), ts, length))
        cursor = cursor + timedelta(seconds=span)
    return events


FRAUD_GENERATORS = {
    "aggressive": _gen_aggressive,
    "threshold_evasion": _gen_threshold_evasion,
    "stealthy": _gen_stealthy,
    "slow_drip": _gen_slow_drip,
    "mixed_recipient": _gen_mixed_recipient,
    "evolving": _gen_evolving,
}


def _sample_fraud_persona() -> str:
    names = list(config.FRAUD_PERSONAS)
    weights = [config.FRAUD_PERSONAS[n]["weight"] for n in names]
    return random.choices(names, weights=weights)[0]


def _allocate_fraud_personas(n: int) -> list[str]:
    """
    Deterministically allocate persona counts proportional to their weights
    (largest-remainder rounding), then shuffle the assignment order. The
    population *mix* is the realism claim; drawing each line's persona
    independently would only add sampling noise that starves rare personas
    of evaluation coverage (e.g. a 0.15-weight persona getting 1 line of 25).
    """
    names = list(config.FRAUD_PERSONAS)
    weights = np.array([config.FRAUD_PERSONAS[m]["weight"] for m in names], dtype=float)
    weights = weights / weights.sum()
    exact = weights * n
    counts = np.floor(exact).astype(int)
    for idx in np.argsort(-(exact - counts))[: n - counts.sum()]:
        counts[idx] += 1
    personas = [name for name, c in zip(names, counts) for _ in range(c)]
    random.shuffle(personas)
    return personas


def _persona_duration_hours(name: str, p: dict) -> float:
    """Upper bound on campaign span, to reserve enough runway before window_end."""
    if name == "aggressive" or name == "mixed_recipient":
        return p["burst_minutes"][1] / 60
    if name == "threshold_evasion":
        return p["n_windows"][1] * config.WINDOW_MINUTES / 60
    if name == "stealthy":
        return p["spread_hours"][1]
    if name == "slow_drip":
        return p["spread_days"][1] * 24
    if name == "evolving":
        return sum(
            phase["spread_hours"][1] if "spread_hours" in phase else phase["spread_minutes"][1] / 60
            for phase in p["phases"]
        )
    return 1.0


def generate_fraud_user_events(
    user_id: str, start: datetime, persona_name: str | None = None
) -> tuple[list[dict], dict]:
    """
    Fraud line: newly activated, no pre-seeded contact history (returns no
    preexisting_pairs -- main()'s novelty pass never seeds fraud senders).
    Recipients are drawn from the shared MSISDN space legit new-contact draws
    use, so high novelty emerges from behaviour rather than an artificial ID
    scheme.

    Persona (config.FRAUD_PERSONAS) controls pacing/volume/length variation/
    recipient reuse/campaign structure -- NOT recipient novelty, which stays
    a production-observed constant across personas. If persona_name is None,
    one is sampled by weight.
    """
    window_end = start + timedelta(hours=config.SIM_PERIOD_HOURS)
    if persona_name is None:
        persona_name = _sample_fraud_persona()
    p = config.FRAUD_PERSONAS[persona_name]

    duration_hours = _persona_duration_hours(persona_name, p)
    latest_start_hours = max(config.SIM_PERIOD_HOURS - duration_hours - 1, 0)
    campaign_start = start + timedelta(hours=random.uniform(0, latest_start_hours))
    activation_date = campaign_start - timedelta(hours=random.uniform(*p["activation_lead_hours"]))

    events = FRAUD_GENERATORS[persona_name](user_id, campaign_start, p)

    # Evolving's own first phase is already a camouflage-like warm-up, so the
    # generic prelude only applies to the other five personas.
    camouflage_used = False
    if persona_name != "evolving":
        camouflage_events = _generate_camouflage(user_id, campaign_start, start)
        if camouflage_events:
            camouflage_used = True
            events = camouflage_events + events

    # Safety clip: every event must land inside the observation window.
    events = [e for e in events if start <= e["timestamp"] <= window_end]

    # Labels follow fraud ONSET (campaign_start), not line identity: at the
    # decision time of a camouflage event, no fraudulent behaviour has
    # happened yet, so no oracle could label it positive. Labelling pre-onset
    # events fraud lets the model train on retrospective line outcomes
    # (target-timing leakage) and fakes "early detection" — the model gets
    # credit for flagging warm-up windows that precede any observable fraud.
    # The line as a whole still evaluates as fraud: campaign events keep
    # label 1, and line-level metrics take the max over a sender's windows.
    for e in events:
        e["label_fraud"] = int(e["timestamp"] >= campaign_start)

    metadata = {
        "sender_id": user_id,
        "group": "fraud",
        "persona": persona_name,
        "activation_date": activation_date,
        "campaign_start": campaign_start,
        "camouflage_used": camouflage_used,
        "n_events": len(events),
    }
    return events, metadata


def generate_dataset(seed: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate one full synthetic population. Returns (events_df, metadata_df)
    in memory; main() wraps this and writes the CSVs. The seed parameter lets
    the multi-seed backtest regenerate independent populations without going
    through the filesystem.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    start = datetime(2026, 1, 1)
    all_events = []
    preexisting_pairs: set[tuple[str, str]] = set()
    sim_metadata = []  # simulator-internal diagnostics only; never fed to features/models

    fraud_user_ids = random.sample(range(config.N_USERS), config.N_FRAUD_USERS)
    persona_by_user = dict(zip(fraud_user_ids, _allocate_fraud_personas(config.N_FRAUD_USERS)))

    for i in range(config.N_USERS):
        user_id = f"user_{i}"
        if i in persona_by_user:
            fraud_events, meta = generate_fraud_user_events(user_id, start, persona_by_user[i])
            all_events.extend(fraud_events)
            sim_metadata.append(meta)
        else:
            legit_events, pairs = generate_normal_user_events(user_id, start)
            all_events.extend(legit_events)
            preexisting_pairs.update(pairs)
            sim_metadata.append({
                "sender_id": user_id, "group": "legit", "persona": None,
                "activation_date": None, "campaign_start": None,
                "camouflage_used": None, "n_events": len(legit_events),
            })

    df = pd.DataFrame(all_events)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["event_id"] = df.index.map(lambda i: f"evt_{i}")

    # is_new_recipient: has this sender->recipient pair not appeared before this
    # row. Seeded with established lines' pre-existing contacts so their first
    # *logged* message to a long-time contact isn't misread as novel — fraud
    # users get no such seed, since they're modeled as newly activated lines
    # with no prior relationships (consistent with the production
    # characteristic that fraud predominantly targets never-before-contacted
    # numbers).
    seen_pairs = set(preexisting_pairs)
    is_new = []
    for row in df.itertuples():
        pair = (row.sender_id, row.recipient_id)
        is_new.append(pair not in seen_pairs)
        seen_pairs.add(pair)
    df["is_new_recipient"] = is_new

    cols = ["event_id", "sender_id", "recipient_id", "timestamp",
            "message_length", "is_new_recipient", "label_fraud"]
    return df[cols], pd.DataFrame(sim_metadata)


def main():
    df, metadata = generate_dataset(config.RANDOM_STATE)

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RAW_EVENTS_PATH, index=False)
    print(f"Generated {len(df)} events for {config.N_USERS} users "
          f"({config.N_FRAUD_USERS} fraud) -> {config.RAW_EVENTS_PATH}")

    # Diagnostics sidecar: simulator metadata only (persona, activation,
    # campaign timing). NEVER consumed by feature_engineering.py, model
    # training, or scoring -- evaluation/diagnostics use only, analogous to
    # investigation case-file annotations in production.
    meta_path = config.DATA_DIR / "_sim_metadata.csv"
    metadata.to_csv(meta_path, index=False)
    print(f"Saved simulator metadata -> {meta_path}")


if __name__ == "__main__":
    main()
