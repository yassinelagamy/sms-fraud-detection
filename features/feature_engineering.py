"""
Turns raw SMS events into per-user, per-window features.

This is the ML replacement for the current if-condition rule. Instead of a
single hard threshold ("if message_count > X and length_variance < Y"), we
compute a richer feature vector per (user, time window) that a model can
learn thresholds and interactions from.

Features computed per window:
    message_count        : total messages sent in the window
    unique_recipients     : distinct recipients messaged
    new_recipient_ratio   : fraction of messages to recipients never
                             messaged before
    length_mean            : mean message length
    length_std              : std dev of message length (low std = suspicious,
                             mirrors the "same length with variance" rule)
    length_cv               : coefficient of variation (std/mean), scale-free
                             version of the variance signal
    interarrival_mean_sec  : mean seconds between consecutive messages
    interarrival_std_sec   : std dev of inter-message gaps (bursty = low std)
    messages_per_recipient : message_count / unique_recipients (fan-out ratio)

Extend this with graph features (shared recipients across users) once you
have cross-user linkage available.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
import config


def compute_window_features(events: pd.DataFrame) -> pd.DataFrame:
    """
    events: raw event-level dataframe (see generate_synthetic_data.py schema)
    Returns one row per incoming SMS.  Each row is a trailing, event-time
    window ending at that SMS; this avoids the blind spot created by
    clock-aligned/tumbling buckets.
    """
    output_columns = [
        "sender_id", "window_start", "window_end", "message_count", "unique_recipients",
        "new_recipient_ratio", "length_mean", "length_std", "length_cv",
        "interarrival_mean_sec", "interarrival_std_sec",
        "messages_per_recipient", "label_fraud",
    ]
    if events.empty:
        return pd.DataFrame(columns=output_columns)

    events = events.copy()
    events["timestamp"] = pd.to_datetime(events["timestamp"])
    events = events.sort_values(["sender_id", "timestamp"])

    rows = []
    window = pd.Timedelta(minutes=config.WINDOW_MINUTES)
    for sender, sender_events in events.groupby("sender_id", sort=False):
        sender_events = sender_events.sort_values("timestamp").reset_index(drop=True)
        left = 0

        # Emit a feature snapshot after every event.  The pointer makes the
        # trailing-window membership explicit and naturally handles sends that
        # straddle a clock boundary.
        for right, event in sender_events.iterrows():
            window_end = event["timestamp"]
            window_start = window_end - window
            while sender_events.at[left, "timestamp"] < window_start:
                left += 1

            grp = sender_events.iloc[left:right + 1]
            n = len(grp)
            unique_recipients = grp["recipient_id"].nunique()
            new_ratio = grp["is_new_recipient"].mean()

            length_mean = grp["message_length"].mean()
            length_std = grp["message_length"].std(ddof=0) if n > 1 else 0.0
            length_cv = (length_std / length_mean) if length_mean else 0.0

            if n > 1:
                gaps = grp["timestamp"].diff().dt.total_seconds().dropna()
                interarrival_mean = gaps.mean()
                interarrival_std = gaps.std(ddof=0)
            else:
                interarrival_mean = np.nan
                interarrival_std = np.nan

            messages_per_recipient = n / unique_recipients if unique_recipients else 0.0

            # label: window is fraud if any event in it is labeled fraud
            # (only present because this is synthetic data with ground truth)
            label = int(grp["label_fraud"].max()) if "label_fraud" in grp else np.nan

            rows.append({
                "sender_id": sender,
                "window_start": window_start,
                "window_end": window_end,
                "message_count": n,
                "unique_recipients": unique_recipients,
                "new_recipient_ratio": new_ratio,
                "length_mean": length_mean,
                "length_std": length_std,
                "length_cv": length_cv,
                "interarrival_mean_sec": interarrival_mean,
                "interarrival_std_sec": interarrival_std,
                "messages_per_recipient": messages_per_recipient,
                "label_fraud": label,
            })

    feature_df = pd.DataFrame(rows, columns=output_columns) if rows else pd.DataFrame(columns=output_columns)
    feature_df[["interarrival_mean_sec", "interarrival_std_sec"]] = (
        feature_df[["interarrival_mean_sec", "interarrival_std_sec"]].fillna(0)
    )
    return feature_df


def main():
    events = pd.read_csv(config.RAW_EVENTS_PATH)
    features = compute_window_features(events)
    features.to_csv(config.FEATURES_PATH, index=False)
    print(f"Computed features for {len(features)} (user, window) rows "
          f"-> {config.FEATURES_PATH}")
    print(features["label_fraud"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
