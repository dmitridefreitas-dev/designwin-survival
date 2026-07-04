"""Schema and loader for a hand-collected design-win announcement corpus.

The motivating research question of this repo: when a semiconductor company
announces a design win (its chip designed into a customer's product), how
long until the win shows up as revenue acceleration — and can the hazard of
"revenue event within t quarters" be priced?

Announcement data cannot be downloaded; it has to be collected by hand from
press releases and investor-relations archives. This module fixes the schema
so collection can start immediately and the analysis drops in unchanged: the
output of `to_survival_table` feeds the same Kaplan-Meier / Cox machinery
the drawdown study validates.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "ticker": "issuer, e.g. MCHP",
    "announce_date": "press-release date, YYYY-MM-DD",
    "customer": "customer or platform named in the release",
    "segment": "end market: auto, industrial, consumer, comms, compute",
    "source_url": "link to the primary source",
}
OPTIONAL_COLUMNS = {
    "revenue_event_date": "date the win was first credited in reported revenue "
                          "(from earnings calls/10-Q); blank = not yet observed",
}

VALID_SEGMENTS = {"auto", "industrial", "consumer", "comms", "compute"}


def load_designwins(path: str | Path) -> pd.DataFrame:
    """Load and validate a design-win corpus CSV. Fails loudly on bad rows."""
    frame = pd.read_csv(path)

    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"corpus is missing required columns: {sorted(missing)}")

    frame["announce_date"] = pd.to_datetime(frame["announce_date"], errors="raise")
    if "revenue_event_date" in frame.columns:
        frame["revenue_event_date"] = pd.to_datetime(
            frame["revenue_event_date"], errors="raise"
        )
    else:
        frame["revenue_event_date"] = pd.NaT

    bad_segments = set(frame["segment"].dropna()) - VALID_SEGMENTS
    if bad_segments:
        raise ValueError(
            f"unknown segments {sorted(bad_segments)}; expected {sorted(VALID_SEGMENTS)}"
        )
    if frame["ticker"].isna().any() or (frame["ticker"].astype(str).str.strip() == "").any():
        raise ValueError("every row needs a ticker")

    inverted = frame["revenue_event_date"].notna() & (
        frame["revenue_event_date"] < frame["announce_date"]
    )
    if inverted.any():
        raise ValueError(
            f"revenue_event_date precedes announce_date on rows {list(frame.index[inverted])}"
        )
    return frame


def to_survival_table(corpus: pd.DataFrame, as_of: str | pd.Timestamp) -> pd.DataFrame:
    """Convert a corpus into (duration, event) survival rows.

    Wins with an observed revenue event have duration = days from
    announcement to that event; wins still waiting are right-censored at
    `as_of`. This is exactly the input shape KaplanMeier.fit and Cox expect.
    """
    as_of = pd.Timestamp(as_of)
    observed = corpus["revenue_event_date"].notna()
    end = corpus["revenue_event_date"].where(observed, as_of)
    duration = (end - corpus["announce_date"]).dt.days

    if (duration < 0).any():
        raise ValueError("as_of predates some announcements")

    return pd.DataFrame({
        "ticker": corpus["ticker"],
        "segment": corpus["segment"],
        "duration_days": duration,
        "event": observed,
    })
