"""Kaplan-Meier survival estimator, from scratch.

Right-censoring is the whole reason this estimator exists: an episode that
hasn't ended by the close of the sample still tells you survival lasted *at
least* that long, and throwing it away (or treating it as an event) biases
every duration statistic. KM uses exactly the information each observation
carries: censored subjects leave the risk set without contributing an event.

    S(t) = prod over event times t_i <= t of (1 - d_i / n_i)

where d_i is the number of events at t_i and n_i the number still at risk.
Variance by Greenwood's formula. Cross-validated against lifelines in tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class KaplanMeier:
    """Fitted KM curve. Use `KaplanMeier.fit(durations, events)`."""

    event_times: np.ndarray      # distinct event times, ascending
    survival: np.ndarray         # S(t) at each event time
    std_error: np.ndarray        # Greenwood standard error of S(t)
    at_risk: np.ndarray          # n_i at each event time
    n_events: np.ndarray         # d_i at each event time
    n_observations: int
    n_censored: int

    @classmethod
    def fit(cls, durations, events) -> "KaplanMeier":
        """Fit from durations and event flags (1 = event, 0 = censored)."""
        durations = np.asarray(durations, dtype=float)
        events = np.asarray(events, dtype=bool)
        if durations.shape != events.shape:
            raise ValueError("durations and events must have the same length")
        if len(durations) == 0:
            raise ValueError("no observations")
        if (durations < 0).any():
            raise ValueError("durations must be non-negative")

        order = np.argsort(durations, kind="stable")
        durations, events = durations[order], events[order]

        event_times, survival, std_err, at_risk_list, d_list = [], [], [], [], []
        s = 1.0
        greenwood_sum = 0.0
        n = len(durations)
        i = 0
        while i < n:
            t = durations[i]
            # everyone with duration >= t is still at risk at t
            n_at_risk = n - i
            d = 0
            while i < n and durations[i] == t:
                d += int(events[i])
                i += 1
            if d > 0:
                s *= 1.0 - d / n_at_risk
                if n_at_risk > d:
                    greenwood_sum += d / (n_at_risk * (n_at_risk - d))
                event_times.append(t)
                survival.append(s)
                std_err.append(s * np.sqrt(greenwood_sum))
                at_risk_list.append(n_at_risk)
                d_list.append(d)

        return cls(
            event_times=np.array(event_times),
            survival=np.array(survival),
            std_error=np.array(std_err),
            at_risk=np.array(at_risk_list),
            n_events=np.array(d_list),
            n_observations=n,
            n_censored=int((~events).sum()),
        )

    def survival_at(self, t: float) -> float:
        """S(t): probability the episode lasts longer than t."""
        idx = np.searchsorted(self.event_times, t, side="right") - 1
        return 1.0 if idx < 0 else float(self.survival[idx])

    def median(self) -> float:
        """Smallest event time with S(t) <= 0.5; NaN if the curve never gets there."""
        below = np.nonzero(self.survival <= 0.5)[0]
        return float(self.event_times[below[0]]) if len(below) else float("nan")

    def confidence_band(self, z: float = 1.96) -> pd.DataFrame:
        """Plain Greenwood band, clipped to [0, 1]."""
        return pd.DataFrame({
            "time": self.event_times,
            "survival": self.survival,
            "lower": np.clip(self.survival - z * self.std_error, 0.0, 1.0),
            "upper": np.clip(self.survival + z * self.std_error, 0.0, 1.0),
        })

    def as_step(self) -> pd.DataFrame:
        """Curve including the t=0, S=1 anchor, convenient for step plots."""
        return pd.DataFrame({
            "time": np.concatenate([[0.0], self.event_times]),
            "survival": np.concatenate([[1.0], self.survival]),
        })
