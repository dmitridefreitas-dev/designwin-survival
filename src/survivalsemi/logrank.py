"""Two-sample log-rank test, from scratch.

Null hypothesis: both groups share one hazard function. At every event time
the observed events in group A are compared with the expectation under the
pooled hazard (a hypergeometric draw of that time's events from that time's
risk set); the standardised sum is chi-squared with 1 degree of freedom.
Cross-validated against lifelines in tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2


@dataclass(frozen=True)
class LogRankResult:
    statistic: float
    p_value: float


def logrank_test(durations_a, events_a, durations_b, events_b) -> LogRankResult:
    da = np.asarray(durations_a, dtype=float)
    ea = np.asarray(events_a, dtype=bool)
    db = np.asarray(durations_b, dtype=float)
    eb = np.asarray(events_b, dtype=bool)

    all_event_times = np.unique(np.concatenate([da[ea], db[eb]]))

    observed_minus_expected = 0.0
    variance = 0.0
    for t in all_event_times:
        n_a = int((da >= t).sum())
        n_b = int((db >= t).sum())
        d_a = int(((da == t) & ea).sum())
        d_b = int(((db == t) & eb).sum())
        n = n_a + n_b
        d = d_a + d_b
        if n_a == 0 or n_b == 0 or d == 0:
            continue
        expected_a = d * n_a / n
        observed_minus_expected += d_a - expected_a
        if n > 1:
            variance += d * (n_a / n) * (n_b / n) * (n - d) / (n - 1)

    if variance == 0.0:
        return LogRankResult(statistic=0.0, p_value=1.0)

    statistic = observed_minus_expected**2 / variance
    return LogRankResult(statistic=float(statistic),
                         p_value=float(chi2.sf(statistic, df=1)))
