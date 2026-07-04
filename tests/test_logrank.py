"""Log-rank against lifelines, plus behavioural sanity checks."""

import numpy as np
import pytest
from lifelines.statistics import logrank_test as lifelines_logrank

from survivalsemi.logrank import logrank_test


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_matches_lifelines(seed):
    rng = np.random.default_rng(seed)
    da = rng.exponential(40, 150).round() + 1
    ea = rng.random(150) > 0.25
    db = rng.exponential(60, 120).round() + 1
    eb = rng.random(120) > 0.25

    ours = logrank_test(da, ea, db, eb)
    theirs = lifelines_logrank(da, db, event_observed_A=ea, event_observed_B=eb)

    assert ours.statistic == pytest.approx(theirs.test_statistic, rel=1e-9)
    assert ours.p_value == pytest.approx(theirs.p_value, rel=1e-9)


def test_identical_groups_are_not_distinguished():
    rng = np.random.default_rng(9)
    d = rng.exponential(40, 200).round() + 1
    e = np.ones(200, dtype=bool)
    result = logrank_test(d, e, d, e)
    assert result.statistic == pytest.approx(0.0, abs=1e-12)
    assert result.p_value == pytest.approx(1.0)


def test_clearly_different_hazards_are_detected():
    rng = np.random.default_rng(10)
    fast = rng.exponential(10, 200).round() + 1
    slow = rng.exponential(100, 200).round() + 1
    events = np.ones(200, dtype=bool)
    result = logrank_test(fast, events, slow, events)
    assert result.p_value < 1e-10
