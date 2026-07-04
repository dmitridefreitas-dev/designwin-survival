"""Episode extraction on constructed paths with hand-known answers."""

import numpy as np
import pandas as pd
import pytest

from survivalsemi.drawdowns import extract_episodes


def make_series(values) -> pd.Series:
    return pd.Series(
        [float(v) for v in values],
        index=pd.bdate_range("2020-01-01", periods=len(values)),
    )


class TestKnownEpisodes:
    def test_single_recovered_episode(self):
        # 100 -> 85 crosses the 10% trigger; recovery when 100 regained.
        close = make_series([100, 95, 85, 90, 101, 102])
        eps = extract_episodes(close, threshold=0.10, window=252)
        assert len(eps) == 1
        ep = eps.iloc[0]
        assert ep.recovered
        assert ep.entry_date == close.index[2]
        assert ep.exit_date == close.index[4]
        assert ep.duration_days == 2
        assert ep.max_depth == pytest.approx(0.15)

    def test_censored_episode(self):
        close = make_series([100, 85, 80, 82])  # never recovers
        eps = extract_episodes(close, threshold=0.10, window=252)
        assert len(eps) == 1
        ep = eps.iloc[0]
        assert not ep.recovered
        assert pd.isna(ep.exit_date)
        assert ep.duration_days == 2  # entry at index 1, censored at index 3
        assert ep.max_depth == pytest.approx(0.20)

    def test_deeper_drop_extends_same_episode(self):
        # One episode even though the price crosses -10% twice before recovery.
        close = make_series([100, 88, 94, 82, 90, 100, 101])
        eps = extract_episodes(close, threshold=0.10, window=252)
        assert len(eps) == 1
        assert eps.iloc[0].max_depth == pytest.approx(0.18)
        assert eps.iloc[0].recovered

    def test_two_separate_episodes(self):
        close = make_series([100, 85, 100, 110, 95, 111])
        eps = extract_episodes(close, threshold=0.10, window=252)
        assert len(eps) == 2
        assert eps.recovered.all()

    def test_no_episode_without_trigger(self):
        close = make_series([100, 95, 97, 99, 100])
        eps = extract_episodes(close, threshold=0.10, window=252)
        assert len(eps) == 0


class TestCovariates:
    def test_market_dd_is_measured_at_entry(self):
        close = make_series([100, 95, 85, 90, 101])
        market = make_series([100, 96, 92, 95, 100])  # 8% down at entry
        eps = extract_episodes(close, threshold=0.10, window=252, market=market)
        assert eps.iloc[0].market_dd == pytest.approx(0.08)

    def test_rolling_window_reference_forgets_old_highs(self):
        # With a 3-day reference window, an old peak stops mattering.
        values = [100] + [98] * 5 + [88] + [99] * 3
        close = make_series(values)
        eps = extract_episodes(close, threshold=0.10, window=3)
        # 88 vs the 3-day high of 98 is a 10.2% gap -> triggers; recovery at 99 >= 98.
        assert len(eps) == 1
        assert eps.iloc[0].recovered
