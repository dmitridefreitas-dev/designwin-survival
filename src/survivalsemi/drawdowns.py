"""Turn a price series into right-censored drawdown-recovery episodes.

An episode opens when price closes `threshold` below its reference high
(the trailing `window`-day high by default), and it ends — the "event" —
when price regains the reference high that was in force at entry. Deeper
drops before recovery extend the same episode. An episode still open at the
end of the sample is right-censored: recovery took *at least* that long.

Baseline covariates are measured at episode entry only. Anything measured
after entry (like the eventual maximum depth) is post-baseline information
and must not be used as a Cox covariate — the study notebook demonstrates
why with the trap sprung on purpose.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def extract_episodes(
    close: pd.Series,
    threshold: float = 0.10,
    window: int = 252,
    vol_window: int = 60,
    market: pd.Series | None = None,
) -> pd.DataFrame:
    """Extract drawdown episodes from daily closes.

    Args:
        close: Daily adjusted closes for one ticker.
        threshold: Entry trigger, as a fraction below the reference high.
        window: Lookback for the reference high in trading days.
        vol_window: Lookback for the baseline realised-vol covariate.
        market: Optional market index closes (e.g. SPY) for the baseline
            market-drawdown covariate, aligned by date.

    Returns:
        One row per episode: entry_date, exit_date (NaT if censored),
        duration_days (trading days), recovered (event flag), plus baseline
        covariates trailing_vol and market_dd, and post-baseline max_depth
        (kept only to demonstrate why it must not be a covariate).
    """
    reference_high = close.rolling(window, min_periods=1).max()
    returns = close.pct_change()
    trailing_vol = returns.rolling(vol_window).std(ddof=1) * math.sqrt(TRADING_DAYS)

    market_dd = None
    if market is not None:
        market = market.reindex(close.index).ffill()
        market_dd = market / market.cummax() - 1.0

    episodes = []
    in_episode = False
    entry_idx = entry_high = depth = None

    values = close.to_numpy()
    highs = reference_high.to_numpy()

    for i in range(len(values)):
        if not in_episode:
            if values[i] <= highs[i] * (1.0 - threshold):
                in_episode = True
                entry_idx = i
                entry_high = highs[i]
                depth = values[i] / entry_high - 1.0
        else:
            depth = min(depth, values[i] / entry_high - 1.0)
            if values[i] >= entry_high:
                episodes.append((entry_idx, i, True, depth))
                in_episode = False
    if in_episode:
        episodes.append((entry_idx, len(values) - 1, False, depth))

    rows = []
    for entry_i, exit_i, recovered, max_depth in episodes:
        entry_date = close.index[entry_i]
        rows.append({
            "entry_date": entry_date,
            "exit_date": close.index[exit_i] if recovered else pd.NaT,
            "duration_days": exit_i - entry_i,
            "recovered": recovered,
            "trailing_vol": float(trailing_vol.iloc[entry_i]),
            "market_dd": float(-market_dd.iloc[entry_i]) if market_dd is not None else np.nan,
            "max_depth": float(-max_depth),
        })
    return pd.DataFrame(rows)


def episodes_for_universe(
    prices: dict[str, pd.Series],
    market: pd.Series,
    threshold: float = 0.10,
    window: int = 252,
) -> pd.DataFrame:
    """Stack episodes for several tickers into one labelled table."""
    frames = []
    for ticker, close in prices.items():
        eps = extract_episodes(close, threshold=threshold, window=window, market=market)
        eps.insert(0, "ticker", ticker)
        frames.append(eps)
    return pd.concat(frames, ignore_index=True)
