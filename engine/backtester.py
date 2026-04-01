from __future__ import annotations

import pandas as pd

from strategies.base import Strategy


def run_backtest(strategy: Strategy, close: pd.DataFrame) -> pd.Series:
    """Run a strategy on close price data and return the daily strategy returns."""
    positions = strategy.generate_signals(close)
    return strategy.compute_returns(close, positions)
