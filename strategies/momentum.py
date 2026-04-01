from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.base import Strategy


class MomentumStrategy(Strategy):
    """Cross-sectional momentum strategy using long/short quintiles."""

    def __init__(
        self,
        lookback_window: int = 63,
        quantiles: int = 5,
        transaction_cost: float = 0.0005,
    ) -> None:
        super().__init__(name="momentum_quintile", transaction_cost=transaction_cost)
        self.lookback_window = lookback_window
        self.quantiles = quantiles

    def generate_signals(self, close: pd.DataFrame) -> pd.DataFrame:
        """Generate portfolio weights based on prior momentum ranks."""
        if close.isna().any().any():
            raise ValueError("Close price data contains NaN values. Clean data before strategy generation.")

        lookback_returns = close.pct_change(periods=self.lookback_window).shift(1)
        rank_pct = lookback_returns.rank(axis=1, method="first", pct=True)

        long_threshold = 1.0 - 1.0 / self.quantiles
        short_threshold = 1.0 / self.quantiles

        signals = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        signals[rank_pct >= long_threshold] = 1.0
        signals[rank_pct <= short_threshold] = -1.0

        weight_scale = signals.abs().sum(axis=1).replace(0.0, np.nan)
        signals = signals.div(weight_scale, axis=0).fillna(0.0)

        return signals
