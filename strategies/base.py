from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class Strategy(ABC):
    """Abstract base class for portfolio construction and return calculation."""

    def __init__(self, name: str, transaction_cost: float = 0.0005) -> None:
        self.name = name
        self.transaction_cost = transaction_cost

    @abstractmethod
    def generate_signals(self, close: pd.DataFrame) -> pd.DataFrame:
        """Generate daily portfolio weights from historical close prices.

        The returned DataFrame must have the same shape as `close` and use
        weights per ticker. Long weights should be positive and short weights
        negative.
        """
        raise NotImplementedError

    def compute_returns(self, close: pd.DataFrame, positions: pd.DataFrame) -> pd.Series:
        """Calculate strategy returns from positions and daily close prices.

        Positions are shifted one day to avoid lookahead bias: the portfolio
        held on day T is based on signals generated from data available before T.
        """
        price_returns = close.pct_change().fillna(0.0)
        held_positions = positions.shift(1).fillna(0.0)

        strategy_returns = (held_positions * price_returns).sum(axis=1)

        turnover = held_positions.diff().abs().sum(axis=1).fillna(0.0)
        transaction_costs = turnover * self.transaction_cost

        return pd.Series(strategy_returns - transaction_costs, index=close.index, name=self.name)
