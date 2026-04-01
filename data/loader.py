from __future__ import annotations

from typing import Sequence

import pandas as pd
import yfinance as yf

DEFAULT_START = "2018-01-01"
DEFAULT_END = "2024-12-31"
DEFAULT_INTERVAL = "1d"


class DataLoader:
    """Download and normalize historical OHLCV equity data from yfinance."""

    def __init__(
        self,
        tickers: Sequence[str],
        start: str = DEFAULT_START,
        end: str = DEFAULT_END,
        interval: str = DEFAULT_INTERVAL,
    ) -> None:
        self.tickers = list(tickers)
        self.start = start
        self.end = end
        self.interval = interval

    def download_data(self) -> pd.DataFrame:
        """Download OHLCV data for the configured tickers and date range."""
        raw_data = yf.download(
            tickers=self.tickers,
            start=self.start,
            end=self.end,
            interval=self.interval,
            group_by="column",
            auto_adjust=False,
            threads=True,
        )

        return self._clean(raw_data)

    def _clean(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Normalize raw yfinance data and fill missing values safely.

        Forward-filling uses only prior values, which helps preserve the
        non-lookahead requirement for daily strategy signals.
        """
        if raw_data.empty:
            raise ValueError(
                "Downloaded data is empty. Check ticker symbols and date range."
            )

        cleaned = raw_data.sort_index().ffill()

        if cleaned.isna().any().any():
            raise ValueError(
                "Downloaded data still contains missing values after forward-fill. "
                "Verify ticker list and market dates."
            )

        return cleaned
