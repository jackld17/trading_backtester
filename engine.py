"""
engine.py — market data loading (with a per-day in-memory cache).

Both the Manual form and the AI Analyst now run through strategy_dsl.run_spec, so there is
one simulator, one cost model and one set of metrics. The old MA/RSI-only simulation that
used to live here (zero costs, slightly different RSI warm-up) has been removed; see git
history if you need it.
"""

import threading
from datetime import date

import yfinance as yf
import pandas as pd

# ── Data ──────────────────────────────────────────────────────────────────────
# Each ticker's full history is downloaded once per day and kept in memory, then sliced
# per request. Re-runs and refinements on the same ticker make no network calls.
# The cache resets when the app restarts, and each entry refreshes on the first request
# of a new day (adjusted prices change when dividends are paid).

_CACHE: dict[str, tuple[date, pd.DataFrame]] = {}
_LOCK = threading.Lock()


def _download_history(ticker: str) -> pd.DataFrame:
    raw = yf.download(ticker, period="max", auto_adjust=True, progress=False)

    # yfinance sometimes returns a MultiIndex when auto_adjust=True; flatten it
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # High/Low feed stops & ATR, Volume feeds volume rules in the AI analyst
    cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in raw.columns]
    return raw[cols].dropna()


def _history(ticker: str) -> pd.DataFrame:
    today = date.today()
    with _LOCK:
        hit = _CACHE.get(ticker)
        if hit and hit[0] == today:
            return hit[1]
    df = _download_history(ticker)
    if not df.empty:                    # never cache a failed / empty download
        with _LOCK:
            _CACHE[ticker] = (today, df)
    return df


def clear_cache() -> None:
    with _LOCK:
        _CACHE.clear()


def load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Daily OHLCV for *ticker* from *start* (inclusive) to *end* (exclusive), YYYY-MM-DD strings,
    indexed by date — same contract as yf.download(start=, end=).
    Raises ValueError if nothing comes back (bad ticker or date range).
    """
    ticker = ticker.strip().upper()
    hist = _history(ticker)
    df = hist.iloc[0:0] if hist.empty else \
        hist[(hist.index >= pd.Timestamp(start)) & (hist.index < pd.Timestamp(end))].copy()

    if df.empty:
        raise ValueError(
            f"No data returned for '{ticker}' between {start} and {end}. "
            "Check the ticker symbol and date range."
        )

    return df
