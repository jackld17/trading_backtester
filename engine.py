"""
engine.py — market data loading.

Both the Manual form and the AI Analyst now run through strategy_dsl.run_spec, so there is
one simulator, one cost model and one set of metrics. The old MA/RSI-only simulation that
used to live here (zero costs, slightly different RSI warm-up) has been removed; see git
history if you need it.
"""

import yfinance as yf
import pandas as pd

# ── Data ──────────────────────────────────────────────────────────────────────

def load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download OHLC data for *ticker* between *start* and *end* (YYYY-MM-DD strings).
    Returns a DataFrame with Open and Close columns, indexed by date.
    Raises ValueError if the download comes back empty (bad ticker or date range).
    """
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    # yfinance sometimes returns a MultiIndex when auto_adjust=True; flatten it
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # High/Low feed stops & ATR, Volume feeds volume rules in the AI analyst
    cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in raw.columns]
    df = raw[cols].copy()
    df.dropna(inplace=True)

    if df.empty:
        raise ValueError(
            f"No data returned for '{ticker}' between {start} and {end}. "
            "Check the ticker symbol and date range."
        )

    return df
