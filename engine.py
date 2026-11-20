"""
engine.py — reusable backtest functions used by both app.py and (optionally) backtest.py.

Design principle: strategy functions only write a Signal column (+1 / -1 / 0) to the
dataframe. The simulation loop is strategy-agnostic; it just reads that column.
Lookahead-bias rule: signal generated at day t's CLOSE, executed at day t+1's OPEN.
"""

import io
import base64

import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for a web server
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


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


# ── Strategies ────────────────────────────────────────────────────────────────

def ma_crossover_signals(df: pd.DataFrame, ma_window: int = 50) -> pd.DataFrame:
    """
    Moving-average crossover strategy.

    Adds two columns to df (in-place) and returns it:
      SMA    — simple moving average of Close over ma_window days
      Signal — +1 (close above SMA → want to be long)
               -1 (close below SMA → want to be flat)
                0 (not enough history yet — first ma_window-1 rows)

    The 0 rows are important: the simulation loop only updates pending_signal
    when Signal != 0, so those early rows simply mean "no opinion yet."
    """
    df = df.copy()

    # min_periods=ma_window means the first ma_window-1 rows are NaN,
    # so we never trade on an incomplete average.
    df["SMA"] = df["Close"].rolling(window=ma_window, min_periods=ma_window).mean()

    df["Signal"] = 0
    df.loc[df["Close"] > df["SMA"], "Signal"] = 1
    df.loc[df["Close"] < df["SMA"], "Signal"] = -1

    return df


def rsi_signals(
    df: pd.DataFrame,
    rsi_window: int = 14,
    oversold: int = 30,
    overbought: int = 70,
) -> pd.DataFrame:
    """
    RSI (Relative Strength Index) momentum strategy.

    How RSI works:
      - Measures the ratio of average up-days to average down-days over rsi_window days.
      - RSI ranges from 0 to 100.
      - Low RSI (< oversold, e.g. 30)  → market is "oversold" → expect a bounce → BUY.
      - High RSI (> overbought, e.g. 70) → market is "overbought" → expect a drop → SELL.
      - Between thresholds → no new opinion → Signal = 0 (hold whatever we decided last).

    Adds two columns to df and returns it:
      RSI    — 0-100 momentum indicator
      Signal — +1 / -1 / 0  (same semantics as ma_crossover_signals)

    We use Wilder smoothing (EWM with com = rsi_window - 1) — this is the standard
    RSI formula.  A simple rolling mean would give slightly different numbers.
    """
    df = df.copy()

    delta = df["Close"].diff()                        # day-over-day price change

    gain = delta.clip(lower=0)                        # positive changes only
    loss = (-delta).clip(lower=0)                     # negative changes (made positive)

    # Wilder smoothing: exponentially weighted mean, equivalent to a rolling
    # average that decays at rate 1/rsi_window.
    avg_gain = gain.ewm(com=rsi_window - 1, min_periods=rsi_window).mean()
    avg_loss = loss.ewm(com=rsi_window - 1, min_periods=rsi_window).mean()

    rs = avg_gain / avg_loss                          # relative strength
    df["RSI"] = 100 - (100 / (1 + rs))

    # Signals — only at the extremes; 0 means "keep whatever pending_signal we have"
    df["Signal"] = 0
    df.loc[df["RSI"] < oversold,   "Signal"] = 1     # oversold → buy
    df.loc[df["RSI"] > overbought, "Signal"] = -1    # overbought → sell

    return df


# ── Simulation ────────────────────────────────────────────────────────────────

def run_simulation(df: pd.DataFrame, starting_cash: float = 10_000.0):
    """
    Day-by-day portfolio simulation.  Strategy-agnostic — it only reads the
    Signal column written by whichever strategy function was called before this.

    Lookahead-bias rule (same as backtest.py):
      Step A: execute yesterday's pending_signal at TODAY'S OPEN
      Step B: record portfolio value at TODAY'S CLOSE
      Step C: update pending_signal from TODAY'S Signal (executed tomorrow)

    Returns:
      trade_log     — list of human-readable strings, one per BUY or SELL execution
      trade_returns — list of floats, one per completed buy→sell round trip
      df            — input df with two new columns: PortfolioValue, BuyAndHold
    """
    df = df.copy()

    cash           = starting_cash
    shares         = 0.0
    pending_signal = 0       # signal decided yesterday, to be acted on at today's open
    buy_price      = None    # open price at which we last bought (for return calculation)

    trade_log     = []       # human-readable record
    trade_returns = []       # return of each completed round trip
    portfolio_values = []

    for date, row in df.iterrows():
        open_px  = row["Open"]
        close_px = row["Close"]
        signal   = row["Signal"]

        # ── Step A: execute yesterday's signal at today's open ────────────────
        if pending_signal == 1 and shares == 0 and cash > 0:
            # BUY: go all-in with available cash
            shares    = cash / open_px
            buy_price = open_px
            cash      = 0.0
            trade_log.append(
                f"  BUY  {date.date()}  @ ${open_px:.2f}  ({shares:.4f} shares)"
            )

        elif pending_signal == -1 and shares > 0:
            # SELL: liquidate entire position
            proceeds      = shares * open_px
            trade_return  = (open_px - buy_price) / buy_price
            trade_returns.append(trade_return)
            trade_log.append(
                f"  SELL {date.date()}  @ ${open_px:.2f}  return: {trade_return:+.1%}"
            )
            cash      = proceeds
            shares    = 0.0
            buy_price = None

        # ── Step B: mark portfolio value at today's close ─────────────────────
        portfolio_values.append(cash + shares * close_px)

        # ── Step C: record today's signal — will execute tomorrow ─────────────
        # Only update when Signal is non-zero; 0 means "no new opinion, keep prior."
        if signal != 0:
            pending_signal = signal

    df["PortfolioValue"] = portfolio_values

    # Buy-and-hold benchmark: buy at the very first open, hold to the end
    bah_shares        = starting_cash / df["Open"].iloc[0]
    df["BuyAndHold"]  = bah_shares * df["Close"]

    return trade_log, trade_returns, df


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(
    df: pd.DataFrame,
    trade_log: list,
    trade_returns: list,
    starting_cash: float = 10_000.0,
) -> dict:
    """
    Summarise a completed simulation into a metrics dictionary.
    All the same formulas as backtest.py lines 109-125.
    """
    final_value  = df["PortfolioValue"].iloc[-1]
    total_return = (final_value - starting_cash) / starting_cash

    bah_final  = df["BuyAndHold"].iloc[-1]
    bah_return = (bah_final - starting_cash) / starting_cash

    n_round_trips = len(trade_returns)
    n_wins        = sum(1 for r in trade_returns if r > 0)
    win_rate      = n_wins / n_round_trips if n_round_trips else 0.0

    rolling_peak = df["PortfolioValue"].cummax()
    drawdowns    = (df["PortfolioValue"] - rolling_peak) / rolling_peak
    max_drawdown = drawdowns.min()

    years             = (df.index[-1] - df.index[0]).days / 365.25
    annualized_return = (1 + total_return) ** (1 / years) - 1

    daily_returns = df["PortfolioValue"].pct_change().dropna()
    std           = daily_returns.std()
    sharpe_ratio  = (daily_returns.mean() / std) * (252 ** 0.5) if std != 0 else 0.0

    return {
        "final_value"      : final_value,
        "total_return"     : total_return,
        "annualized_return": annualized_return,
        "sharpe_ratio"     : sharpe_ratio,
        "bah_return"       : bah_return,
        "outperformance"   : total_return - bah_return,
        "n_trades"         : len(trade_log),
        "n_round_trips"    : n_round_trips,
        "win_rate"         : win_rate,
        "n_wins"           : n_wins,
        "max_drawdown"     : max_drawdown,
        "start_date"       : df.index[0].date(),
        "end_date"         : df.index[-1].date(),
    }


# ── Chart ─────────────────────────────────────────────────────────────────────

def build_chart(df: pd.DataFrame, ticker: str, strategy_name: str) -> str:
    """
    Render the same 2-panel chart as backtest.py but return it as a base64 PNG
    string so the Flask template can embed it directly in an <img> tag:

        <img src="data:image/png;base64,{{ chart_b64 }}">

    Top panel:   strategy portfolio value vs buy-and-hold, with drawdown shading
    Bottom panel: price + whichever indicator column exists (SMA or RSI)
    """
    rolling_peak = df["PortfolioValue"].cummax()
    final_value  = df["PortfolioValue"].iloc[-1]
    bah_final    = df["BuyAndHold"].iloc[-1]
    starting_cash = df["BuyAndHold"].iloc[0] / (df["Close"].iloc[0] / df["Open"].iloc[0])
    total_return = (final_value  - df["PortfolioValue"].iloc[0]) / df["PortfolioValue"].iloc[0]
    bah_return   = (bah_final    - df["BuyAndHold"].iloc[0])     / df["BuyAndHold"].iloc[0]

    # Re-derive returns from the series so the chart annotation is always correct
    total_return = (df["PortfolioValue"].iloc[-1] - df["PortfolioValue"].iloc[0]) / df["PortfolioValue"].iloc[0]
    bah_return   = (df["BuyAndHold"].iloc[-1]     - df["BuyAndHold"].iloc[0])     / df["BuyAndHold"].iloc[0]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor('#0d1117')
    for ax in axes:
        ax.set_facecolor('#161b22')

    # ── Top panel: portfolio value ────────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(df.index, df["PortfolioValue"], label=strategy_name,  color="#1f77b4", linewidth=1.5)
    ax1.plot(df.index, df["BuyAndHold"],     label="Buy & Hold",   color="#ff7f0e", linewidth=1.5, alpha=0.8)
    ax1.fill_between(
        df.index, df["PortfolioValue"], rolling_peak,
        where=(df["PortfolioValue"] < rolling_peak),
        alpha=0.15, color="red", label="Drawdown",
    )
    ax1.set_title(f"{ticker} — {strategy_name} vs Buy & Hold", fontsize=13)
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend(loc="upper left", facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.grid(color='#30363d', linewidth=0.5)
    ax1.annotate(f"{total_return:+.1%}", xy=(df.index[-1], df["PortfolioValue"].iloc[-1]),
                 xytext=(8, 0), textcoords="offset points", color="#1f77b4", fontweight="bold")
    ax1.annotate(f"{bah_return:+.1%}",   xy=(df.index[-1], df["BuyAndHold"].iloc[-1]),
                 xytext=(8, 0), textcoords="offset points", color="#ff7f0e", fontweight="bold")

    # ── Bottom panel: price + indicator (SMA or RSI) ──────────────────────────
    ax2 = axes[1]
    ax2.plot(df.index, df["Close"], label=f"{ticker} Close", color="gray", linewidth=0.8, alpha=0.7)

    if "SMA" in df.columns:
        ax2.plot(df.index, df["SMA"], label="SMA", color="purple", linewidth=1.2)
        ax2.set_ylabel("Price ($)")
    elif "RSI" in df.columns:
        # RSI lives on a 0-100 scale, so put it on a twin axis
        ax2.set_visible(False)
        ax2 = axes[1]
        ax2.set_visible(True)
        ax2.plot(df.index, df["RSI"], label="RSI", color="darkorange", linewidth=1.2)
        ax2.axhline(y=70, color="red",   linestyle="--", linewidth=0.8, alpha=0.7)
        ax2.axhline(y=30, color="green", linestyle="--", linewidth=0.8, alpha=0.7)
        ax2.set_ylabel("RSI")
        ax2.set_ylim(0, 100)

    ax2.legend(loc="upper left", fontsize=8, facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3')
    ax2.grid(color='#30363d', linewidth=0.5)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.tick_params(colors='#e6edf3', labelsize=8)
        ax.yaxis.label.set_color('#e6edf3')
        ax.xaxis.label.set_color('#e6edf3')
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')
    ax1.title.set_color('#e6edf3')

    plt.tight_layout()

    # Write to an in-memory buffer instead of a file
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")
