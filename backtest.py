"""
Stock backtester: 50-day moving average crossover strategy on SPY.
Signal is generated at day t's close; trade executes at day t+1's open.
This one-day gap is how we avoid lookahead bias.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Configuration ─────────────────────────────────────────────────────────────

TICKER = "SPY"
STARTING_CASH = 10_000.0
MA_WINDOW = 50          # days for the moving average
YEARS_BACK = 5          # how far back to pull data

# ── 1. Pull Data ──────────────────────────────────────────────────────────────

print(f"Downloading {YEARS_BACK} years of {TICKER} data...")
raw = yf.download(TICKER, period=f"{YEARS_BACK}y", auto_adjust=True, progress=False)

# yfinance may return a MultiIndex when auto_adjust=True; flatten if so
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)

df = raw[["Open", "Close"]].copy()
df.dropna(inplace=True)

print(f"  {len(df)} trading days loaded ({df.index[0].date()} → {df.index[-1].date()})")

# ── 2. Compute 50-Day Simple Moving Average ───────────────────────────────────

# min_periods=MA_WINDOW means the first MA_WINDOW-1 rows will be NaN,
# so we never trade with an incomplete average.
df["SMA50"] = df["Close"].rolling(window=MA_WINDOW, min_periods=MA_WINDOW).mean()

# ── 3. Generate Signals (end-of-day, based on Close vs SMA50) ────────────────

# +1 = close above SMA → want to be long
# -1 = close below SMA → want to be flat
# 0  = not enough history yet (first 49 rows)
df["Signal"] = 0
df.loc[df["Close"] > df["SMA50"], "Signal"] = 1
df.loc[df["Close"] < df["SMA50"], "Signal"] = -1

# ── 4. Simulate Day-by-Day ────────────────────────────────────────────────────

cash = STARTING_CASH
shares = 0.0

# For tracking
portfolio_values = []   # portfolio value at each day's close
pending_signal = 0      # signal generated yesterday, to be acted on today's open

# For win/loss tracking on completed round trips
buy_price = None        # open price at which we last bought
trade_returns = []      # return of each completed buy→sell round trip
trade_log = []          # human-readable trade log

for i, (date, row) in enumerate(df.iterrows()):
    open_px  = row["Open"]
    close_px = row["Close"]
    signal   = row["Signal"]

    # ── Step A: Execute yesterday's pending signal at TODAY'S OPEN ──────────
    # This is the core of lookahead avoidance: we decided yesterday, we act today.

    if pending_signal == 1 and shares == 0 and cash > 0:
        # BUY: go all-in with available cash
        shares = cash / open_px
        buy_price = open_px
        cash = 0.0
        trade_log.append(f"  BUY  {date.date()}  @ ${open_px:.2f}  ({shares:.4f} shares)")

    elif pending_signal == -1 and shares > 0:
        # SELL: liquidate entire position
        proceeds = shares * open_px
        trade_return = (open_px - buy_price) / buy_price
        trade_returns.append(trade_return)
        trade_log.append(f"  SELL {date.date()}  @ ${open_px:.2f}  return: {trade_return:+.1%}")
        cash = proceeds
        shares = 0.0
        buy_price = None

    # ── Step B: Mark portfolio value at TODAY'S CLOSE ───────────────────────
    # (shares are valued at close; any uninvested cash earns nothing)
    portfolio_values.append(cash + shares * close_px)

    # ── Step C: Record today's signal — will be executed tomorrow ───────────
    # Only update the pending signal when SMA is valid (Signal != 0).
    # If Signal is 0 (not enough history), leave pending_signal unchanged.
    if signal != 0:
        pending_signal = signal

# Store results back in df
df["PortfolioValue"] = portfolio_values

# ── 5. Buy-and-Hold Benchmark ─────────────────────────────────────────────────

# Buy at the open of the very first day, hold to the end.
bah_shares = STARTING_CASH / df["Open"].iloc[0]
df["BuyAndHold"] = bah_shares * df["Close"]

# ── 6. Compute Metrics ────────────────────────────────────────────────────────

final_value   = df["PortfolioValue"].iloc[-1]
total_return  = (final_value - STARTING_CASH) / STARTING_CASH

bah_final     = df["BuyAndHold"].iloc[-1]
bah_return    = (bah_final - STARTING_CASH) / STARTING_CASH

# Number of individual executions (each BUY counts once, each SELL counts once)
n_trades = len([t for t in trade_log])   # each line is one execution

# Win rate: fraction of completed round trips that were profitable
n_wins      = sum(1 for r in trade_returns if r > 0)
win_rate    = n_wins / len(trade_returns) if trade_returns else 0.0

# Max drawdown: largest peak-to-trough drop in strategy portfolio value
rolling_peak = df["PortfolioValue"].cummax()
drawdowns    = (df["PortfolioValue"] - rolling_peak) / rolling_peak
max_drawdown = drawdowns.min()   # most negative value

# ── 7. Print Results ──────────────────────────────────────────────────────────

print("\n" + "=" * 50)
print(f"  BACKTEST RESULTS  |  {TICKER}  |  {YEARS_BACK} years")
print("=" * 50)
print(f"  Starting capital : ${STARTING_CASH:,.2f}")
print(f"  Final value      : ${final_value:,.2f}")
print(f"  Total return     : {total_return:+.2%}")
print(f"  Buy-and-hold     : {bah_return:+.2%}")
print(f"  Outperformance   : {total_return - bah_return:+.2%}")
print(f"  Number of trades : {n_trades}  ({len(trade_returns)} round trips)")
print(f"  Win rate         : {win_rate:.1%}  ({n_wins}/{len(trade_returns)})")
print(f"  Max drawdown     : {max_drawdown:.2%}")
print("=" * 50)

print("\nTrade log:")
for entry in trade_log:
    print(entry)

# ── 8. Chart ──────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]})

# ── Top panel: portfolio value ───────────────────────────────────────────────
ax1 = axes[0]
ax1.plot(df.index, df["PortfolioValue"], label="MA Crossover Strategy", color="#1f77b4", linewidth=1.5)
ax1.plot(df.index, df["BuyAndHold"],     label="Buy & Hold",            color="#ff7f0e", linewidth=1.5, alpha=0.8)

# Shade the drawdown area to make it visible
ax1.fill_between(df.index, df["PortfolioValue"], rolling_peak,
                 where=(df["PortfolioValue"] < rolling_peak),
                 alpha=0.15, color="red", label="Drawdown")

ax1.set_title(f"{TICKER} — MA{MA_WINDOW} Crossover vs Buy & Hold ({YEARS_BACK} years)", fontsize=13)
ax1.set_ylabel("Portfolio Value ($)")
ax1.legend(loc="upper left")
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax1.grid(alpha=0.3)

# Annotate final values
ax1.annotate(f"{total_return:+.1%}", xy=(df.index[-1], final_value),
             xytext=(8, 0), textcoords="offset points", color="#1f77b4", fontweight="bold")
ax1.annotate(f"{bah_return:+.1%}", xy=(df.index[-1], bah_final),
             xytext=(8, 0), textcoords="offset points", color="#ff7f0e", fontweight="bold")

# ── Bottom panel: SPY price + SMA50 ──────────────────────────────────────────
ax2 = axes[1]
ax2.plot(df.index, df["Close"], label="SPY Close", color="gray",    linewidth=0.8, alpha=0.7)
ax2.plot(df.index, df["SMA50"], label="SMA50",     color="purple",  linewidth=1.2)
ax2.set_ylabel("Price ($)")
ax2.legend(loc="upper left", fontsize=8)
ax2.grid(alpha=0.3)

# Shared x-axis formatting
for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())

plt.tight_layout()
chart_path = "backtest_results.png"
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
print(f"\nChart saved to {chart_path}")
plt.show()
