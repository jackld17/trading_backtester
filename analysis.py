"""
analysis.py — performance metrics, robustness diagnostics and the chart for DSL strategies.

Everything here is deterministic: the AI analyst reads these numbers, it never makes them up.
"""

from __future__ import annotations

import base64
import io
import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import strategy_dsl as dsl

TRADING_DAYS = 252


# ── Metrics ───────────────────────────────────────────────────────────────────

def _series_stats(values: pd.Series) -> dict:
    values = values.dropna()
    rets = values.pct_change().dropna()
    if len(values) < 2:
        return {"total_return": 0.0, "cagr": 0.0, "vol": 0.0, "sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0}
    total = values.iloc[-1] / values.iloc[0] - 1
    years = max((values.index[-1] - values.index[0]).days / 365.25, 1 / 365.25)
    cagr = (1 + total) ** (1 / years) - 1 if total > -1 else -1.0
    sd = rets.std()
    down = rets[rets < 0].std()
    dd = values / values.cummax() - 1
    return {
        "total_return": float(total),
        "cagr": float(cagr),
        "vol": float(sd * math.sqrt(TRADING_DAYS)) if sd == sd else 0.0,
        "sharpe": float(rets.mean() / sd * math.sqrt(TRADING_DAYS)) if sd and sd == sd else 0.0,
        "sortino": float(rets.mean() / down * math.sqrt(TRADING_DAYS)) if down and down == down else 0.0,
        "max_drawdown": float(dd.min()),
    }


def compute_metrics(res: dsl.SimResult, starting_cash: float = 10_000.0) -> dict:
    df = res.df
    pv, bh = df["PortfolioValue"], df["BuyAndHold"]
    s, b = _series_stats(pv), _series_stats(bh)

    # drawdown duration (longest run below a prior peak, in trading days)
    under = (pv < pv.cummax()).astype(int)
    runs = under.groupby((under != under.shift()).cumsum()).cumsum()
    dd_days = int((runs * under).max()) if len(runs) else 0

    closed = [t for t in res.trades if t["exit_date"]]
    rets = np.array([t["return"] for t in closed]) if closed else np.array([])
    wins, losses = rets[rets > 0], rets[rets <= 0]
    gross_win, gross_loss = wins.sum(), -losses.sum()

    sr, br = pv.pct_change().dropna(), bh.pct_change().dropna()
    beta = float(sr.cov(br) / br.var()) if br.var() > 0 else 0.0
    corr = float(sr.corr(br)) if sr.std() > 0 and br.std() > 0 else 0.0

    def by_year(series):
        ye = series.groupby(series.index.year).last()
        prev = ye.shift(1)
        prev.iloc[0] = starting_cash
        return ye / prev - 1

    ys, yb = by_year(pv), by_year(bh)
    yearly = [{"year": int(y), "strategy": float(ys[y]), "buy_hold": float(yb[y])} for y in ys.index]

    m = {
        # keys the existing results template already understands
        "final_value": float(pv.iloc[-1]),
        "total_return": s["total_return"],
        "annualized_return": s["cagr"],
        "sharpe_ratio": s["sharpe"],
        "max_drawdown": s["max_drawdown"],
        "bah_return": b["total_return"],
        "outperformance": s["total_return"] - b["total_return"],
        "n_round_trips": len(closed),
        "n_trades": len(res.trades),
        "n_wins": int(len(wins)),
        "win_rate": float(len(wins) / len(closed)) if closed else 0.0,
        "start_date": str(df.index[0].date()),
        "end_date": str(df.index[-1].date()),
        # extended
        "volatility": s["vol"],
        "sortino": s["sortino"],
        "calmar": float(s["cagr"] / abs(s["max_drawdown"])) if s["max_drawdown"] < 0 else 0.0,
        "max_drawdown_days": dd_days,
        "bah_cagr": b["cagr"],
        "bah_sharpe": b["sharpe"],
        "bah_max_drawdown": b["max_drawdown"],
        "exposure": float((df["Position"] != 0).mean()),
        "avg_trade": float(rets.mean()) if len(rets) else 0.0,
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "best_trade": float(rets.max()) if len(rets) else 0.0,
        "worst_trade": float(rets.min()) if len(rets) else 0.0,
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0),
        "avg_bars_held": float(np.mean([t["bars"] for t in closed])) if closed else 0.0,
        "beta_to_underlying": beta,
        "corr_to_underlying": corr,
        "open_position_at_end": bool(res.trades and not res.trades[-1]["exit_date"]),
        "exit_reasons": {r: sum(1 for t in closed if t["exit_reason"] == r) for r in {t["exit_reason"] for t in closed}},
        "yearly": yearly,
    }
    return m


# ── Robustness diagnostics ────────────────────────────────────────────────────

def diagnostics(df: pd.DataFrame, spec: dict, base: dsl.SimResult, metrics: dict,
                split: float = 0.7) -> dict:
    """In/out-of-sample split, cost sensitivity, parameter sensitivity, and red flags."""
    pv = base.df["PortfolioValue"]
    cut = int(len(pv) * split)
    is_stats, oos_stats = _series_stats(pv.iloc[:cut]), _series_stats(pv.iloc[cut - 1:])
    bh = base.df["BuyAndHold"]
    oos_bh = _series_stats(bh.iloc[cut - 1:])
    split_info = {
        "split_date": str(pv.index[cut].date()),
        "in_sample": is_stats,
        "out_of_sample": oos_stats,
        "out_of_sample_buy_hold": oos_bh,
    }

    costs = []
    for bps in sorted({0.0, float(spec["costs_bps"]), 10.0, 25.0}):
        r = base if bps == spec["costs_bps"] else dsl.run_spec(df, spec, costs_bps=bps)
        st = _series_stats(r.df["PortfolioValue"])
        costs.append({"bps": bps, "total_return": st["total_return"], "sharpe": st["sharpe"]})

    sensitivity = []
    for path, value, label in dsl.numeric_params(spec):
        row = {"param": label, "base": value, "variants": []}
        for f in (0.75, 1.25):
            try:
                alt = dsl.validate_spec(dsl.perturb(spec, path, f))
            except dsl.SpecError:
                continue
            node = alt
            for p in path:
                node = node[p]
            st = _series_stats(dsl.run_spec(df, alt).df["PortfolioValue"])
            row["variants"].append({"value": node, "sharpe": st["sharpe"], "total_return": st["total_return"]})
        if row["variants"]:
            sensitivity.append(row)

    flags = red_flags(spec, metrics, split_info, costs, sensitivity, len(df))
    out = {"split": split_info, "costs": costs, "sensitivity": sensitivity, "red_flags": flags}
    out["checks"] = checks(metrics, out)
    return out


def red_flags(spec, m, split_info, costs, sensitivity, n_bars) -> list[str]:
    flags = []
    n = m["n_round_trips"]
    if n == 0:
        flags.append("No completed trades — the rules never fired (or never exited) in this window.")
    elif n < 10:
        flags.append(f"Only {n} completed trades — far too few to separate skill from luck.")
    if m["exposure"] > 0.85 and m["corr_to_underlying"] > 0.9 and spec["direction"] == "long":
        flags.append(f"In the market {m['exposure']:.0%} of the time with {m['corr_to_underlying']:.2f} correlation "
                     "to the stock — results are mostly buy-and-hold beta, not a timing edge.")
    is_s, oos_s = split_info["in_sample"]["sharpe"], split_info["out_of_sample"]["sharpe"]
    if is_s > 0.5 and oos_s < 0.5 * is_s:
        flags.append(f"Sharpe falls from {is_s:.2f} before {split_info['split_date']} to {oos_s:.2f} after — "
                     "performance decays out of sample.")
    base_sh = m["sharpe_ratio"]
    worst_cost = max(costs, key=lambda c: c["bps"])
    if base_sh > 0.3 and worst_cost["sharpe"] < 0.5 * base_sh:
        flags.append(f"Cost-sensitive: Sharpe drops to {worst_cost['sharpe']:.2f} at {worst_cost['bps']:g} bps per side.")
    for row in sensitivity:
        sh = [v["sharpe"] for v in row["variants"]]
        if base_sh > 0.4 and (min(sh) < 0 or base_sh - min(sh) > 0.6):
            flags.append(f"Fragile to '{row['param']}': a ±25% change moves Sharpe from {base_sh:.2f} to {min(sh):.2f}.")
    if m["total_return"] < 0:
        flags.append(f"Loses money overall ({m['total_return']:+.1%}).")
    if m["total_return"] < m["bah_return"] and m["sharpe_ratio"] < m["bah_sharpe"]:
        flags.append("Lags buy-and-hold on both total return and Sharpe.")
    if m["max_drawdown"] < m["bah_max_drawdown"]:
        flags.append("Drawdown is deeper than simply holding the stock.")
    wu = dsl.warmup_bars(spec)
    if wu > 0.25 * n_bars:
        flags.append(f"Indicators need ~{wu} bars to warm up — over a quarter of the test window is unusable.")
    return flags


# ── Chart ─────────────────────────────────────────────────────────────────────

BG, PANEL, GRID, TXT = "#0d1117", "#161b22", "#30363d", "#e6edf3"
OSCILLATORS = {"rsi", "macd", "roc", "zscore", "volatility", "gap", "atr"}


def build_chart(res: dsl.SimResult, spec: dict, ticker: str, label: str) -> str:
    df = res.df
    used = dsl._refs(spec["entry"], set())
    if spec["exit"]:
        dsl._refs(spec["exit"], used)
    inds = [i for i in spec["indicators"] if i["id"] in used]
    overlays = [i for i in inds if i["type"] not in OSCILLATORS and not (i["type"] == "bbands" and i["field"] in ("pctb", "width"))]
    oscs = [i for i in inds if i not in overlays]

    rows = 3 if oscs else 2
    ratios = [3, 2, 1][:rows]
    fig, axes = plt.subplots(rows, 1, figsize=(12, 9 if oscs else 8), sharex=True,
                             gridspec_kw={"height_ratios": ratios})
    fig.patch.set_facecolor(BG)

    ax = axes[0]
    pv, bh = df["PortfolioValue"], df["BuyAndHold"]
    ax.plot(df.index, pv, color="#58a6ff", lw=1.5, label=label)
    ax.plot(df.index, bh, color="#ff7f0e", lw=1.3, alpha=0.8, label="Buy & Hold")
    peak = pv.cummax()
    ax.fill_between(df.index, pv, peak, where=pv < peak, color="#f85149", alpha=0.15, label="Drawdown")
    ax.set_title(f"{ticker} — {label}", fontsize=13, color=TXT)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    ax = axes[1]
    ax.plot(df.index, df["Close"], color="gray", lw=0.8, alpha=0.8, label=f"{ticker} Close")
    palette = ["#bc8cff", "#3fb950", "#d29922"]
    for k, ind in enumerate(overlays[:3]):
        ax.plot(df.index, res.indicators[ind["id"]], lw=1.0, color=palette[k], label=dsl._ind_label(ind))
    ent, ex = df[df["Entry"]], df[df["Exit"]]
    up = spec["direction"] == "long"
    ax.scatter(ent.index, ent["Open"], marker="^" if up else "v", color="#3fb950" if up else "#f85149",
               s=28, zorder=5, label="Entry")
    ax.scatter(ex.index, ex["Close"], marker="x", color=TXT, s=22, zorder=5, label="Exit")

    if oscs:
        ax = axes[2]
        for k, ind in enumerate(oscs[:2]):
            ax.plot(df.index, res.indicators[ind["id"]], lw=1.0, color=palette[k], label=dsl._ind_label(ind))
            if ind["type"] == "rsi":
                ax.set_ylim(0, 100)

    for ax in axes:
        ax.set_facecolor(PANEL)
        ax.grid(color=GRID, lw=0.5)
        ax.tick_params(colors=TXT, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)
        ax.legend(loc="upper left", fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TXT)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def clean(obj):
    """Make a result JSON-safe: NaN/inf -> None, numpy scalars -> Python."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def checks(metrics: dict, diag: dict) -> list[dict]:
    """Boil the robustness tables down to pass / warn / fail tags for the scorecard."""
    out = []
    base = metrics["sharpe_ratio"]

    # First question a desk would ask: did it beat just holding the stock?
    bh_sh = metrics["bah_sharpe"]
    if metrics["total_return"] < 0:
        out.append({"status": "fail", "label": "Loses money"})
    elif base >= bh_sh:
        out.append({"status": "pass", "label": "Beats buy & hold (Sharpe)"})
    elif base >= 0.75 * bh_sh:
        out.append({"status": "warn", "label": "Close to buy & hold"})
    else:
        out.append({"status": "fail", "label": "Trails buy & hold"})

    sp = diag["split"]
    yr = sp["split_date"][:4]
    is_s, oos_s = sp["in_sample"]["sharpe"], sp["out_of_sample"]["sharpe"]
    if is_s <= 0.2:
        st = "pass" if oos_s >= is_s else "warn"
        out.append({"status": st, "label": f"Similar after {yr}" if st == "pass" else f"Weaker after {yr}"})
    else:
        r = oos_s / is_s
        out.append({"status": "pass" if r >= 0.6 else "warn" if r >= 0.3 else "fail",
                    "label": f"Holds up after {yr}" if r >= 0.6 else f"Weaker after {yr}" if r >= 0.3 else f"Fades after {yr}"})

    n = metrics["n_round_trips"]
    out.append({"status": "pass" if n >= 30 else "warn" if n >= 10 else "fail",
                "label": f"Enough trades ({n})" if n >= 30 else f"Few trades ({n})" if n >= 10 else f"Too few trades ({n})"})

    worst = max(diag["costs"], key=lambda c: c["bps"])
    if base > 0.1:
        r = worst["sharpe"] / base
        out.append({"status": "pass" if r >= 0.75 else "warn" if r >= 0.5 else "fail",
                    "label": "Survives higher costs" if r >= 0.75 else "Cost sensitive" if r >= 0.5 else "Costs erase the edge"})

    sens = [v["sharpe"] for row in diag["sensitivity"] for v in row["variants"]]
    if sens and base > 0.1:
        drop = base - min(sens)
        out.append({"status": "pass" if drop <= 0.3 else "warn" if drop <= 0.6 else "fail",
                    "label": "Stable settings" if drop <= 0.3 else "Somewhat fragile settings" if drop <= 0.6 else "Fragile settings"})
    return out
