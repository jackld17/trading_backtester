"""
strategy_dsl.py — a small, safe strategy language that the AI writes and the engine runs.

The AI never writes Python. It writes a JSON *spec* (see SPEC_REFERENCE below), which is
validated here and then compiled into entry/exit signals and simulated bar by bar.

Timing rules (no lookahead):
  * Conditions are evaluated at day t's CLOSE using data through day t.
  * Entries/exits from signals execute at day t+1's OPEN.
  * Stop-loss / take-profit / trailing stops trigger intraday using the day's HIGH/LOW
    (filled at the stop level, or at the open if the price gaps through it).
    If a stop and a target are both touched on the same day, the stop is assumed first.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


PRICE_FIELDS = ("open", "high", "low", "close", "volume")

# indicator type -> (required params, optional params with defaults)
INDICATOR_TYPES = {
    "sma":        ({"window"}, {"source": "close"}),
    "ema":        ({"window"}, {"source": "close"}),
    "rsi":        ({"window"}, {"source": "close"}),
    "macd":       (set(), {"fast": 12, "slow": 26, "signal": 9, "field": "macd", "source": "close"}),
    "bbands":     ({"window"}, {"num_std": 2.0, "field": "middle", "source": "close"}),
    "atr":        ({"window"}, {}),
    "highest":    ({"window"}, {"source": "high"}),
    "lowest":     ({"window"}, {"source": "low"}),
    "roc":        ({"window"}, {"source": "close"}),
    "zscore":     ({"window"}, {"source": "close"}),
    "volatility": ({"window"}, {"source": "close"}),
    "gap":        (set(), {}),
}
MACD_FIELDS = {"macd", "signal", "hist"}
BB_FIELDS = {"upper", "middle", "lower", "pctb", "width"}
COMPARE_OPS = {">", "<", ">=", "<=", "crosses_above", "crosses_below"}
MAX_WINDOW = 500

SPEC_REFERENCE = """
STRATEGY SPEC (JSON)
{
  "name": short name,
  "direction": "long" | "short",
  "indicators": [ {"id": "<unique id>", "type": <type>, ...params, "offset": <int, optional>} ],
  "entry": <condition>,
  "exit":  <condition or null>,
  "risk": {"stop_loss_pct": dec|null, "take_profit_pct": dec|null,
           "trailing_stop_pct": dec|null, "max_hold_days": int|null},
  "position_size": fraction of equity per trade, 0 < x <= 1 (default 1.0),
  "costs_bps": cost per side in basis points (default 5)
}

INDICATOR TYPES (all percentages/returns are DECIMALS: 0.05 = 5%)
  sma, ema      window, source            moving averages
  rsi           window, source            0-100 Wilder RSI
  macd          fast, slow, signal, field ("macd"|"signal"|"hist")
  bbands        window, num_std, field ("upper"|"middle"|"lower"|"pctb"|"width")
  atr           window                    average true range, in price units
  highest       window, source (default high)   rolling max
  lowest        window, source (default low)    rolling min
  roc           window, source            % change over window (decimal)
  zscore        window, source            (x - rolling mean) / rolling std
  volatility    window                    annualized std of daily returns (decimal)
  gap           (none)                    open / prior close - 1 (decimal)
  source can be a price field (open, high, low, close, volume) or an earlier indicator id.
  offset N shifts the indicator back N bars, e.g. "yesterday's 20-day high" =
  {"id":"hh20","type":"highest","window":20,"offset":1}.

CONDITIONS
  comparison: {"left": <operand>, "op": ">"|"<"|">="|"<="|"crosses_above"|"crosses_below",
               "right": <operand>}
  operand:    a number, a price field name, an indicator id, or
              {"ref": "<field or id>", "mult": 1.02, "add": 0}   (value = ref*mult + add)
  logic:      {"all": [cond, ...]}  {"any": [cond, ...]}  {"not": cond}

At least one way out is required: an exit condition and/or a risk rule.
"""


class SpecError(ValueError):
    """Raised when a strategy spec is invalid. .errors holds every problem found."""

    def __init__(self, errors):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


# ── Validation ────────────────────────────────────────────────────────────────

def validate_spec(spec: dict) -> dict:
    """Return a normalized deep copy of *spec*, or raise SpecError listing every problem."""
    if not isinstance(spec, dict):
        raise SpecError(["spec must be a JSON object"])
    s = copy.deepcopy(spec)
    errs: list[str] = []

    s.setdefault("name", "Custom strategy")
    s["direction"] = str(s.get("direction") or "long").lower()
    if s["direction"] not in ("long", "short"):
        errs.append("direction must be 'long' or 'short'")

    # indicators
    known = set(PRICE_FIELDS)
    inds = s.get("indicators") or []
    if not isinstance(inds, list):
        errs.append("indicators must be a list")
        inds = []
    norm_inds = []
    for i, ind in enumerate(inds):
        where = f"indicators[{i}]"
        if not isinstance(ind, dict):
            errs.append(f"{where} must be an object")
            continue
        iid, typ = ind.get("id"), str(ind.get("type", "")).lower()
        if not iid or not isinstance(iid, str):
            errs.append(f"{where} needs a string id")
            continue
        if iid in known:
            errs.append(f"{where}: id '{iid}' is duplicated or shadows a price field")
            continue
        if typ not in INDICATOR_TYPES:
            errs.append(f"{where} ('{iid}'): unknown type '{typ}'. Allowed: {sorted(INDICATOR_TYPES)}")
            continue
        required, optional = INDICATOR_TYPES[typ]
        out = {"id": iid, "type": typ}
        for p in required:
            if p not in ind:
                errs.append(f"{where} ('{iid}'): missing '{p}'")
        for p, default in optional.items():
            out[p] = ind.get(p, default)
        for p in required:
            if p in ind:
                out[p] = ind[p]
        for p in ("window", "fast", "slow", "signal"):
            if p in out:
                try:
                    out[p] = int(out[p])
                    if not 1 <= out[p] <= MAX_WINDOW:
                        raise ValueError
                except (TypeError, ValueError):
                    errs.append(f"{where} ('{iid}'): {p} must be an integer 1-{MAX_WINDOW}")
        if typ == "rsi" and out.get("window", 2) < 2:
            errs.append(f"{where} ('{iid}'): rsi window must be >= 2")
        if typ == "macd":
            if out["field"] not in MACD_FIELDS:
                errs.append(f"{where} ('{iid}'): macd field must be one of {sorted(MACD_FIELDS)}")
            if isinstance(out.get("fast"), int) and isinstance(out.get("slow"), int) and out["fast"] >= out["slow"]:
                errs.append(f"{where} ('{iid}'): macd fast must be < slow")
        if typ == "bbands":
            if out["field"] not in BB_FIELDS:
                errs.append(f"{where} ('{iid}'): bbands field must be one of {sorted(BB_FIELDS)}")
            try:
                out["num_std"] = float(out["num_std"])
            except (TypeError, ValueError):
                errs.append(f"{where} ('{iid}'): num_std must be a number")
        src = out.get("source")
        if src is not None:
            src = str(src).lower() if str(src).lower() in PRICE_FIELDS else src
            out["source"] = src
            if src not in known:
                errs.append(f"{where} ('{iid}'): source '{src}' is not a price field or an EARLIER indicator id")
        try:
            out["offset"] = int(ind.get("offset", 0) or 0)
            if not 0 <= out["offset"] <= MAX_WINDOW:
                raise ValueError
        except (TypeError, ValueError):
            errs.append(f"{where} ('{iid}'): offset must be an integer >= 0")
        known.add(iid)
        norm_inds.append(out)
    s["indicators"] = norm_inds

    # conditions
    def check_operand(op, where):
        if isinstance(op, bool):
            errs.append(f"{where}: booleans are not valid operands")
        elif isinstance(op, (int, float)):
            return
        elif isinstance(op, str):
            if op.lower() in PRICE_FIELDS or op in known:
                return
            errs.append(f"{where}: unknown reference '{op}'")
        elif isinstance(op, dict) and "ref" in op:
            check_operand(op["ref"], where)
            for k in ("mult", "add"):
                if k in op and not isinstance(op[k], (int, float)):
                    errs.append(f"{where}: '{k}' must be a number")
        else:
            errs.append(f"{where}: operand must be a number, a name, or {{'ref':...}}")

    def check_cond(c, where):
        if not isinstance(c, dict):
            errs.append(f"{where}: condition must be an object")
            return
        if "all" in c or "any" in c:
            key = "all" if "all" in c else "any"
            items = c[key]
            if not isinstance(items, list) or not items:
                errs.append(f"{where}.{key} must be a non-empty list")
                return
            for j, sub in enumerate(items):
                check_cond(sub, f"{where}.{key}[{j}]")
        elif "not" in c:
            check_cond(c["not"], f"{where}.not")
        else:
            if c.get("op") not in COMPARE_OPS:
                errs.append(f"{where}: op must be one of {sorted(COMPARE_OPS)}")
            for side in ("left", "right"):
                if side not in c:
                    errs.append(f"{where}: missing '{side}'")
                else:
                    check_operand(c[side], f"{where}.{side}")

    if not s.get("entry"):
        errs.append("entry condition is required")
    else:
        check_cond(s["entry"], "entry")
    if s.get("exit"):
        check_cond(s["exit"], "exit")
    else:
        s["exit"] = None

    # risk
    risk = s.get("risk") or {}
    if not isinstance(risk, dict):
        errs.append("risk must be an object")
        risk = {}
    norm_risk = {}
    for k in ("stop_loss_pct", "take_profit_pct", "trailing_stop_pct"):
        v = risk.get(k)
        if v in (None, 0):
            norm_risk[k] = None
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            errs.append(f"risk.{k} must be a number")
            continue
        if not 0 < v < 1:
            errs.append(f"risk.{k} must be a decimal between 0 and 1 (0.05 = 5%), got {v}")
        norm_risk[k] = v
    mh = risk.get("max_hold_days")
    if mh in (None, 0):
        norm_risk["max_hold_days"] = None
    else:
        try:
            norm_risk["max_hold_days"] = int(mh)
            if norm_risk["max_hold_days"] < 1:
                raise ValueError
        except (TypeError, ValueError):
            errs.append("risk.max_hold_days must be a positive integer")
    s["risk"] = norm_risk

    if s["exit"] is None and not any(norm_risk.get(k) for k in norm_risk):
        errs.append("no way out: add an exit condition or at least one risk rule")

    try:
        s["position_size"] = float(s.get("position_size", 1.0) or 1.0)
        if not 0 < s["position_size"] <= 1:
            raise ValueError
    except (TypeError, ValueError):
        errs.append("position_size must be a fraction in (0, 1]")
    try:
        s["costs_bps"] = float(s.get("costs_bps", 5) if s.get("costs_bps") is not None else 5)
        if not 0 <= s["costs_bps"] <= 500:
            raise ValueError
    except (TypeError, ValueError):
        errs.append("costs_bps must be between 0 and 500")

    if errs:
        raise SpecError(errs)
    return s


# ── Indicators ────────────────────────────────────────────────────────────────

def _wilder(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def compute_indicators(df: pd.DataFrame, spec: dict) -> dict[str, pd.Series]:
    """Return {name: series} for price fields and every indicator in the (validated) spec."""
    cols = {f: df[f.capitalize()].astype(float) for f in PRICE_FIELDS if f.capitalize() in df.columns}
    if "volume" not in cols:
        cols["volume"] = pd.Series(np.nan, index=df.index)
    for f in ("open", "high", "low"):
        cols.setdefault(f, cols["close"])

    for ind in spec["indicators"]:
        t, w = ind["type"], ind.get("window")
        src = cols.get(ind.get("source") or "close")
        if t == "sma":
            out = src.rolling(w, min_periods=w).mean()
        elif t == "ema":
            out = src.ewm(span=w, adjust=False, min_periods=w).mean()
        elif t == "rsi":
            d = src.diff()
            g, l = _wilder(d.clip(lower=0), w), _wilder((-d).clip(lower=0), w)
            out = 100 - 100 / (1 + g / l.replace(0, np.nan))
            out = out.where(~((l == 0) & g.notna()), 100.0)
        elif t == "macd":
            fast = src.ewm(span=ind["fast"], adjust=False, min_periods=ind["fast"]).mean()
            slow = src.ewm(span=ind["slow"], adjust=False, min_periods=ind["slow"]).mean()
            line = fast - slow
            sig = line.ewm(span=ind["signal"], adjust=False, min_periods=ind["signal"]).mean()
            out = {"macd": line, "signal": sig, "hist": line - sig}[ind["field"]]
        elif t == "bbands":
            mid = src.rolling(w, min_periods=w).mean()
            sd = src.rolling(w, min_periods=w).std()
            up, lo = mid + ind["num_std"] * sd, mid - ind["num_std"] * sd
            out = {"upper": up, "middle": mid, "lower": lo,
                   "pctb": (src - lo) / (up - lo), "width": (up - lo) / mid}[ind["field"]]
        elif t == "atr":
            prev = cols["close"].shift(1)
            tr = pd.concat([cols["high"] - cols["low"], (cols["high"] - prev).abs(),
                            (cols["low"] - prev).abs()], axis=1).max(axis=1)
            out = _wilder(tr, w)
        elif t == "highest":
            out = src.rolling(w, min_periods=w).max()
        elif t == "lowest":
            out = src.rolling(w, min_periods=w).min()
        elif t == "roc":
            out = src.pct_change(w)
        elif t == "zscore":
            m, sd = src.rolling(w, min_periods=w).mean(), src.rolling(w, min_periods=w).std()
            out = (src - m) / sd.replace(0, np.nan)
        elif t == "volatility":
            out = src.pct_change().rolling(w, min_periods=w).std() * math.sqrt(252)
        elif t == "gap":
            out = cols["open"] / cols["close"].shift(1) - 1
        else:  # pragma: no cover — validation prevents this
            raise SpecError([f"unsupported indicator {t}"])
        if ind.get("offset"):
            out = out.shift(ind["offset"])
        cols[ind["id"]] = out.astype(float)
    return cols


def warmup_bars(spec: dict) -> int:
    """Rough number of bars before every indicator has a value."""
    n = 0
    for ind in spec["indicators"]:
        w = ind.get("window") or 0
        if ind["type"] == "macd":
            w = ind["slow"] + ind["signal"]
        n = max(n, w + ind.get("offset", 0))
    return n


# ── Conditions ────────────────────────────────────────────────────────────────

def _operand(op, cols, index):
    if isinstance(op, (int, float)) and not isinstance(op, bool):
        return pd.Series(float(op), index=index)
    if isinstance(op, str):
        return cols[op.lower() if op.lower() in PRICE_FIELDS else op]
    base = _operand(op["ref"], cols, index)
    return base * float(op.get("mult", 1.0)) + float(op.get("add", 0.0))


def evaluate(cond, cols, index) -> pd.Series:
    """Evaluate a condition tree to a boolean Series (NaN comparisons are False)."""
    if cond is None:
        return pd.Series(False, index=index)
    if "all" in cond:
        out = pd.Series(True, index=index)
        for c in cond["all"]:
            out &= evaluate(c, cols, index)
        return out
    if "any" in cond:
        out = pd.Series(False, index=index)
        for c in cond["any"]:
            out |= evaluate(c, cols, index)
        return out
    if "not" in cond:
        inner = cond["not"]
        valid = _validity(inner, cols, index)
        return (~evaluate(inner, cols, index)) & valid
    a, b, op = _operand(cond["left"], cols, index), _operand(cond["right"], cols, index), cond["op"]
    if op == ">":
        return (a > b).fillna(False)
    if op == "<":
        return (a < b).fillna(False)
    if op == ">=":
        return (a >= b).fillna(False)
    if op == "<=":
        return (a <= b).fillna(False)
    pa, pb = a.shift(1), b.shift(1)
    if op == "crosses_above":
        return ((a > b) & (pa <= pb)).fillna(False)
    return ((a < b) & (pa >= pb)).fillna(False)


def _validity(cond, cols, index) -> pd.Series:
    """True where every operand in *cond* has data (so NOT doesn't fire during warmup)."""
    if "all" in cond or "any" in cond:
        out = pd.Series(True, index=index)
        for c in cond.get("all", cond.get("any")):
            out &= _validity(c, cols, index)
        return out
    if "not" in cond:
        return _validity(cond["not"], cols, index)
    return (_operand(cond["left"], cols, index).notna() & _operand(cond["right"], cols, index).notna())


# ── Simulation ────────────────────────────────────────────────────────────────

@dataclass
class SimResult:
    df: pd.DataFrame                  # input + PortfolioValue, BuyAndHold, Position, Entry, Exit
    trades: list = field(default_factory=list)
    indicators: dict = field(default_factory=dict)

    @property
    def trade_log(self) -> list[str]:
        lines = []
        for t in self.trades:
            side = "BUY " if t["side"] == "long" else "SHORT"
            lines.append(f"  {side} {t['entry_date']}  @ ${t['entry_price']:.2f}")
            if t["exit_date"]:
                lines.append(f"  {'SELL' if t['side']=='long' else 'COVER'} {t['exit_date']}  @ ${t['exit_price']:.2f}"
                             f"  net: {t['return']:+.1%}  ({t['exit_reason']}, {t['bars']}d)")
            else:
                lines.append(f"  (still open at end, net mark {t['return']:+.1%})")
        return lines


def run_spec(df: pd.DataFrame, spec: dict, starting_cash: float = 10_000.0,
             costs_bps: float | None = None) -> SimResult:
    """Simulate a *validated* spec on an OHLC(V) DataFrame indexed by date."""
    cols = compute_indicators(df, spec)
    idx = df.index
    entry_sig = evaluate(spec["entry"], cols, idx).to_numpy()
    exit_sig = evaluate(spec["exit"], cols, idx).to_numpy()

    o, h, l, c = (cols[k].to_numpy() for k in ("open", "high", "low", "close"))
    bps = spec["costs_bps"] if costs_bps is None else costs_bps
    cost_rate = bps / 1e4
    sign = 1 if spec["direction"] == "long" else -1
    size = spec["position_size"]
    risk = spec["risk"]
    sl, tp, ts, mh = (risk.get("stop_loss_pct"), risk.get("take_profit_pct"),
                      risk.get("trailing_stop_pct"), risk.get("max_hold_days"))

    cash, shares = starting_cash, 0.0
    pending, pending_reason = None, ""
    entry_px = entry_i = extreme = None
    trades, equity = [], np.empty(len(idx))
    position = np.zeros(len(idx))
    entries = np.zeros(len(idx), dtype=bool)
    exits = np.zeros(len(idx), dtype=bool)

    def _trade_returns(e_px, x_px):
        """Gross price return, and net return after paying costs on both sides.
        Net is P&L over entry notional, so with 100% sizing equity grows by exactly (1 + net)."""
        gross = sign * (x_px - e_px) / e_px
        return gross, gross - cost_rate * (1 + x_px / e_px)

    def close_trade(i, px, reason):
        nonlocal cash, shares, entry_px, entry_i, extreme
        cash += shares * px - abs(shares * px) * cost_rate
        gross, net = _trade_returns(entry_px, px)
        trades[-1].update(exit_date=str(idx[i].date()), exit_price=float(px), exit_reason=reason,
                          bars=int(i - entry_i), gross_return=float(gross), **{"return": float(net)})
        shares, entry_px, entry_i, extreme = 0.0, None, None, None
        exits[i] = True

    for i in range(len(idx)):
        # A) execute yesterday's decision at today's open
        if pending == "enter" and shares == 0 and not math.isnan(o[i]):
            eq = cash
            shares = sign * eq * size / o[i]
            cash -= shares * o[i] + abs(shares * o[i]) * cost_rate
            entry_px, entry_i, extreme = o[i], i, o[i]
            entries[i] = True
            trades.append({"side": spec["direction"], "entry_date": str(idx[i].date()),
                           "entry_price": float(o[i]), "exit_date": None, "exit_price": None,
                           "exit_reason": None, "bars": None, "return": None})
        elif pending == "exit" and shares != 0:
            close_trade(i, o[i], pending_reason)
        pending = None

        # B) intraday risk checks
        if shares != 0:
            # Stop levels use the best price seen through YESTERDAY (extreme), then
            # today's high/low is folded in only if no exit happened.
            sl_lvl = entry_px * (1 - sign * sl) if sl else None
            ts_lvl = extreme * (1 - sign * ts) if ts else None
            tp_lvl = entry_px * (1 + sign * tp) if tp else None
            levels = [x for x in (sl_lvl, ts_lvl) if x is not None]
            stop_lvl = (max(levels) if sign > 0 else min(levels)) if levels else None
            gap_ok = i != entry_i            # today's open was a fill, not a gap, on entry day
            if stop_lvl is not None and (l[i] <= stop_lvl if sign > 0 else h[i] >= stop_lvl):
                reason = "trailing stop" if ts_lvl is not None and stop_lvl == ts_lvl and stop_lvl != sl_lvl else "stop loss"
                fill = (min(o[i], stop_lvl) if sign > 0 else max(o[i], stop_lvl)) if gap_ok else stop_lvl
                close_trade(i, fill, reason)
            elif tp_lvl is not None and (h[i] >= tp_lvl if sign > 0 else l[i] <= tp_lvl):
                fill = (max(o[i], tp_lvl) if sign > 0 else min(o[i], tp_lvl)) if gap_ok else tp_lvl
                close_trade(i, fill, "take profit")
            else:
                extreme = max(extreme, h[i]) if sign > 0 else min(extreme, l[i])

        # C) mark to market
        equity[i] = cash + shares * c[i]
        position[i] = np.sign(shares)

        # D) decide for tomorrow
        if shares != 0:
            if exit_sig[i]:
                pending, pending_reason = "exit", "exit signal"
            elif mh and (i - entry_i + 1) >= mh:
                pending, pending_reason = "exit", "time stop"
        elif entry_sig[i]:
            pending = "enter"   # also allows re-entry the bar after a stop-out

    if shares != 0 and trades:
        gross, net = _trade_returns(entry_px, c[-1])        # marked as if closed at the last close
        trades[-1]["return"], trades[-1]["gross_return"] = float(net), float(gross)
        trades[-1]["bars"] = int(len(idx) - 1 - entry_i)

    out = df.copy()
    out["PortfolioValue"] = equity
    out["BuyAndHold"] = starting_cash / o[0] * c
    out["Position"] = position
    out["Entry"] = entries
    out["Exit"] = exits
    return SimResult(df=out, trades=trades, indicators=cols)


# ── Plain-English rendering (deterministic — shows exactly what was run) ──────

def _ind_label(ind):
    t = ind["type"]
    src = "" if ind.get("source") in (None, "close") or t in ("highest", "lowest") else f" of {ind['source']}"
    if t in ("sma", "ema", "rsi", "atr", "roc", "zscore", "volatility"):
        lab = {"roc": "ROC", "zscore": "z-score", "volatility": "volatility"}.get(t, t.upper())
        s = f"{lab}({ind['window']}){src}"
    elif t == "macd":
        s = f"MACD {ind['field']} ({ind['fast']},{ind['slow']},{ind['signal']})"
    elif t == "bbands":
        s = f"Bollinger {ind['field']} ({ind['window']}, {ind['num_std']:g}σ)"
    elif t == "highest":
        s = f"{ind['window']}-day high" + ("" if ind["source"] == "high" else f" of {ind['source']}")
    elif t == "lowest":
        s = f"{ind['window']}-day low" + ("" if ind["source"] == "low" else f" of {ind['source']}")
    else:
        s = "overnight gap"
    if ind.get("offset"):
        s += f" as of {ind['offset']} bar{'s' if ind['offset'] > 1 else ''} ago"
    return s


def describe_spec(spec: dict) -> dict:
    """Return {'entry': str, 'exit': str, 'risk': [str], 'sizing': str} in plain English."""
    labels = {f: f.capitalize() for f in PRICE_FIELDS}
    labels.update({i["id"]: _ind_label(i) for i in spec["indicators"]})

    def opnd(x):
        if isinstance(x, (int, float)):
            return f"{x:g}"
        if isinstance(x, str):
            return labels.get(x.lower(), labels.get(x, x))
        s = opnd(x["ref"])
        m, a = x.get("mult", 1), x.get("add", 0)
        if m != 1:
            s = f"{s} × {m:g}"
        if a:
            s = f"{s} {'+' if a > 0 else '−'} {abs(a):g}"
        return s

    words = {">": "is above", "<": "is below", ">=": "is at or above", "<=": "is at or below",
             "crosses_above": "crosses above", "crosses_below": "crosses below"}

    def cond(c):
        if c is None:
            return "—"
        if "all" in c:
            return " AND ".join(f"({cond(x)})" if ("any" in x) else cond(x) for x in c["all"])
        if "any" in c:
            return " OR ".join(f"({cond(x)})" if ("all" in x) else cond(x) for x in c["any"])
        if "not" in c:
            return f"NOT ({cond(c['not'])})"
        return f"{opnd(c['left'])} {words[c['op']]} {opnd(c['right'])}"

    r = spec["risk"]
    risk = []
    if r.get("stop_loss_pct"):
        risk.append(f"Stop loss {r['stop_loss_pct']:.1%} from entry")
    if r.get("take_profit_pct"):
        risk.append(f"Take profit at +{r['take_profit_pct']:.1%}")
    if r.get("trailing_stop_pct"):
        risk.append(f"Trailing stop {r['trailing_stop_pct']:.1%}")
    if r.get("max_hold_days"):
        risk.append(f"Exit after {r['max_hold_days']} trading days")
    return {
        "direction": spec["direction"],
        "entry": cond(spec["entry"]),
        "exit": cond(spec["exit"]) if spec["exit"] else "No signal exit (risk rules only)",
        "risk": risk,
        "sizing": f"{spec['position_size']:.0%} of equity per trade, {spec['costs_bps']:g} bps cost per side",
    }


# ── Parameter perturbation (for robustness checks) ────────────────────────────

def _refs(c, acc):
    if isinstance(c, dict):
        for key in ("all", "any"):
            for sub in c.get(key, []):
                _refs(sub, acc)
        if "not" in c:
            _refs(c["not"], acc)
        for side in ("left", "right", "ref"):
            v = c.get(side)
            if isinstance(v, str):
                acc.add(v)
            elif isinstance(v, dict):
                _refs(v, acc)
    return acc


def numeric_params(spec: dict, limit: int = 8) -> list[tuple]:
    """List (path, value, label) for the tunable numbers that actually affect the strategy:
    thresholds in the rules first, then risk settings, then lookback windows of used indicators."""
    used = _refs(spec["entry"], set())
    if spec["exit"]:
        _refs(spec["exit"], used)
    by_id = {ind["id"]: ind for ind in spec["indicators"]}
    stack = list(used)
    while stack:                                   # include indicators used as sources
        src = by_id.get(stack.pop(), {}).get("source")
        if src in by_id and src not in used:
            used.add(src)
            stack.append(src)

    thresholds, risk, windows = [], [], []

    def walk(c, path):
        if not isinstance(c, dict):
            return
        for key in ("all", "any"):
            if key in c:
                for j, sub in enumerate(c[key]):
                    walk(sub, path + (key, j))
                return
        if "not" in c:
            walk(c["not"], path + ("not",))
            return
        lhs = c.get("left")
        name = lhs if isinstance(lhs, str) else (lhs.get("ref") if isinstance(lhs, dict) else "value")
        for side in ("left", "right"):
            v = c.get(side)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v != 0:
                thresholds.append((path + (side,), v, f"{path[0]} threshold on {name}"))
            elif isinstance(v, dict) and v.get("mult", 1) != 1:
                thresholds.append((path + (side, "mult"), v["mult"], f"{path[0]} multiplier on {v['ref']}"))

    walk(spec["entry"], ("entry",))
    if spec["exit"]:
        walk(spec["exit"], ("exit",))
    for k in ("stop_loss_pct", "take_profit_pct", "trailing_stop_pct", "max_hold_days"):
        if spec["risk"].get(k):
            risk.append((("risk", k), spec["risk"][k], k.replace("_", " ")))
    for k, ind in enumerate(spec["indicators"]):
        if ind["id"] not in used:
            continue
        for p in ("window", "fast", "slow"):
            if ind.get(p):
                windows.append((("indicators", k, p), ind[p], f"{ind['id']} {p}"))
    return (thresholds + risk + windows)[:limit]


def perturb(spec: dict, path: tuple, factor: float) -> dict:
    s = copy.deepcopy(spec)
    node = s
    for p in path[:-1]:
        node = node[p]
    key = path[-1]
    v = node[key]
    if key == "mult":
        new = 1 + (v - 1) * factor          # perturb the distance from 1, not the multiplier
    elif key in ("window", "fast", "slow", "signal"):
        new = max(2, int(round(v * factor)))         # lookbacks stay whole bars
    elif key == "max_hold_days":
        new = max(1, int(round(v * factor)))
    else:
        # Thresholds and risk percentages: scale the value and keep its sign.
        # (Rounding/clamping here used to turn "z < -2" into "z < 2".)
        new = v * factor
        if float(new).is_integer():
            new = int(new)
    node[key] = new
    return s
