"""
app.py — Flask web app wrapping the backtest engine.

Routes:
  GET  /           → render the page
  POST /           → manual mode: MA/RSI form → DSL spec → same engine as /analyze
  POST /analyze    → AI Analyst: plain-English idea (or a follow-up tweak) → spec → backtest
                     → robustness diagnostics → Claude's written review.  Returns JSON.
"""

import json
import os
import re
from datetime import datetime

from flask import Flask, render_template, request, jsonify, abort

import engine
import strategy_dsl as dsl
import analysis
import ai_analyst

app = Flask(__name__)

DEFAULT_START = "2015-01-01"


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _check_dates(start: str, end: str) -> None:
    """Raise ValueError with a readable message unless start/end are valid YYYY-MM-DD, start < end."""
    try:
        s, e = datetime.strptime(start, "%Y-%m-%d"), datetime.strptime(end, "%Y-%m-%d")
    except (TypeError, ValueError):
        raise ValueError(f"Dates must be YYYY-MM-DD (got '{start}' and '{end}').")
    if s >= e:
        raise ValueError("Start date must be before end date.")


def manual_spec(form: dict) -> tuple[dict, str]:
    """Turn the Manual form into a DSL spec, so it runs on the same engine as the AI Analyst.

    MA Crossover: long while Close > SMA(n), flat when Close < SMA(n).
    RSI:          buy when RSI < oversold, sell when RSI > overbought.
    Both use the default 5 bps per side, matching AI mode.
    """
    if form["strategy"] == "MA Crossover":
        w = form["ma_window"]
        spec = {"name": f"MA{w} Crossover", "direction": "long",
                "indicators": [{"id": "sma", "type": "sma", "window": w}],
                "entry": {"left": "close", "op": ">", "right": "sma"},
                "exit": {"left": "close", "op": "<", "right": "sma"}}
    else:
        w, lo, hi = form["rsi_window"], form["oversold"], form["overbought"]
        spec = {"name": f"RSI({w}, oversold={lo}, overbought={hi})", "direction": "long",
                "indicators": [{"id": "rsi", "type": "rsi", "window": w}],
                "entry": {"left": "rsi", "op": "<", "right": lo},
                "exit": {"left": "rsi", "op": ">", "right": hi}}
    spec = dsl.validate_spec(spec)
    return spec, f"{spec['name']} · {spec['costs_bps']:g} bps/side"

@app.route("/", methods=["GET", "POST"])
def index():
    # Default form values — shown on first load and re-populated after an error
    form = {
        "ticker"     : "SPY",
        "strategy"   : "MA Crossover",
        "ma_window"  : 50,
        "rsi_window" : 14,
        "oversold"   : 30,
        "overbought" : 70,
        "start_date" : DEFAULT_START,
        "end_date"   : _today(),
    }

    if request.method == "GET":
        return render_template("index.html", form=form)

    # ── Parse form ────────────────────────────────────────────────────────────
    form["ticker"]     = request.form.get("ticker", "SPY").strip().upper()
    form["strategy"]   = request.form.get("strategy", "MA Crossover")
    form["start_date"] = request.form.get("start_date", form["start_date"])
    form["end_date"]   = request.form.get("end_date",   form["end_date"])

    try:
        form["ma_window"]  = int(request.form.get("ma_window",  50))
        form["rsi_window"] = int(request.form.get("rsi_window", 14))
        form["oversold"]   = int(request.form.get("oversold",   30))
        form["overbought"] = int(request.form.get("overbought", 70))
    except ValueError:
        return render_template("index.html", form=form,
                               error="Strategy parameters must be whole numbers.")

    # ── Run backtest (same engine, costs and metrics as the AI Analyst) ───────
    try:
        _check_dates(form["start_date"], form["end_date"])
        spec, strategy_label = manual_spec(form)
        df = engine.load_data(form["ticker"], form["start_date"], form["end_date"])
        if len(df) <= dsl.warmup_bars(spec) + 20:
            raise ValueError(f"Only {len(df)} bars of data, but the indicator needs "
                             f"~{dsl.warmup_bars(spec)} to warm up. Use a longer date range.")
        res       = dsl.run_spec(df, spec)
        metrics   = analysis.compute_metrics(res)
        trade_log = res.trade_log
        chart_b64 = analysis.build_chart(res, spec, form["ticker"], spec["name"])

    except ValueError as exc:            # includes dsl.SpecError (e.g. window out of range)
        return render_template("index.html", form=form, error=str(exc))

    return render_template(
        "index.html",
        form=form,
        metrics=metrics,
        trade_log=trade_log,
        chart_b64=chart_b64,
        strategy_label=strategy_label,
    )




@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Body (JSON):
      text          plain-English idea, or a follow-up change when current_spec is sent
      ticker, start_date, end_date
      current_spec  (optional) the spec being refined
      spec          (optional) run this exact spec, skipping the AI interpreter
      review        (optional, default true) ask Claude for the written review
    """
    body = request.get_json(force=True) or {}
    text = (body.get("text") or "").strip()
    ticker = (body.get("ticker") or "SPY").strip().upper()
    start, end = body.get("start_date") or DEFAULT_START, body.get("end_date") or _today()

    try:
        # 1) idea -> spec
        if body.get("spec"):
            spec = dsl.validate_spec(body["spec"])
            interp = {"status": "ok", "interpretation": "Ran your edited spec as-is.", "assumptions": []}
        else:
            if not text:
                return jsonify({"status": "error", "error": "Describe a strategy first."}), 400
            interp = ai_analyst.translate(text, current_spec=body.get("current_spec"))
            if interp.get("status") != "ok":
                return jsonify(analysis.clean(interp))
            spec = interp["spec"]
            ticker = (interp.get("ticker") or ticker).strip().upper()
            start = interp.get("start_date") or start
            end = interp.get("end_date") or end

        # 2) data + backtest + diagnostics
        _check_dates(start, end)
        df = engine.load_data(ticker, start, end)
        if len(df) <= dsl.warmup_bars(spec) + 20:
            return jsonify({"status": "error", "spec": spec,
                            "error": f"Only {len(df)} bars of data, but the indicators need "
                                     f"~{dsl.warmup_bars(spec)} to warm up. Use a longer date range."}), 400
        res = dsl.run_spec(df, spec)
        metrics = analysis.compute_metrics(res)
        diag = analysis.diagnostics(df, spec, res, metrics)
        rules = dsl.describe_spec(spec)

        # 3) Claude's review (the backtest still returns if this part fails)
        report, report_error = None, None
        if body.get("review", True):
            try:
                report = ai_analyst.review(text or "(edited spec)", ticker, spec, rules,
                                           metrics, diag, res.trades)
            except ai_analyst.AIError as exc:
                report_error = str(exc)

        return jsonify(analysis.clean({
            "status": "ok",
            "interpretation": interp.get("interpretation"),
            "assumptions": interp.get("assumptions", []),
            "spec": spec,
            "rules": rules,
            "ticker": ticker, "start_date": start, "end_date": end,
            "metrics": metrics,
            "diagnostics": diag,
            "report": report,
            "report_error": report_error,
            "chart_b64": analysis.build_chart(res, spec, ticker, spec.get("name", "Strategy")),
            "trade_log": res.trade_log[-300:],
        }))

    except dsl.SpecError as exc:
        return jsonify({"status": "error", "error": "Invalid spec: " + "; ".join(exc.errors)}), 400
    except ai_analyst.AIError as exc:
        return jsonify({"status": "error", "error": str(exc)}), 502
    except ValueError as exc:            # bad ticker / empty data
        return jsonify({"status": "error", "error": str(exc)}), 400


# ── Saved reports ─────────────────────────────────────────────────────────────
# Each saved test is one JSON file in ./reports/ (spec, metrics, diagnostics, review, chart).
# Nothing leaves your machine; delete the folder to wipe them all.

REPORTS_DIR = os.path.join(app.root_path, "reports")
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,120}$")


def _report_path(rid):
    if not _ID_RE.match(rid):
        abort(404)
    path = os.path.join(REPORTS_DIR, rid + ".json")
    if not os.path.exists(path):
        abort(404)
    return path


@app.route("/reports", methods=["GET"])
def list_reports():
    out = []
    if os.path.isdir(REPORTS_DIR):
        for name in os.listdir(REPORTS_DIR):
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(REPORTS_DIR, name)) as f:
                    r = json.load(f)
            except (OSError, ValueError):
                continue
            res, m = r.get("result", {}), r.get("result", {}).get("metrics", {})
            out.append({
                "id": r["id"], "title": r.get("title"), "saved_at": r.get("saved_at"),
                "ticker": res.get("ticker"), "strategy": res.get("spec", {}).get("name"),
                "total_return": m.get("total_return"), "sharpe": m.get("sharpe_ratio"),
                "verdict": (res.get("report") or {}).get("verdict"),
            })
    out.sort(key=lambda x: x["saved_at"] or "", reverse=True)
    return jsonify(out)


@app.route("/reports", methods=["POST"])
def save_report():
    body = request.get_json(force=True) or {}
    result = body.get("result")
    if not result or result.get("status") != "ok":
        return jsonify({"error": "Nothing to save — run an analysis first."}), 400
    title = (body.get("title") or result.get("spec", {}).get("name") or "Backtest").strip()[:120]
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:50] or "report"
    now = datetime.now()
    rid = f"{now:%Y%m%d-%H%M%S}-{slug}"
    record = {"id": rid, "title": title, "notes": (body.get("notes") or "").strip(),
              "idea": body.get("idea") or "", "saved_at": now.isoformat(timespec="seconds"),
              "result": result}
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(os.path.join(REPORTS_DIR, rid + ".json"), "w") as f:
        json.dump(record, f)
    return jsonify({"id": rid, "title": title})


def _load_report(rid):
    with open(_report_path(rid)) as f:
        r = json.load(f)
    d = r.get("result", {})
    if "checks" not in d.get("diagnostics", {}):          # reports saved before the scorecard
        d["diagnostics"]["checks"] = analysis.checks(d["metrics"], d["diagnostics"])
    return r


@app.route("/reports/<rid>", methods=["GET"])
def get_report(rid):
    return jsonify(_load_report(rid))


@app.route("/reports/<rid>", methods=["DELETE"])
def delete_report(rid):
    os.remove(_report_path(rid))
    return jsonify({"deleted": rid})


@app.route("/reports/<rid>/view")
def view_report(rid):
    """Standalone, printable page (Cmd+P → Save as PDF)."""
    r = _load_report(rid)
    return render_template("report.html", r=r, d=r["result"])


if __name__ == "__main__":
    # debug=True → auto-reloads on file save, shows tracebacks in browser
    # PORT is set by Backtester.command (5000 clashes with macOS AirPlay and your networking tracker)
    app.run(debug=True, port=int(os.environ.get("PORT", 5000)))
