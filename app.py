"""
app.py — Flask web app wrapping the backtest engine.

Routes:
  GET  /           → render the page
  POST /           → manual mode: parse form, run MA/RSI backtest, render results + chart
  POST /interpret  → (legacy) map plain English to the MA/RSI form fields
  POST /analyze    → AI Analyst: plain-English idea (or a follow-up tweak) → spec → backtest
                     → robustness diagnostics → Claude's written review.  Returns JSON.
"""

import json
import os
import re
from datetime import datetime

import anthropic
from flask import Flask, render_template, request, jsonify, abort

import engine
import strategy_dsl as dsl
import analysis
import ai_analyst

app = Flask(__name__)

# System prompt for the legacy /interpret route.
INTERPRET_SYSTEM_PROMPT = """You are a financial strategy parser. Map the user's plain-English description to exactly one of two supported trading strategies and return a JSON object — nothing else. No markdown, no backticks, no explanation.

Supported strategies:
1. MA Crossover: {"strategy": "MA Crossover", "ma_window": <integer>}
2. RSI: {"strategy": "RSI", "rsi_window": <integer>, "oversold": <integer>, "overbought": <integer>}

If the input cannot be confidently mapped to one of these two strategies, return: {"error": "Could not interpret strategy"}

Return raw JSON only."""


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
        "start_date" : "2015-01-01",
        "end_date"   : "2025-01-01",
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

    # ── Run backtest ──────────────────────────────────────────────────────────
    try:
        df = engine.load_data(form["ticker"], form["start_date"], form["end_date"])

        if form["strategy"] == "MA Crossover":
            df = engine.ma_crossover_signals(df, ma_window=form["ma_window"])
            strategy_label = f"MA{form['ma_window']} Crossover"
        else:
            df = engine.rsi_signals(
                df,
                rsi_window=form["rsi_window"],
                oversold=form["oversold"],
                overbought=form["overbought"],
            )
            strategy_label = (
                f"RSI({form['rsi_window']}, "
                f"oversold={form['oversold']}, overbought={form['overbought']})"
            )

        trade_log, trade_returns, df = engine.run_simulation(df)
        metrics   = engine.compute_metrics(df, trade_log, trade_returns)
        chart_b64 = engine.build_chart(df, form["ticker"], strategy_label)

    except ValueError as exc:
        return render_template("index.html", form=form, error=str(exc))

    return render_template(
        "index.html",
        form=form,
        metrics=metrics,
        trade_log=trade_log,
        chart_b64=chart_b64,
        strategy_label=strategy_label,
    )


@app.route("/interpret", methods=["POST"])
def interpret():
    """Legacy: map plain-English text to the MA/RSI form fields."""
    text = request.get_json(force=True).get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        message = client.messages.create(
            model=ai_analyst.MODEL,
            max_tokens=256,
            system=INTERPRET_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": text}],
        )
        raw = message.content[0].text.strip()
        return jsonify(json.loads(raw))
    except json.JSONDecodeError:
        return jsonify({"error": "Model returned a non-JSON response"}), 500
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


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
    start, end = body.get("start_date") or "2015-01-01", body.get("end_date") or "2025-01-01"

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
