"""
ai_analyst.py — Claude turns a plain-English idea into a strategy spec, then reviews the results.

Two calls per analysis:
  1. translate()  idea (or a follow-up tweak) -> validated JSON spec   [tool use, auto-repaired]
  2. review()     spec + metrics + robustness diagnostics -> analyst report  [tool use]

The model never writes code and never sees anything it could execute. Every number in the
report comes from analysis.py; the prompt tells Claude to only interpret those numbers.
"""

from __future__ import annotations

import json
import os

import anthropic

import strategy_dsl as dsl
from analysis import clean

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-5-5")
MAX_REPAIRS = 2


class AIError(RuntimeError):
    pass


def _client():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise AIError("ANTHROPIC_API_KEY is not set. Run:  export ANTHROPIC_API_KEY=sk-ant-...  then restart the app.")
    return anthropic.Anthropic(api_key=key)


# ── 1. Idea -> spec ───────────────────────────────────────────────────────────

TRANSLATE_SYSTEM = f"""You are the strategy interpreter inside a single-stock daily backtester.
Your job: turn the user's trading idea into a strategy spec in the JSON language below, by calling
the submit_strategy tool exactly once. Never answer in prose.

{dsl.SPEC_REFERENCE}

HOW TO INTERPRET
- Be faithful to the idea. Use the user's numbers when given; otherwise pick conventional defaults
  (RSI 14 with 30/70, SMA 50/200, Bollinger 20/2, MACD 12/26/9, ATR 14) and list each default you chose
  in "assumptions".
- The engine trades ONE ticker on DAILY bars, long-only or short-only, one position at a time.
  If the idea needs something it can't do (fundamentals, earnings, news, sentiment, other tickers as
  signals, options, intraday bars, pairs), build the closest faithful approximation from price/volume and
  say exactly what you changed in "assumptions". Use status "unsupported" only when no meaningful
  price-based version exists, and explain why in unsupported_reason.
- Use status "needs_clarification" only when the idea is so vague that any spec would be a guess
  (e.g. "make money", "buy good stocks"). Ask ONE short question. Otherwise just decide and state it.
- An idea with an entry but no exit still needs a way out: use the natural opposite of the entry
  (e.g. RSI back above 50, close back below the average) and say so.
- "Breakout above the N-day high" must compare to the PRIOR N days: use offset 1 on the highest/lowest
  indicator, otherwise close can never exceed a high that includes today.
- Percentages are decimals everywhere: 5% stop -> 0.05; "2% above the SMA" -> {{"ref":"sma","mult":1.02}}.
- Only fill ticker / start_date / end_date if the user explicitly names them.

REFINING
If a CURRENT SPEC is provided, the user is asking for a change to it. Return the FULL updated spec,
changing only what they asked for, and describe the change in "interpretation".

EXAMPLE
Idea: "buy SPY when it closes above its 200 day and the 50 crosses over, 8% stop"
spec: {{"name":"Golden cross with trend filter","direction":"long",
 "indicators":[{{"id":"sma50","type":"sma","window":50}},{{"id":"sma200","type":"sma","window":200}}],
 "entry":{{"all":[{{"left":"sma50","op":"crosses_above","right":"sma200"}},{{"left":"close","op":">","right":"sma200"}}]}},
 "exit":{{"left":"sma50","op":"crosses_below","right":"sma200"}},
 "risk":{{"stop_loss_pct":0.08}},"position_size":1.0,"costs_bps":5}}
ticker: "SPY"; assumptions: ["Exit on the opposite (death) cross — no exit was specified"]
"""

SPEC_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "direction": {"type": "string", "enum": ["long", "short"]},
        "indicators": {"type": "array", "items": {"type": "object"}},
        "entry": {"type": "object", "description": "condition tree"},
        "exit": {"type": ["object", "null"], "description": "condition tree or null"},
        "risk": {"type": "object"},
        "position_size": {"type": "number"},
        "costs_bps": {"type": "number"},
    },
    "required": ["name", "direction", "indicators", "entry"],
}

TRANSLATE_TOOL = {
    "name": "submit_strategy",
    "description": "Submit the interpreted strategy (or a clarifying question / unsupported verdict).",
    "input_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok", "needs_clarification", "unsupported"]},
            "interpretation": {"type": "string", "description": "One sentence: what you understood / changed."},
            "assumptions": {"type": "array", "items": {"type": "string"}},
            "clarifying_question": {"type": "string"},
            "unsupported_reason": {"type": "string"},
            "ticker": {"type": "string"},
            "start_date": {"type": "string", "description": "YYYY-MM-DD"},
            "end_date": {"type": "string", "description": "YYYY-MM-DD"},
            "spec": SPEC_SCHEMA,
        },
        "required": ["status", "interpretation"],
    },
}


def _find_tool(resp, name):
    for block in resp.content:
        if getattr(block, "type", None) == "tool_use" and block.name == name:
            return block
    return None


def _call_tool(client, system, messages, tool, max_tokens):
    msgs = list(messages)
    for _ in range(2):
        resp = _create(client, system=system, messages=msgs, tools=[tool],
                       tool_choice={"type": "auto"}, max_tokens=max_tokens)
        call = _find_tool(resp, tool["name"])
        if call:
            return resp, call, msgs
        msgs = msgs + [{"role": "assistant", "content": resp.content},
                       {"role": "user", "content": f"Respond only by calling the {tool['name']} tool."}]
    raise AIError("The model did not return a structured answer. Try rephrasing.")


def translate(text: str, current_spec: dict | None = None, client=None) -> dict:
    """Return {'status', 'interpretation', 'assumptions', 'spec' (validated), ...}."""
    client = client or _client()
    user = text if current_spec is None else (
        f"CURRENT SPEC:\n{json.dumps(current_spec, indent=1)}\n\nREQUESTED CHANGE:\n{text}")
    messages = [{"role": "user", "content": user}]

    for attempt in range(MAX_REPAIRS + 1):
        resp, call, messages = _call_tool(client, TRANSLATE_SYSTEM, messages, TRANSLATE_TOOL, 4000)
        out = dict(call.input)
        out.setdefault("assumptions", [])
        if out.get("status") != "ok":
            return out
        try:
            out["spec"] = dsl.validate_spec(out.get("spec") or {})
            return out
        except dsl.SpecError as e:
            if attempt == MAX_REPAIRS:
                raise AIError("Couldn't build a valid strategy: " + "; ".join(e.errors))
            messages += [
                {"role": "assistant", "content": resp.content},
                {"role": "user", "content": [{
                    "type": "tool_result", "tool_use_id": call.id, "is_error": True,
                    "content": "The spec failed validation. Fix these problems and call submit_strategy again:\n- "
                               + "\n- ".join(e.errors)}]},
            ]


# ── 2. Results -> analyst report ──────────────────────────────────────────────

REVIEW_SYSTEM = """You are a skeptical quant reviewing a backtest for a student who wants to work on a
trading desk. You get the strategy's rules, its performance, and robustness diagnostics
(in/out-of-sample split, cost sensitivity, ±25% parameter perturbations, and automatic red flags).

Rules:
- Use ONLY numbers present in the data. Do not invent statistics. Quote them with sensible rounding.
- Lead with whether there is evidence of a real edge, versus luck, beta, or overfitting. A strategy that
  beats buy-and-hold only by being invested less during one crash is not the same as a timing edge.
- Explain WHY it made or lost money in market terms (trend vs mean reversion, which years drove it,
  what the exits did), not just what the metrics are.
- Be direct and brief. Plain English; define jargon once if you must use it.
- headline: ONE plain-English sentence, max 15 words, the bottom line a trader would say out loud.
- takeaways: exactly 3 bullets, each max 14 words, no jargon: what made money ("up"), what hurt ("down"),
  and the overall trade-off ("neutral"). These are what the user reads first; the longer fields are optional detail.
- This is a historical simulation on one ticker, not investment advice; do not recommend trading it live.
Call submit_report exactly once."""

REVIEW_TOOL = {
    "name": "submit_report",
    "description": "Submit the analyst report.",
    "input_schema": {
        "type": "object",
        "properties": {
            "verdict": {"type": "string",
                        "enum": ["promising", "mixed", "weak", "likely_overfit", "mostly_beta", "insufficient_data"]},
            "headline": {"type": "string", "description": "Bottom line, max 15 words."},
            "what_drove_returns": {"type": "string", "description": "2-4 sentences."},
            "where_it_struggled": {"type": "string", "description": "2-4 sentences."},
            "vs_buy_and_hold": {"type": "string", "description": "1-2 sentences, risk-adjusted."},
            "robustness": {"type": "string", "description": "2-3 sentences on split, costs, sensitivity."},
            "red_flags": {"type": "array", "items": {"type": "string"}},
            "takeaways": {
                "type": "array", "minItems": 3, "maxItems": 3,
                "items": {"type": "object",
                          "properties": {"tone": {"type": "string", "enum": ["up", "down", "neutral"]},
                                         "text": {"type": "string", "description": "max 14 words"}},
                          "required": ["tone", "text"]}},
        },
        "required": ["verdict", "headline", "takeaways", "what_drove_returns", "where_it_struggled",
                     "vs_buy_and_hold", "robustness"],
    },
}


def review(idea: str, ticker: str, spec: dict, rules: dict, metrics: dict, diag: dict,
           trades: list, client=None) -> dict:
    client = client or _client()
    closed = [t for t in trades if t["exit_date"]]
    best = sorted(closed, key=lambda t: t["return"], reverse=True)[:3]
    worst = sorted(closed, key=lambda t: t["return"])[:3]
    payload = {
        "idea": idea, "ticker": ticker, "rules_as_run": rules, "spec": spec,
        "metrics": metrics, "diagnostics": diag,
        "best_trades": best, "worst_trades": worst,
    }
    _, call, _ = _call_tool(client, REVIEW_SYSTEM,
                            [{"role": "user", "content": json.dumps(clean(payload), default=str)}],
                            REVIEW_TOOL, 4000)
    return dict(call.input)


# ── helpers ───────────────────────────────────────────────────────────────────

def _create(client, **kw):
    try:
        return client.messages.create(model=MODEL, **kw)
    except anthropic.AuthenticationError:
        raise AIError("Claude rejected the API key — check ANTHROPIC_API_KEY.")
    except anthropic.NotFoundError:
        raise AIError(f"Model '{MODEL}' not found — set ANTHROPIC_MODEL to a model your key can use.")
    except anthropic.APIConnectionError as e:
        cause = repr(e.__cause__) if e.__cause__ else "no details"
        print(f"[ai_analyst] connection error talking to Claude: {cause}", flush=True)
        raise AIError("Couldn't reach Claude's servers (network problem). "
                      f"Details: {cause[:200]}")
    except anthropic.APIError as e:
        raise AIError(f"Claude API error: {e}")
