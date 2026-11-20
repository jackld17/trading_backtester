# Stock Backtester

Type a trading idea in plain English, like *"buy a close above the prior 20-day high, exit below the prior 10-day low, 10% trailing stop"*, and get back:

- a backtest on daily data for any ticker,
- a set of robustness checks that try to break the result, and
- a written review from Claude that says whether there is evidence of a real edge or just luck, beta, or overfitting.

A Flask web app with a manual mode for classic moving-average and RSI strategies.

## How it works

```
idea (English) ──► Claude ──► JSON strategy spec ──► validator ──► simulator ──► metrics + diagnostics ──► Claude review
                    ▲                                    │
                    └──── validation errors fed back ────┘
```

1. **Interpret.** Claude turns the idea into a spec in a small JSON strategy language (`strategy_dsl.py`). It never writes code, and nothing it returns is executed. If the spec fails validation, the errors are sent back and Claude repairs it (up to two retries). Any assumption it had to make (default windows, a missing exit rule) is listed for the user.
2. **Show what will run.** The spec is rendered back into plain English directly from the JSON, so the rules on screen are exactly the rules that were tested, not the model's paraphrase.
3. **Simulate.** A bar-by-bar engine runs the spec on Yahoo Finance daily prices.
4. **Stress-test.** `analysis.py` computes performance metrics and robustness diagnostics.
5. **Review.** Claude reads only those computed numbers and writes the verdict. Every figure in the report comes from the engine.

You can then refine in follow-up messages ("tighten the stop to 5%", "only trade above the 200-day"). Claude edits the existing spec rather than starting over.

## The engine

**Timing (no lookahead).** Conditions are evaluated at day *t*'s close using data through day *t*. Signal entries and exits fill at day *t+1*'s open. Stops and targets trigger intraday on the high and low. They fill at the stop level, or at the open if price gaps through it, and if a stop and a target are both touched on the same day, the stop is assumed to hit first.

**Strategy language.**

| | |
|---|---|
| Indicators | SMA, EMA, RSI (Wilder), MACD (line, signal, histogram), Bollinger Bands (upper, middle, lower, %B, width), ATR, rolling high and low, rate of change, z-score, realized volatility, overnight gap |
| Composition | indicators of indicators (e.g. a z-score of RSI), and `offset` to look N bars back |
| Conditions | `>` `<` `>=` `<=` `crosses_above` `crosses_below`, with operands scaled as `ref × mult + add`, combined with `all` / `any` / `not` |
| Risk rules | stop loss, take profit, trailing stop, max holding period |
| Execution | long or short, position size as a fraction of equity, transaction cost in bps per side (default 5) |

**Metrics.** Total and annualized return, volatility, Sharpe, Sortino, Calmar, max drawdown and its length, market exposure, beta and correlation to the stock, year-by-year returns against buy and hold, and trade statistics (win rate, average win and loss, profit factor, exit reasons). Trade returns are net of costs on both sides.

**Robustness checks.**

- **Buy and hold:** does the strategy beat simply holding the stock on a risk-adjusted basis?
- **Sub-period stability:** Sharpe before and after a 70/30 split of the date range.
- **Trade count:** are there enough trades to tell skill from luck?
- **Costs:** results re-run at 0, 10 and 25 bps per side.
- **Parameter sensitivity:** every threshold, risk setting and lookback the strategy uses is moved ±25% and the strategy re-run.

The results become a pass / warn / fail scorecard plus plain-English red flags, such as "results are mostly buy-and-hold beta" or "costs erase the edge".

### Example

SPY, 2015–2024, 5 bps per side:

| Strategy | Return | Sharpe | Max drawdown | Trades |
|---|---|---|---|---|
| Buy and hold | +240% | 0.78 | | |
| 20-day breakout, 10% trailing stop | +115% | 0.88 | −9% | 45 |
| RSI(14) below 30 dip-buy | +4% | 0.09 | −30% | 15 |

The breakout gives up total return for a much smaller drawdown. The dip-buy barely trades and trails cash.

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...        # needed for AI mode only; manual mode works without it
export ANTHROPIC_MODEL=claude-opus-5-5     # optional; this is the default
python3 app.py                             # then open http://127.0.0.1:5000
```

Set `PORT` to use a different port. On macOS, port 5000 is often taken by AirPlay.

The app reads the key from the environment. If you keep it in a `.env` file, load it into your shell first. `.env` is git-ignored, so never commit the key.

## Using it

- **AI Analyst:** describe a strategy, pick a ticker and dates, and run. Refine with follow-ups, or edit the JSON spec directly and re-run without calling Claude.
- **Manual:** the original MA crossover and RSI form. It now runs on the same engine, costs and metrics as AI mode.
- **Saved reports:** save any run to `reports/` (local JSON files, git-ignored). Each report has a printable page; use Cmd+P → Save as PDF.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

There are 19 tests on synthetic seeded prices, so they need no network or API key. They cover:

- no lookahead: running on truncated data reproduces the full run bar for bar,
- accounting: replaying every fill with costs reproduces final equity to the cent,
- net-of-cost trade returns that compound to the equity curve,
- parameter perturbation,
- the scorecard,
- spec validation, and
- the Flask routes.

## Project layout

```
app.py            Flask routes: manual form, /analyze, saved reports
ai_analyst.py     Claude calls: idea → spec (tool use + auto-repair), results → review
strategy_dsl.py   strategy language: validation, indicators, condition evaluation, simulator
analysis.py       metrics, robustness diagnostics, scorecard, chart
engine.py         price data (yfinance)
templates/        web UI and printable report page
tests/            unit tests
backtest.py       the original standalone 50-day MA script, kept for reference
```

## Limitations

- One ticker on daily bars, holding one position at a time. Ideas that need fundamentals, news, other tickers or intraday data are approximated from price and volume, and the approximation is shown to the user.
- Idle cash earns nothing, and Sharpe assumes a zero risk-free rate. Both understate low-exposure strategies when rates are high.
- Shorts pay no borrow cost.
- Prices are split- and dividend-adjusted Yahoo Finance data. Testing tickers that exist today adds survivorship bias.
- The sub-period split is a stability check, not true out-of-sample testing. Refining a strategy repeatedly against the same history is a form of overfitting.

This is a research and learning tool, not investment advice.
