# Equity Backtester

This repository is the Week 1 foundation for an algorithmic trading strategy backtester.
It includes a clean `DataLoader` class that downloads historical OHLCV equities data from `yfinance` and prepares it for use by strategy and backtest logic.

## What is included

- `data/loader.py` — `DataLoader` class for downloading and cleaning OHLCV equity data
- `main.py` — quick verification script that downloads sample tickers and prints closing prices
- `requirements.txt` — environment dependencies for the full project stack
- package structure placeholders for `strategies/`, `engine/`, and `dashboard/`

## Setup

1. Create or activate a Python 3.11+ virtual environment.
2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Quick test

Run:

```bash
python main.py
```

If the setup is correct, the script will download data for `AAPL`, `MSFT`, `JPM`, and `GS` and print the first few closing prices.

## Notes

- Data loading is separated from strategy logic to keep the architecture clean.
- The loader forward-fills missing observations using prior values only, avoiding lookahead bias in later signal generation.
