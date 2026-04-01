from data.loader import DataLoader
from engine.backtester import run_backtest
from strategies.momentum import MomentumStrategy


def main() -> None:
    """Download sample equity data, run the momentum strategy, and print results."""
    tickers = ["AAPL", "MSFT", "JPM", "GS"]
    loader = DataLoader(
        tickers=tickers,
        start="2018-01-01",
        end="2024-12-31",
    )

    data = loader.download_data()
    close_prices = data["Close"]

    strategy = MomentumStrategy(lookback_window=63, transaction_cost=0.0005)
    returns = run_backtest(strategy, close_prices)

    print("First few momentum strategy returns:")
    print(returns.head())


if __name__ == "__main__":
    main()