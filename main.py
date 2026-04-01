from data.loader import DataLoader


def main() -> None:
    """Download sample equity data and print the first few closing prices."""
    tickers = ["AAPL", "MSFT", "JPM", "GS"]
    loader = DataLoader(
        tickers=tickers,
        start="2018-01-01",
        end="2024-12-31",
    )

    data = loader.download_data()
    print(data["Close"].head())


if __name__ == "__main__":
    main()