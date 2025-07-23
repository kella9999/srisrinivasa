import yfinance as yf
import pandas as pd
import os

def fetch_candle_data(coin="BTC-USD", interval="3m", days=60):
    """Freshest data direct from yFinance"""
    data = yf.download(
        tickers=coin,
        interval=interval,
        period=f"{days}d"
    )
    # Add basic technical indicators
    data['SMA_10'] = data['Close'].rolling(10).mean()
    data['RSI_14'] = 70 - (30 * (data['Close'].pct_change().rolling(14).apply(
        lambda x: (x.where(x < 0, 0).mean() / x.where(x > 0, 0).mean()
    ))
    return data.dropna()

# Example usage:
if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    fetch_candle_data().to_csv("data/BTCUSDT_3m.csv")
    print("Data saved to data/BTCUSDT_3m.csv")
