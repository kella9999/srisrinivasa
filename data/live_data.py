import yfinance as yf
import pandas as pd

def get_live_candles(coin="BTC-USD", interval="3m", lookback="60d"):
    """Real-time data with technical indicators"""
    df = yf.download(coin, interval=interval, period=lookback)
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["RSI_14"] = 70 - (30 * df["Close"].pct_change().ewm(span=14).mean())
    return df.dropna()
