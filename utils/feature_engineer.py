import pandas as pd
import ta

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Centralized feature engineering"""
    df = df.copy()
    
    # Price Features
    df['returns'] = df['Close'].pct_change()
    
    # Momentum
    df['rsi'] = ta.momentum.rsi(df['Close'], window=14)
    df['macd'] = ta.trend.macd_diff(df['Close'])
    
    # Volatility
    bb = ta.volatility.BollingerBands(df['Close'])
    df['bollinger_%'] = (df['Close'] - bb.bollinger_lband()) / \
                       (bb.bollinger_hband() - bb.bollinger_lband())
    
    # Volume
    df['volume_ma'] = df['Volume'].rolling(window=5).mean()
    
    return df.dropna()
