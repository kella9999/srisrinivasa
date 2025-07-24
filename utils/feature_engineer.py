# feature_engineer.py
import pandas as pd
import ta
import numpy as np

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Generate exactly 30 features matching model expectations"""
    if len(df) < 20:
        return pd.DataFrame()
    
    # Ensure numeric types
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # 1. Original 10 Technical Indicators
    df['returns'] = df['Close'].pct_change()
    df['ema_20'] = ta.trend.ema_indicator(df['Close'], window=20)
    df['macd'] = ta.trend.macd_diff(df['Close'])
    df['rsi'] = ta.momentum.rsi(df['Close'], window=14)
    
    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['bb_%b'] = (df['Close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
    df['atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    df['obv'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    df['volume_ma'] = df['Volume'].rolling(20).mean()
    df['price_ma'] = df['Close'].rolling(20).mean()
    
    # 2. Additional 20 Features (should match your training features)
    # Price-based features
    for i in [1, 2, 3, 5, 8, 13]:
        df[f'close_lag_{i}'] = df['Close'].shift(i)
        df[f'volume_lag_{i}'] = df['Volume'].shift(i)
    
    # Volatility features
    df['range'] = df['High'] - df['Low']
    df['close_to_range'] = df['Close'] / df['range']
    
    # Momentum features
    for window in [3, 5, 10]:
        df[f'momentum_{window}'] = df['Close'] - df['Close'].shift(window)
    
    # Volume features
    df['volume_change'] = df['Volume'].pct_change()
    df['volume_std'] = df['Volume'].rolling(10).std()
    
    # Combined features
    df['price_volume'] = df['Close'] * df['Volume']
    df['ma_ratio'] = df['ema_20'] / df['price_ma']
    
    # Ensure we return exactly 30 features in the correct order
    features = [
        'Close', 'Volume', 'ema_20', 'macd', 'rsi', 'bb_%b', 'atr', 'obv', 
        'volume_ma', 'price_ma', 'close_lag_1', 'volume_lag_1', 'close_lag_2',
        'volume_lag_2', 'close_lag_3', 'volume_lag_3', 'close_lag_5', 'volume_lag_5',
        'close_lag_8', 'volume_lag_8', 'close_lag_13', 'volume_lag_13', 'range',
        'close_to_range', 'momentum_3', 'momentum_5', 'momentum_10', 'volume_change',
        'volume_std', 'price_volume', 'ma_ratio'
    ]
    
    return df[features[:30]].dropna()  # Ensure exactly 30 features
