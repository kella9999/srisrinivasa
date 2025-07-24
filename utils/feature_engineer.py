import pandas as pd
import ta
import numpy as np

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Generate exactly 30 features matching what the model expects"""
    if len(df) < 20:
        return pd.DataFrame()
    
    # Convert to numeric (handle any non-numeric data)
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # 1. Core Price Features (10)
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
    
    # 2. Lag Features (10)
    for i in [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]:
        if i < len(df):
            df[f'close_lag_{i}'] = df['Close'].shift(i)
    
    # 3. Derived Features (10)
    df['range'] = df['High'] - df['Low']
    df['close_to_open'] = df['Close'] / df['Close'].shift(1)
    df['volatility'] = df['Close'].rolling(5).std()
    df['momentum_5'] = df['Close'] - df['Close'].shift(5)
    df['volume_change'] = df['Volume'].pct_change()
    df['price_volume'] = df['Close'] * df['Volume']
    df['ma_ratio'] = df['ema_20'] / df['price_ma']
    df['range_ratio'] = df['range'] / df['atr']
    df['rsi_ema'] = ta.trend.ema_indicator(df['rsi'], window=14)
    df['macd_signal'] = ta.trend.macd_signal(df['Close'])
    
    # Return exactly 30 features in fixed order
    feature_cols = [
        'Close', 'Volume', 'ema_20', 'macd', 'rsi', 'bb_%b', 'atr', 'obv',
        'volume_ma', 'price_ma', 'close_lag_1', 'close_lag_2', 'close_lag_3',
        'close_lag_5', 'close_lag_8', 'close_lag_13', 'close_lag_21',
        'close_lag_34', 'close_lag_55', 'close_lag_89', 'range', 'close_to_open',
        'volatility', 'momentum_5', 'volume_change', 'price_volume', 'ma_ratio',
        'range_ratio', 'rsi_ema', 'macd_signal'
    ]
    
    return df[feature_cols].dropna()
