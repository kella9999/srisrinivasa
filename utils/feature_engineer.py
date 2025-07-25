import pandas as pd
import ta
import numpy as np

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates 30+ technical features with robust error handling
    Returns empty DataFrame if insufficient data
    """
    if len(df) < 20:
        return pd.DataFrame()

    # Convert all columns to numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Handle NaN/infinite values
    if df.isnull().values.any() or np.isinf(df.values).any():
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # 1. Core Price & Trend Features
    df['returns'] = df['Close'].pct_change()
    df['ema_20'] = ta.trend.ema_indicator(df['Close'], window=20)
    df['macd'] = ta.trend.macd_diff(df['Close'])
    df['macd_signal'] = ta.trend.macd_signal(df['Close'])
    df['price_ma'] = df['Close'].rolling(20).mean()

    # 2. Momentum Features
    df['rsi'] = ta.momentum.rsi(df['Close'], window=14)
    df['momentum_5'] = df['Close'] - df['Close'].shift(5)
    df['rsi_ema'] = ta.trend.ema_indicator(df['rsi'], window=14)

    # 3. Volatility Features
    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['bb_%b'] = (df['Close'] - bb.bollinger_lband()) / (
        bb.bollinger_hband() - bb.bollinger_lband())
    df['bb_%b'].replace([np.inf, -np.inf], 0, inplace=True)

    df['atr'] = ta.volatility.average_true_range(
        df['High'], df['Low'], df['Close'], window=14)
    df['range'] = df['High'] - df['Low']
    df['volatility'] = df['Close'].rolling(5).std()
    df['range_ratio'] = df['range'] / df['atr']
    df['range_ratio'].replace([np.inf, -np.inf], 0, inplace=True)

    # 4. Volume Features
    df['obv'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    df['volume_ma'] = df['Volume'].rolling(20).mean()
    df['volume_change'] = df['Volume'].pct_change()
    df['price_volume'] = df['Close'] * df['Volume']

    # 5. Derived & Ratio Features
    df['close_to_open'] = df['Close'] / df['Close'].shift(1)
    df['ma_ratio'] = df['ema_20'] / df['price_ma']
    df['ma_ratio'].replace([np.inf, -np.inf], 0, inplace=True)

    # 6. Lag Features
    lag_periods = [1, 2, 3, 5, 8, 13, 21, 34, 55]
    for i in lag_periods:
        df[f'close_lag_{i}'] = df['Close'].shift(i)

    # Final feature selection and validation
    feature_cols = [
        'Close', 'Volume', 'returns', 'ema_20', 'macd', 'macd_signal', 'price_ma',
        'rsi', 'momentum_5', 'rsi_ema', 'bb_%b', 'atr', 'range', 'volatility',
        'range_ratio', 'obv', 'volume_ma', 'volume_change', 'price_volume',
        'close_to_open', 'ma_ratio',
        'close_lag_1', 'close_lag_2', 'close_lag_3', 'close_lag_5', 'close_lag_8',
        'close_lag_13', 'close_lag_21', 'close_lag_34', 'close_lag_55'
    ]

    # Ensure all features exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    return df[feature_cols].dropna()
