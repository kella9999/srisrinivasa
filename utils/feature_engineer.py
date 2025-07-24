import pandas as pd
import ta

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Centralized feature engineering matching training features"""
    df = df.copy()
    
    # Minimum data check for indicators
    if len(df) < 20:
        raise ValueError("Insufficient data for indicator calculation (minimum 20 points needed)")
    
    # Price Features
    df['returns'] = df['Close'].pct_change()
    
    # Trend indicators
    df['ema_20'] = ta.trend.ema_indicator(df['Close'], window=20)
    df['macd'] = ta.trend.macd_diff(df['Close'])
    
    # Momentum indicators
    df['rsi'] = ta.momentum.rsi(df['Close'], window=14)
    
    # Volatility indicators
    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['bb_%b'] = (df['Close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
    df['atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    
    # Volume indicators
    df['obv'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    df['volume_ma'] = df['Volume'].rolling(20).mean()
    
    # Return only the features used in training, in correct order
    return df[['Close', 'Volume', 'ema_20', 'macd', 'rsi', 'bb_%b', 'atr', 'obv', 'volume_ma', 'returns']].dropna()
