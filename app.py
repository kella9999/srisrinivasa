# app.py
from flask import Flask, request, jsonify
from core.predictor import CryptoPredictor # Corrected import path
import os

app = Flask(__name__)

# Initialize predictor
predictor = CryptoPredictor(
    model_path="models/model.json",
    scaler_path="models/btc_3m_scaler.pkl"
)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'Close' not in data:
            return jsonify({"error": "Missing 'Close' value in request"}), 400
            
        close = float(data['Close'])
        volume = float(data.get('Volume', 0)) # Volume is optional
        
        proba = predictor.predict(close, volume)
        signal = 1 if proba >= 0.6 else 0
        
        return jsonify({
            "signal": signal,
            "confidence": proba,
            "ready": len(predictor.history) >= 20
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    # The port should be configured based on environment variables for deployment
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)```

#### 4. `utils/feature_engineer.py` (Unchanged but included for completeness)
*No changes were needed here. This file is the new "single source of truth" for features.*

```python
# utils/feature_engineer.py
import pandas as pd
import ta
import numpy as np

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Generate exactly 30 features matching what the model expects"""
    if len(df) < 20:
        return pd.DataFrame()
    
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # 1. Core Price Features (10)
    df['returns'] = df['Close'].pct_change()
    df['ema_20'] = ta.trend.ema_indicator(df['Close'], window=20)
    df['macd'] = ta.trend.macd_diff(df['Close'])
    df['rsi'] = ta.momentum.rsi(df['Close'], window=14)
    
    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['bb_%b'] = (df['Close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband()) if (bb.bollinger_hband() - bb.bollinger_lband()).all() > 0 else 0
    df['atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    df['obv'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    df['volume_ma'] = df['Volume'].rolling(20).mean()
    df['price_ma'] = df['Close'].rolling(20).mean()
    
    # 2. Lag Features (10) - Use fillna to handle smaller dataframes
    for i in [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]:
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
        'volume_ma', 'price_ma', 'range', 'close_to_open', 'volatility',
        'momentum_5', 'volume_change', 'price_volume', 'ma_ratio',
        'range_ratio', 'rsi_ema', 'macd_signal', 'returns',
        # Lags must be handled carefully as they produce NaNs
        'close_lag_1', 'close_lag_2', 'close_lag_3', 'close_lag_5', 'close_lag_8',
        'close_lag_13', 'close_lag_21', 'close_lag_34', 'close_lag_55'
    ]
    # Drop any remaining NaNs after all calculations are done
    return df[feature_cols].dropna()
