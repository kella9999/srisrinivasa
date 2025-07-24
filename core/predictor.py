import onnxruntime as ort
import numpy as np
import pandas as pd
import ta
from typing import Tuple

class CryptoPredictor:
    def __init__(self, model_path: str, scaler_path: str):
        # Initialize ONNX runtime session
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        
        # Load scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Required features (must match training)
        self.required_features = [
            'Close', 'Volume',
            'rsi', 'macd', 'bollinger_%',
            'returns', 'volume_ma'
        ]
    
    def _validate_input(self, df: pd.DataFrame) -> Tuple[bool, str]:
        missing = [f for f in self.required_features if f not in df.columns]
        if missing:
            return False, f"Missing features: {missing}"
        return True, ""
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identical to training phase feature engineering"""
        df = df.copy()
        
        # Price Features
        df['returns'] = df['Close'].pct_change()
        
        # Momentum
        df['rsi'] = ta.momentum.rsi(df['Close'], window=14)
        df['macd'] = ta.trend.macd_diff(df['Close'])
        
        # Volatility
        bb = ta.volatility.BollingerBands(df['Close'], window=5, window_dev=2)
        df['bollinger_%'] = (df['Close'] - bb.bollinger_lband()) / \
                           (bb.bollinger_hband() - bb.bollinger_lband())
        
        # Volume
        df['volume_ma'] = df['Volume'].rolling(window=5).mean()
        
        return df.dropna()
    
    def preprocess(self, df: pd.DataFrame) -> np.ndarray:
        """Transforms raw OHLCV data into model-ready features"""
        # Add indicators
        df = self.add_technical_indicators(df)
        
        # Validate
        is_valid, msg = self._validate_input(df)
        if not is_valid:
            raise ValueError(msg)
        
        # Select and scale features
        features = df[self.required_features].iloc[-1:].values  # Use most recent data
        return self.scaler.transform(features)
    
    def predict(self, df: pd.DataFrame) -> float:
        """Returns probability of price increase (0-1)"""
        processed = self.preprocess(df)
        inputs = {self.session.get_inputs()[0].name: processed.astype(np.float32)}
        return self.session.run(None, inputs)[0][0][1]  # Probability of class 1
