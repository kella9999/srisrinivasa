import onnxruntime as ort
import numpy as np
import pandas as pd
from utils.feature_engineer import add_technical_indicators
import pickle
from functools import lru_cache

class CryptoPredictor:
    def __init__(self, model_path: str, scaler_path: str):
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
    
    @lru_cache(maxsize=100)
    def _calculate_indicators(self, close: float, volume: float) -> tuple:
        """Cached indicator calculation"""
        df = pd.DataFrame([[close, volume]], columns=['Close', 'Volume'])
        df = add_technical_indicators(df)
        return tuple(df.iloc[0].values)
    
    def predict(self, close: float, volume: float) -> float:
        """Optimized prediction with caching"""
        features = self._calculate_indicators(close, volume)
        scaled = self.scaler.transform([features])
        inputs = {self.session.get_inputs()[0].name: scaled.astype(np.float32)}
        return self.session.run(None, inputs)[0][0][1]  # Probability of price increase
