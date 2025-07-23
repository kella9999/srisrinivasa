import joblib
import pandas as pd

class CryptoPredictor:
    def __init__(self, model_path="models/BTCUSDT_3m.joblib"):
        self.model = joblib.load(model_path)
    
    def predict(self, last_price: float) -> float:
        """Predict next candle close"""
        return float(self.model.predict([[last_price]])[0])
