import onnxruntime as ort
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import ta

class CryptoPredictor:
    def __init__(self, model_path: str, scaler: StandardScaler):
        self.model = ort.InferenceSession(model_path)
        self.scaler = scaler

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['rsi'] = ta.momentum.rsi(df['Close'], window=14)
        df['macd'] = ta.trend.macd(df['Close'])
        bb = ta.volatility.BollingerBands(close=df['Close'], window=5, window_dev=2)
        df['bollinger_%'] = (df['Close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
        df = df.dropna()
        return df

    def preprocess(self, df: pd.DataFrame) -> np.ndarray:
        df = self.add_technical_indicators(df)
        features = df[['Close', 'Volume', 'rsi', 'macd', 'bollinger_%']]
        return self.scaler.transform(features)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        processed = self.preprocess(df)
        inputs = {self.model.get_inputs()[0].name: processed.astype(np.float32)}
        pred = self.model.run(None, inputs)
        return pred[0]
