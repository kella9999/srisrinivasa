import onnxruntime as ort
import numpy as np
import pandas as pd
from feature_engineer import add_technical_indicators
import pickle
from functools import lru_cache

class CryptoPredictor:
    def __init__(self, model_path: str, scaler_path: str):
        # Load ONNX model
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )

        # Load the scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        # ✅ Debug print: confirm ONNX input name
        input_name = self.session.get_inputs()[0].name
        print(f"[DEBUG] ONNX model input name: {input_name}")
        self.input_name = input_name  # store for reuse

    @lru_cache(maxsize=100)
    def _calculate_indicators(self, close: float, volume: float) -> tuple:
        """Calculate features from close and volume"""
        df = pd.DataFrame([[close, volume]], columns=['Close', 'Volume'])
        df = add_technical_indicators(df)
        return tuple(df.iloc[0].values)

    def predict(self, close: float, volume: float) -> float:
        """Predict the probability of price increase"""
        try:
            # Compute features
            features = self._calculate_indicators(close, volume)

            # Scale features
            scaled = self.scaler.transform([features])

            # Run ONNX inference
            inputs = {self.input_name: scaled.astype(np.float32)}
            output = self.session.run(None, inputs)

            # Return probability of class 1 (price increase)
            return float(output[0][0][1])
        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
            return 0.0  # fallback
