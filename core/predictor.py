# predictor.py
import os
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from feature_engineer import add_technical_indicators

class CryptoPredictor:
    def __init__(self, model_path: str, scaler_path: str):
        """Initialize predictor with model and scaler"""
        self.model = XGBClassifier()
        self.model.load_model(os.path.join(os.path.dirname(model_path), "model.json"))
        
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
            
        # Initialize history
        self.history = pd.DataFrame(columns=['Close', 'Volume', 'High', 'Low'])
        
    def update_history(self, close: float, volume: float, high: float = None, low: float = None):
        """Maintain rolling window of prices"""
        high = high if high is not None else close * 1.0005
        low = low if low is not None else close * 0.9995
        
        new_row = pd.DataFrame([[close, volume, high, low]], 
                             columns=['Close', 'Volume', 'High', 'Low'],
                             index=[pd.Timestamp.now()])
        
        self.history = pd.concat([self.history, new_row]).tail(100)  # Keep last 100 points

    def predict(self, close: float, volume: float) -> float:
        """Make prediction with proper feature engineering"""
        try:
            # Update history
            self.update_history(close, volume)
            
            # Skip if not enough data
            if len(self.history) < 20:
                return 0.5
                
            # Calculate features (matches training)
            df = add_technical_indicators(self.history.copy())
            if df.empty:
                return 0.5
                
            # Get most recent features
            features = df.iloc[-1].values
            
            # Scale and predict
            scaled = self.scaler.transform([features])
            proba = self.model.predict_proba(scaled)[0][1]
            return float(proba)
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0.5
