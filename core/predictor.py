import os
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from feature_engineer import add_technical_indicators

class CryptoPredictor:
    def __init__(self, model_path: str, scaler_path: str):
        """Initialize predictor with model and scaler"""
        # Load model
        self.model = XGBClassifier()
        self.model.load_model(model_path)
        
        # Load scaler
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
            
        # Initialize history
        self.history = pd.DataFrame(columns=['Close', 'Volume', 'High', 'Low'])
        
        print(f"Predictor initialized with {self.model.n_features_in_} features")

    def update_history(self, close: float, volume: float, high: float = None, low: float = None):
        """Maintain rolling window of prices"""
        high = high if high is not None else close * 1.0005
        low = low if low is not None else close * 0.9995
        
        new_row = pd.DataFrame([[close, volume, high, low]], 
                             columns=['Close', 'Volume', 'High', 'Low'])
        
        # Fixed concatenation
        self.history = pd.concat([self.history, new_row], ignore_index=True).tail(100)

    def predict(self, close: float, volume: float) -> float:
        """Make prediction with proper feature engineering"""
        try:
            # Update history
            self.update_history(close, volume)
            
            # Check if we have enough data
            if len(self.history) < 20:
                return 0.5
                
            # Calculate features
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
