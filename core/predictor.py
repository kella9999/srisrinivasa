# core/predictor.py
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import pickle
import os
import sys

# Make sure the utils module can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.feature_engineer import add_technical_indicators

class CryptoPredictor:
    def __init__(self, model_path: str, scaler_path: str):
        self.model = XGBClassifier()
        self.model.load_model(model_path)
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
            
        self.history = pd.DataFrame(columns=['Close', 'Volume', 'High', 'Low'])
        print(f"Model loaded and expects {self.model.n_features_in_} features.")

    def update_history(self, close: float, volume: float):
        """Adds new data to the history, keeping it at a max of 100 entries."""
        new_data = pd.DataFrame([[close, volume, close * 1.0005, close * 0.9995]],
                              columns=['Close', 'Volume', 'High', 'Low'],
                              index=[pd.Timestamp.now()])
        
        self.history = pd.concat([self.history, new_data]).iloc[-100:]

    def predict(self, close: float, volume: float) -> float:
        """Makes a prediction based on the latest data."""
        try:
            self.update_history(close, volume)
            
            if len(self.history) < 20:
                print(f"Collecting more data... {len(self.history)}/20 entries.")
                return 0.5
                
            features = add_technical_indicators(self.history)
            
            if features.empty:
                print("Feature generation failed, not enough data for all indicators.")
                return 0.5
            
            # Ensure feature columns match the model's training order
            model_features = self.model.get_booster().feature_names
            features = features[model_features]
            
            # Scale the latest set of features
            scaled_features = self.scaler.transform(features.iloc[-1:])
            
            # Predict the probability
            proba = self.model.predict_proba(scaled_features)[0][1]
            return float(proba)
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0.5
