import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import pickle
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.feature_engineer import add_technical_indicators

class CryptoPredictor:
    def __init__(self, model_path: str, scaler_path: str):
        self.model = XGBClassifier()
        self.model.load_model(model_path)
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
            
        self.history = pd.DataFrame(columns=['Close', 'Volume', 'High', 'Low', 'Open'])
        self.required_features = self.model.get_booster().feature_names
        print(f"Model loaded successfully. Expecting {len(self.required_features)} features.")

    def update_history(self, close: float, volume: float):
        """Maintains a rolling window of the last 100 data points"""
        new_data = pd.DataFrame([{
            'Close': close,
            'Volume': volume,
            'High': close * 1.0005,
            'Low': close * 0.9995,
            'Open': close * 0.9998
        }], index=[pd.Timestamp.now()])
        
        self.history = pd.concat([self.history, new_data]).iloc[-100:]

    def predict(self, close: float, volume: float) -> float:
        """Generates a prediction probability with robust error handling"""
        self.update_history(close, volume)
        
        # Minimum data requirement
        if len(self.history) < 20:
            print(f"Collecting initial data... {len(self.history)}/20 points")
            return 0.5
            
        try:
            # Generate features
            features = add_technical_indicators(self.history)
            
            if features.empty:
                print("Feature generation failed - not enough data points")
                return 0.5
                
            # Ensure all required features are present
            missing_features = set(self.required_features) - set(features.columns)
            if missing_features:
                print(f"Warning: Filling missing features {missing_features} with 0")
                for feat in missing_features:
                    features[feat] = 0.0
                    
            # Maintain consistent feature order
            features = features[self.required_features]
            
            # Scale and predict
            scaled_features = self.scaler.transform(features.iloc[-1:].values.reshape(1, -1))
            proba = float(self.model.predict_proba(scaled_features)[0][1])
            
            return proba
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return 0.5
