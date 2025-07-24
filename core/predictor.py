import pandas as pd
import numpy as np
from xgboost import XGBClassifier

class CryptoPredictor:
    def __init__(self, model_path: str, scaler_path: str):
        self.model = XGBClassifier()
        self.model.load_model(model_path)
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
            
        self.history = pd.DataFrame(columns=['Close', 'Volume', 'High', 'Low'])
        print(f"Model expects {self.model.n_features_in_} features")

    def update_history(self, close: float, volume: float):
        """Clean history update without warnings"""
        new_data = pd.DataFrame([[close, volume, close*1.0005, close*0.9995]],
                              columns=['Close', 'Volume', 'High', 'Low'],
                              index=[pd.Timestamp.now()])
        
        if self.history.empty:
            self.history = new_data
        else:
            self.history = pd.concat([self.history, new_data]).iloc[-100:]

    def predict(self, close: float, volume: float) -> float:
        try:
            self.update_history(close, volume)
            
            if len(self.history) < 20:
                print(f"Need {20-len(self.history)} more data points")
                return 0.5
                
            features = add_technical_indicators(self.history)
            if features.empty:
                print("Feature generation failed")
                return 0.5
                
            print("Generated features:", features.shape[1])  # Debug feature count
            scaled = self.scaler.transform([features.iloc[-1].values])
            proba = self.model.predict_proba(scaled)[0][1]
            return float(proba)
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0.5
