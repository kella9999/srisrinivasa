import os
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from feature_engineer import add_technical_indicators

class CryptoPredictor:
    def __init__(self, model_path: str, scaler_path: str):
        """
        Initialize predictor with model and scaler.
        
        Args:
            model_path: Path to XGBoost model (JSON format)
            scaler_path: Path to scaler (pickle format)
        """
        # Load model
        self.model = XGBClassifier()
        self.model.load_model(model_path)
        
        # Load scaler (using the fixed 30-feature scaler)
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
            
        # Initialize price history
        self.history = pd.DataFrame(columns=['Close', 'Volume', 'High', 'Low'])
        
        # Verify feature count match
        self.expected_features = self.model.n_features_in_
        print(f"✅ Predictor initialized (expects {self.expected_features} features)")

    def update_history(self, close: float, volume: float, high: float = None, low: float = None):
        """
        Maintain a rolling window of price data for indicators.
        Fixed to handle pandas FutureWarnings.
        """
        high = high if high is not None else close * 1.0005
        low = low if low is not None else close * 0.9995
        
        new_row = pd.DataFrame({
            'Close': [close],
            'Volume': [volume],
            'High': [high],
            'Low': [low]
        }, index=[pd.Timestamp.now()])
        
        # Clean concatenation that avoids warnings
        self.history = pd.concat([self.history, new_row]).iloc[-100:]

    def predict(self, close: float, volume: float) -> float:
        """
        Make a prediction using the current market data.
        
        Args:
            close: Current closing price
            volume: Current trading volume
            
        Returns:
            float: Probability of price increase (0-1)
        """
        try:
            # 1. Update price history
            self.update_history(close, volume)
            
            # 2. Check if we have enough data
            if len(self.history) < 20:
                print(f"⚠️ Need 20 data points (has {len(self.history)})")
                return 0.5
                
            # 3. Calculate technical indicators
            df_features = add_technical_indicators(self.history.copy())
            if df_features.empty:
                print("⚠️ No features generated")
                return 0.5
                
            # 4. Verify feature count matches model
            features = df_features.iloc[-1].values
            if len(features) != self.expected_features:
                print(f"⚠️ Feature mismatch (model wants {self.expected_features}, got {len(features)})")
                return 0.5
                
            # 5. Scale features and predict
            scaled_features = self.scaler.transform([features])
            proba = self.model.predict_proba(scaled_features)[0][1]
            
            # 6. Return prediction (clamped to 0-1 range)
            return float(np.clip(proba, 0, 1))
            
        except Exception as e:
            print(f"❌ Prediction failed: {str(e)}")
            return 0.5  # Fallback neutral probability
