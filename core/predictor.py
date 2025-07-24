import os
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from feature_engineer import add_technical_indicators

class CryptoPredictor:
    def __init__(self, model_path: str, scaler_path: str):
        """
        Initialize the predictor with model and scaler.
        
        Args:
            model_path: Path to XGBoost model (JSON format)
            scaler_path: Path to scaler (pickle format)
        """
        # Load model
        self.model = XGBClassifier()
        self.model.load_model(model_path)
        print(f"✅ Loaded XGBoost model from {model_path}")
        
        # Load scaler (using the fixed 30-feature scaler)
        try:
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            print(f"✅ Loaded scaler from {scaler_path}")
            print(f"Scaler features: {getattr(self.scaler, 'n_features_in_', 'Unknown')}")
        except Exception as e:
            print(f"❌ Failed to load scaler: {e}")
            raise
        
        # Initialize price history
        self.history = pd.DataFrame(columns=['Close', 'Volume', 'High', 'Low'])
        
        # Debug info
        print(f"Model expects {self.model.n_features_in_} features")
        print("Predictor initialized successfully")

    def update_history(self, close: float, volume: float, high: float = None, low: float = None):
        """
        Maintain a rolling window of price data for indicators.
        
        Args:
            close: Closing price
            volume: Trading volume
            high: High price (optional)
            low: Low price (optional)
        """
        high = high if high is not None else close * 1.0005  # Default if not provided
        low = low if low is not None else close * 0.9995    # Default if not provided
        
        new_row = pd.DataFrame([[close, volume, high, low]], 
                             columns=['Close', 'Volume', 'High', 'Low'],
                             index=[pd.Timestamp.now()])
        
        # Clean concatenation to avoid warnings
        if not self.history.empty:
            self.history = pd.concat([self.history, new_row], axis=0)
        else:
            self.history = new_row
            
        # Keep last 100 data points
        self.history = self.history.tail(100)

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
            # Update price history
            self.update_history(close, volume)
            
            # Check if we have enough data
            if len(self.history) < 20:
                print("⚠️ Not enough data points (need 20, have {len(self.history)})")
                return 0.5  # Neutral probability
                
            # Calculate technical indicators
            df_features = add_technical_indicators(self.history.copy())
            if df_features.empty:
                print("⚠️ No features generated - check indicator calculations")
                return 0.5
                
            # Get most recent features
            features = df_features.iloc[-1].values
            
            # Debug: Verify feature count matches
            if len(features) != self.model.n_features_in_:
                print(f"⚠️ Feature mismatch: Model expects {self.model.n_features_in_}, got {len(features)}")
                return 0.5
                
            # Scale features
            scaled_features = self.scaler.transform([features])
            
            # Make prediction
            proba = self.model.predict_proba(scaled_features)[0][1]
            print(f"📊 Prediction: {proba:.4f} (Price: {close}, Volume: {volume})")
            return float(proba)
            
        except Exception as e:
            print(f"❌ Prediction error: {str(e)}")
            return 0.5  # Fallback neutral probability
