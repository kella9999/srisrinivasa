import onnxruntime as ort
import numpy as np
import pandas as pd
from feature_engineer import add_technical_indicators
import pickle
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CryptoPredictor:
    def __init__(self, model_path: str, scaler_path: str):
        """Initialize predictor with model and scaler"""
        try:
            # Load ONNX model
            self.session = ort.InferenceSession(
                model_path,
                providers=['CPUExecutionProvider']
            )
            
            # Load the scaler
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Store input name for ONNX model
            self.input_name = self.session.get_inputs()[0].name
            
            # Initialize history DataFrame
            self.history = pd.DataFrame(columns=['Close', 'Volume', 'High', 'Low'])
            
            logger.info("Predictor initialized successfully")
            
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise

    def update_history(self, close: float, volume: float, high: float = None, low: float = None):
        """Maintain a rolling window of recent prices"""
        try:
            # Set default values if not provided
            high = high if high is not None else close * 1.0005
            low = low if low is not None else close * 0.9995
            
            # Create new row with timestamp
            new_row = pd.DataFrame([{
                'Close': close,
                'Volume': volume,
                'High': high,
                'Low': low
            }], index=[pd.Timestamp.now()])
            
            # Add to history (keep last 100 points)
            self.history = pd.concat([self.history, new_row]).tail(100)
            
        except Exception as e:
            logger.error(f"Failed to update history: {str(e)}")
            raise

    def predict(self, close: float, volume: float, high: float = None, low: float = None) -> float:
        """Predict the probability of price increase"""
        try:
            # Update history with new data point
            self.update_history(close, volume, high, low)
            
            # Check if we have enough data
            if len(self.history) < 20:
                logger.warning(f"Insufficient data points ({len(self.history)}), returning neutral probability")
                return 0.5  # Neutral probability
            
            # Calculate features
            df_with_features = add_technical_indicators(self.history.copy())
            if df_with_features.empty:
                logger.warning("Feature calculation returned empty DataFrame")
                return 0.5
                
            # Get most recent features
            features = df_with_features.iloc[-1].values
            
            # Scale features
            scaled = self.scaler.transform([features])
            
            # Run ONNX inference
            inputs = {self.input_name: scaled.astype(np.float32)}
            output = self.session.run(None, inputs)
            
            # Return probability of class 1 (price increase)
            return float(output[0][0][1])
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return 0.5  # Fallback to neutral probability
