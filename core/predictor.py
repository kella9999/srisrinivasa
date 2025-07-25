import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import pickle
import os
import sys
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
from utils.feature_engineer import add_technical_indicators

class CryptoPredictor:
    def __init__(self, model_path: str, scaler_path: str):
        """Initialize predictor with model and scaler"""
        try:
            # Load model
            self.model = XGBClassifier()
            self.model.load_model(model_path)
            
            # Load scaler
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Initialize history buffer
            self.history = pd.DataFrame(columns=['Close', 'Volume', 'High', 'Low', 'Open'])
            self.required_features = self.model.get_booster().feature_names
            
            # Validate feature set
            if not isinstance(self.required_features, list) or len(self.required_features) == 0:
                raise ValueError("Model has no feature names associated with it")
                
            logger.info(f"Model loaded successfully. Expecting {len(self.required_features)} features: {self.required_features}")
            
        except Exception as e:
            logger.error(f"Failed to initialize predictor: {str(e)}")
            raise

    def update_history(self, close: float, volume: float) -> None:
        """Maintains a rolling window of the last 100 data points with simulated OHLC data"""
        try:
            new_data = pd.DataFrame([{
                'Close': float(close),
                'Volume': float(volume),
                'High': float(close) * 1.0005,  # Simulated high (0.05% above close)
                'Low': float(close) * 0.9995,   # Simulated low (0.05% below close)
                'Open': float(close) * 0.9998   # Simulated open (0.02% below close)
            }], index=[pd.Timestamp.now()])
            
            self.history = pd.concat([self.history, new_data]).iloc[-100:]
            logger.debug(f"Updated history. Current size: {len(self.history)}")
            
        except Exception as e:
            logger.error(f"Error updating history: {str(e)}")
            raise

    def _validate_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required features are present and in correct order"""
        # Fill missing features with 0
        missing_features = set(self.required_features) - set(features.columns)
        if missing_features:
            logger.warning(f"Filling missing features {missing_features} with 0")
            for feat in missing_features:
                features[feat] = 0.0
        
        # Reorder features to match model expectations
        features = features[self.required_features]
        
        # Validate no NaN/inf values
        if features.isnull().values.any():
            logger.warning("NaN values detected in features - filling with 0")
            features = features.fillna(0)
            
        if np.isinf(features.values).any():
            logger.warning("Infinite values detected in features - replacing with 0")
            features = features.replace([np.inf, -np.inf], 0)
            
        return features

    def predict(self, close: float, volume: float) -> Dict[str, Any]:
        """Generates a prediction with detailed diagnostics"""
        try:
            self.update_history(close, volume)
            
            # Minimum data requirement
            if len(self.history) < 20:
                logger.info(f"Insufficient data ({len(self.history)}/20 points)")
                return {
                    "probability": 0.5,
                    "ready": False,
                    "message": "Collecting initial data",
                    "data_points": len(self.history)
                }
            
            # Generate features
            features = add_technical_indicators(self.history)
            
            if features.empty:
                logger.warning("Feature generation returned empty DataFrame")
                return {
                    "probability": 0.5,
                    "ready": False,
                    "message": "Feature generation failed",
                    "data_points": len(self.history)
                }
            
            # Validate and prepare features
            features = self._validate_features(features)
            logger.debug(f"Features for prediction:\n{features.iloc[-1:]}")
            
            # Scale and predict
            scaled_features = self.scaler.transform(features.iloc[-1:].values.reshape(1, -1))
            proba = float(self.model.predict_proba(scaled_features)[0][1])
            
            return {
                "probability": proba,
                "ready": True,
                "message": "Prediction successful",
                "data_points": len(self.history),
                "features": features.iloc[-1:].to_dict()
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}", exc_info=True)
            return {
                "probability": 0.5,
                "ready": False,
                "message": f"Prediction error: {str(e)}",
                "data_points": len(self.history)
            }

    def get_status(self) -> Dict[str, Any]:
        """Return current predictor status"""
        return {
            "data_points": len(self.history),
            "ready": len(self.history) >= 20,
            "required_features": self.required_features,
            "model_loaded": hasattr(self, 'model'),
            "scaler_loaded": hasattr(self, 'scaler')
        }
