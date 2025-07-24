from flask import Flask, request, jsonify
import pandas as pd
import pickle
from predictor import CryptoPredictor
import logging

# Configuration
MODEL_PATH = "models/btc_3m.onnx"
SCALER_PATH = "models/btc_3m_scaler.pkl"

# Initialize Flask
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Load model globally
try:
    predictor = CryptoPredictor(MODEL_PATH, SCALER_PATH)
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Model loading failed: {str(e)}")
    raise

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "model": "BTC_3m"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input
        data = request.get_json()
        
        # Basic validation
        if not data or 'Close' not in data or 'Volume' not in data:
            return jsonify({
                "error": "Invalid input",
                "required": ["Close", "Volume"],
                "received": list(data.keys()) if data else None
            }), 400
        
        # Convert to DataFrame
        input_df = pd.DataFrame([data])
        
        # Get prediction
        proba = predictor.predict(input_df)
        signal = 1 if proba >= 0.6 else 0  # Confidence threshold
        
        return jsonify({
            "signal": signal,
            "probability": float(proba),
            "confidence": "high" if abs(proba - 0.5) > 0.2 else "low",
            "features_used": predictor.required_features
        })
        
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({
            "error": str(e),
            "example_request": {
                "Close": 42000.0,
                "Volume": 1500.0
            }
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
