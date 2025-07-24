from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from predictor import CryptoPredictor
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["10 per minute"]
)

# Initialize predictor
try:
    predictor = CryptoPredictor(
        model_path="models/btc_3m_quant.onnx",
        scaler_path="models/btc_3m_scaler.pkl"
    )
    logger.info("Predictor initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize predictor: {str(e)}")
    raise

@app.route('/health')
def health():
    """Health check endpoint"""
    status = {
        "status": "healthy",
        "model_loaded": os.path.exists("models/btc_3m_quant.onnx"),
        "data_points": len(predictor.history),
        "ready_for_predictions": len(predictor.history) >= 20
    }
    return jsonify(status)

@app.route('/predict', methods=['POST'])
@limiter.limit("5 per second")
def predict():
    """Prediction endpoint"""
    try:
        data = request.get_json()
        
        # Validate input
        if not data or 'Close' not in data:
            return jsonify({"error": "Missing required 'Close' price"}), 400
        
        close = float(data['Close'])
        volume = float(data.get('Volume', 0))
        high = float(data.get('High', close * 1.0005))
        low = float(data.get('Low', close * 0.9995))
        
        # Get prediction
        proba = predictor.predict(close, volume, high, low)
        signal = 1 if proba >= 0.6 else 0
        
        return jsonify({
            "signal": signal,
            "confidence": float(proba),
            "ready": len(predictor.history) >= 20,
            "features_used": [
                "Close", "Volume", "ema_20", "macd", "rsi", 
                "bb_%b", "atr", "obv", "volume_ma", "returns"
            ]
        })
    
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
