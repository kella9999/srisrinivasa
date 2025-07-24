from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from predictor import CryptoPredictor
import os

app = Flask(__name__)
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["10 per minute"]
)

# Initialize predictor
predictor = CryptoPredictor(
    model_path="models/btc_3m_quant.onnx",
    scaler_path="models/btc_3m_scaler.pkl"
)

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": os.path.exists("models/btc_3m_quant.onnx")
    })

@app.route('/predict', methods=['POST'])
@limiter.limit("5 per second")
def predict():
    try:
        data = request.get_json()
        if not data or 'Close' not in data:
            return jsonify({"error": "Missing 'Close' price"}), 400
        
        close = float(data['Close'])
        volume = float(data.get('Volume', 0))  # Default volume
        
        proba = predictor.predict(close, volume)
        signal = 1 if proba >= 0.6 else 0
        
        return jsonify({
            "signal": signal,
            "confidence": float(proba),
            "features_used": ["Close", "Volume", "rsi", "macd", "bollinger_%"]
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
