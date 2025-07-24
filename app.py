# app.py
from flask import Flask, request, jsonify
from predictor import CryptoPredictor
import os

app = Flask(__name__)

# Initialize predictor
predictor = CryptoPredictor(
    model_path="models/model.json",  # Now using native format
    scaler_path="models/btc_3m_scaler.pkl"
)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        close = float(data['Close'])
        volume = float(data.get('Volume', 0))
        
        proba = predictor.predict(close, volume)
        signal = 1 if proba >= 0.6 else 0
        
        return jsonify({
            "signal": signal,
            "confidence": proba,
            "ready": len(predictor.history) >= 20
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
