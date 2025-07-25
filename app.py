# app.py
from flask import Flask, request, jsonify
from core.predictor import CryptoPredictor
import os

app = Flask(__name__)

# Initialize predictor
try:
    predictor = CryptoPredictor(
        model_path="models/model.json",
        scaler_path="models/btc_3m_scaler.pkl"
    )
except Exception as e:
    print(f"Failed to initialize predictor: {e}")
    # If the model can't load, we should not start the app.
    # Or handle it gracefully, but for now, exiting is safer.
    exit()


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'Close' not in data:
            return jsonify({"error": "Missing 'Close' value in request"}), 400

        close = float(data['Close'])
        volume = float(data.get('Volume', 0)) # Volume is optional

        proba = predictor.predict(close, volume)
        signal = 1 if proba >= 0.6 else 0

        return jsonify({
            "signal": signal,
            "confidence": f"{proba:.4f}", # Format confidence for readability
            "ready": len(predictor.history) >= 20
        })
    except Exception as e:
        # Log the full error to the console for debugging
        print(f"Error during prediction: {str(e)}")
        return jsonify({"error": "An internal error occurred during prediction."}), 500

if __name__ == '__main__':
    # The port should be configured based on environment variables for deployment
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
