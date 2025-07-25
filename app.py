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
    print(f"FATAL: Could not load the model or scaler. Error: {e}")
    exit()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'Close' not in data:
            return jsonify({"error": "Missing 'Close' value in request"}), 400

        close = float(data['Close'])
        volume = float(data.get('Volume', 0))

        proba = predictor.predict(close, volume)
        
        # Enhanced signal interpretation
        if proba >= 0.65:
            signal = 1  # Strong buy signal
            confidence = "high"
        elif proba >= 0.55:
            signal = 1  # Weak buy signal
            confidence = "medium"
        elif proba <= 0.35:
            signal = -1  # Strong sell signal
            confidence = "high"
        elif proba <= 0.45:
            signal = -1  # Weak sell signal
            confidence = "medium"
        else:
            signal = 0  # Neutral
            confidence = "low"

        return jsonify({
            "signal": signal,
            "signal_strength": confidence,
            "probability": f"{proba:.4f}",
            "ready": len(predictor.history) >= 20,
            "message": "Strong buy signal" if signal == 1 and confidence == "high" else 
                      "Weak buy signal" if signal == 1 else
                      "Strong sell signal" if signal == -1 and confidence == "high" else
                      "Weak sell signal" if signal == -1 else "Neutral"
        })
    except ValueError as ve:
        return jsonify({"error": f"Invalid input value: {str(ve)}"}), 400
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({"error": "An internal error occurred during prediction."}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": hasattr(predictor, 'model'),
        "data_points": len(predictor.history)
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
