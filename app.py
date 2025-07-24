from flask import Flask, request, jsonify
import pandas as pd
import pickle
from core.predictor import CryptoPredictor
from sklearn.preprocessing import StandardScaler

MODEL_PATH = "models/btc_3m.onnx"
SCALER_PATH = "models/btc_3m_scaler.pkl"

app = Flask(__name__)

# Load scaler and predictor globally
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)
predictor = CryptoPredictor(MODEL_PATH, scaler)

@app.route('/predict', methods=['POST'])
def predict():
    # Expect input as JSON with keys: Close, Volume, etc.
    data = request.get_json()
    input_df = pd.DataFrame([data])
    prediction = predictor.predict(input_df)
    signal = int(prediction[0] > 0.5)  # 1=Up, 0=Down
    return jsonify({"prediction": signal})

if __name__ == '__main__':
    app.run(debug=True)

