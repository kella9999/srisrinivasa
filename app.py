from flask import Flask, jsonify
from data.data_fetcher import fetch_candle_data
from utils.predictor import CryptoPredictor

app = Flask(__name__)
predictor = CryptoPredictor()

@app.route("/predict")
def predict():
    data = fetch_candle_data()
    last_close = data["Close"].iloc[-1]
    return jsonify({
        "last_price": last_close,
        "prediction": predictor.predict(last_close),
        "confidence": 85  # Temporary mock value
    })

if __name__ == "__main__":
    app.run()
