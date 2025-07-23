from flask import Flask, jsonify
from data.live_data import get_live_candles
from models.load_model import load_model

app = Flask(__name__)
model = load_model()

@app.route("/predict")
def predict():
    data = get_live_candles()
    last_candle = data.iloc[-1][["Close", "SMA_10", "RSI_14"]].values
    prediction = model.predict([last_candle])[0]
    
    return jsonify({
        "last_price": last_candle[0],
        "prediction": prediction,
        "confidence": min(99, int(abs(prediction - last_candle[0]) * 10))
    })

if __name__ == "__main__":
    app.run()
