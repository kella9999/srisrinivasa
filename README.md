# Crypto Price Movement Prediction API

This API predicts Bitcoin (BTC) price movements using an XGBoost model trained on historical data and technical indicators.

## API Endpoints

### POST /predict
Predicts price movement direction.

**Request:**
```json
{
    "Close": 50000.0,
    "Volume": 12345.67
}
