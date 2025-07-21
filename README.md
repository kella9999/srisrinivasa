# Crypto Model Package
Includes:
- 90-day OHLCV data (.csv)
- XGBoost classifier and regressor models (.pkl)

## Usage
```python
import joblib
clf = joblib.load('models/xgb_BTC_1m_classifier.pkl')
reg = joblib.load('models/xgb_BTC_1m_regressor.pkl')
```
