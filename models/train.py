import os
import pickle
import pandas as pd
import numpy as np
import yfinance as yf
import ta
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from utils.feature_engineer import add_technical_indicators


# Config
DATA_URL = "https://storage.googleapis.com/ai-dev-public-datasets/BTC_3m_2020-2023.csv"
CSV_FILE = "BTC_3m_2020-2023.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def download_dataset():
    if not os.path.exists(CSV_FILE):
        print("Downloading dataset...")
        r = requests.get(DATA_URL)
        with open(CSV_FILE, 'wb') as f:
            f.write(r.content)

def add_technical_indicators(df):
    df = df.copy()
    # Price Features
    df['returns'] = df['Close'].pct_change()
    
    # Momentum
    df['rsi'] = ta.momentum.rsi(df['Close'], window=14)
    df['stoch_%k'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'], window=14)
    df['macd'] = ta.trend.macd_diff(df['Close'])
    
    # Volatility
    bb = ta.volatility.BollingerBands(df['Close'], window=5, window_dev=2)
    df['bollinger_%'] = (df['Close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
    
    # Volume
    df['volume_ma'] = df['Volume'].rolling(window=5).mean()
    
    return df.dropna()

def prepare_data(df):
    df = add_technical_indicators(df)
    
    # Target: Next candle direction (1=up, 0=down)
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Selected features (matches predictor.py)
    features = [
        'Close', 'Volume', 
        'rsi', 'macd', 'bollinger_%',
        'returns', 'volume_ma'
    ]
    
    X = df[features]
    y = df['target'].dropna()
    X = X.loc[y.index]
    
    return X, y

def train_xgboost(X, y):
    # Time-based split (no shuffling)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Optimized model (from live deployment)
    model = XGBClassifier(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.75,
        colsample_bytree=0.75,
        gamma=0.2,
        reg_alpha=0.1,
        reg_lambda=0.1,
        objective='binary:logistic',
        eval_metric='logloss',
        tree_method='hist',
        early_stopping_rounds=25,
        random_state=42
    )
    
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=True
    )
    
    # Evaluation
    y_pred = model.predict(X_test_scaled)
    print("\nModel Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print(classification_report(y_test, y_pred))
    
    return model, scaler

def export_onnx(model, scaler):
     # Add quantization
    quantize_dynamic(
        "models/btc_3m.onnx",
        "models/btc_3m_quant.onnx",
        weight_type=QuantType.QInt8
    )
    print("Model quantized successfully")
    # Convert to ONNX
    initial_type = [('float_input', FloatTensorType([None, scaler.mean_.shape[0]]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    
    # Save unquantized
    onnx_path = os.path.join(MODEL_DIR, "btc_3m_unquantized.onnx")
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    
    # Quantize (for production)
    quant_path = os.path.join(MODEL_DIR, "btc_3m.onnx")
    quantize_dynamic(
        onnx_path,
        quant_path,
        weight_type=QuantType.QInt8
    )
    
    # Save scaler
    scaler_path = os.path.join(MODEL_DIR, "btc_3m_scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    
    print(f"\nModel exported to {quant_path} (Size: {os.path.getsize(quant_path)/1024:.1f}KB)")

if __name__ == "__main__":
    download_dataset()
    df = pd.read_csv(CSV_FILE)
    X, y = prepare_data(df)
    model, scaler = train_xgboost(X, y)
    export_onnx(model, scaler)
