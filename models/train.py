import skl2onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import numpy as np
import requests
import os
import yfinance as yf
import ta


CSV_FILE = "BTC_3m_2020-2023.csv"
DATA_URL = "https://storage.googleapis.com/ai-dev-public-datasets/BTC_3m_2020-2023.csv"

def export_model_and_scaler(model, scaler, model_path="models/btc_3m.onnx", scaler_path="models/btc_3m_scaler.pkl"):
    # Define input type for ONNX converter (number of features)
    initial_type = [('float_input', FloatTensorType([None, scaler.mean_.shape[0]]))]

    # Convert to ONNX
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    with open(model_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"Model exported to {model_path}")

    # Save scaler with pickle
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_path}")

def download_csv_if_needed(filename=CSV_FILE, url=DATA_URL):
    if not os.path.exists(filename):
        print(f"Downloading {filename} ...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(filename, "wb") as file:
                file.write(response.content)
            print("CSV file downloaded successfully!")
        else:
            raise Exception(f"Failed to download CSV file. Status code: {response.status_code}")
    else:
        print(f"{filename} already exists, skipping download.")

def load_csv(filename=CSV_FILE):
    print(f"Loading {filename} ...")
    df = pd.read_csv(filename)
    print(f"Loaded {len(df)} rows.")
    return df

def get_live_data(ticker="BTC-USD", interval="3m", period="60d"):
    print(f"Fetching live {interval} {ticker} data for last {period} ...")
    df = yf.download(ticker, interval=interval, period=period)
    print(f"Fetched {len(df)} rows.")
    return df

def add_technical_indicators(df):
    df = df.copy()
    df['rsi'] = ta.momentum.rsi(df['Close'], window=14)
    df['macd'] = ta.trend.macd(df['Close'])
    bb_indicator = ta.volatility.BollingerBands(close=df['Close'], window=5, window_dev=2)
    df['bollinger_%'] = (df['Close'] - bb_indicator.bollinger_lband()) / (bb_indicator.bollinger_hband() - bb_indicator.bollinger_lband())
    df = df.dropna()
    return df

def prepare_features_and_target(df):
    df = add_technical_indicators(df)
    features = df[['Close', 'Volume', 'rsi', 'macd', 'bollinger_%']]
    target = (df['Close'].shift(-1) > df['Close']).astype(int)
    valid_idx = target.dropna().index
    return features.loc[valid_idx], target.loc[valid_idx]

def train_model(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Keep 80% for training, 20% for testing
    split_idx = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Training samples: {len(X_train)}; Testing samples: {len(X_test)}")

    model = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric='logloss'
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=20,
        verbose=True
    )

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification report:\n", classification_report(y_test, y_pred))
    return model, scaler

if __name__ == "__main__":
    # Existing code: download, load, fetch live data, train
    download_csv_if_needed()
    df_csv = load_csv()

    df_live = get_live_data()

    X, y = prepare_features_and_target(df_csv)
    model, scaler = train_model(X, y)

    # Export model & scaler for inference use
    export_model_and_scaler(model, scaler)
