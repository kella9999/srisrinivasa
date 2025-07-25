# models/train.py
import os
import pickle
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import logging
# Make sure the utils module can be found
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.feature_engineer import add_technical_indicators


# Configuration
CSV_FILE = "BTC_3m_2020-2023.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

def generate_synthetic_data():
    """Generate realistic synthetic BTC price data"""
    np.random.seed(42)
    days = 365 * 3
    base_price = 30000
    daily_volatility = 0.02
    daily_changes = np.random.normal(0, daily_volatility, days)
    cumulative_trend = np.cumsum(daily_changes)
    minute_prices = base_price * (1 + np.repeat(cumulative_trend, 96))
    minute_volatility = daily_volatility / np.sqrt(96)
    minute_prices *= (1 + np.random.normal(0, minute_volatility, len(minute_prices)))
    for i in range(100, len(minute_prices), 1440):
        minute_prices[i:i+100] *= 1.02
    for i in range(500, len(minute_prices), 2880):
        minute_prices[i:i+150] *= 0.98
    dates = pd.date_range("2020-01-01", periods=len(minute_prices), freq="15min")
    df = pd.DataFrame({
        'Open': minute_prices * 0.9998,
        'High': minute_prices * 1.0005,
        'Low': minute_prices * 0.9995,
        'Close': minute_prices,
        'Volume': np.random.lognormal(12, 0.8, len(minute_prices))
    }, index=dates)
    return df

def download_dataset():
    """Get data from Yahoo Finance or generate synthetic"""
    if not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0:
        try:
            logging.info("Downloading live BTC data from Yahoo Finance...")
            df = yf.download("BTC-USD",
                           start="2020-01-01",
                           end="2023-12-31",
                           interval="15m",
                           progress=False)
            if len(df) < 1000:
                raise ValueError("Insufficient data points")
            df.to_csv(CSV_FILE)
            logging.info(f"Downloaded {len(df)} live data points")
        except Exception as e:
            logging.warning(f"Download failed: {str(e)}. Generating synthetic data...")
            df = generate_synthetic_data()
            df.to_csv(CSV_FILE)
            logging.info(f"Generated {len(df)} synthetic data points")

def prepare_data():
    """Load and prepare training data using the unified feature engineer."""
    df = pd.read_csv(CSV_FILE, index_col=0, parse_dates=True)
    
    # Use the centralized feature engineering function from utils
    df_features = add_technical_indicators(df)
    
    # Define target: price moves up in the next period
    df_features['target'] = (df_features['Close'].shift(-1) > df_features['Close']).astype(int)
    df_features = df_features.dropna()
    
    X = df_features.drop('target', axis=1)
    y = df_features['target']
    
    logging.info(f"Using {len(X.columns)} features: {X.columns.tolist()}")
    return X, y

def train_model(X, y):
    """Train and evaluate XGBoost model"""
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = XGBClassifier(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        objective='binary:logistic',
        eval_metric='logloss',
        early_stopping_rounds=25,
        random_state=42
    )
    
    logging.info("Training model...")
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False # Set to True for detailed training logs
    )
    
    y_pred = model.predict(X_test_scaled)
    logging.info("\n=== Model Performance ===")
    logging.info(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    logging.info("Classification Report:")
    logging.info(classification_report(y_test, y_pred))
    
    return model, scaler

def export_model(model, scaler):
    """Export model to native XGBoost JSON and save the scaler."""
    model_path = os.path.join(MODEL_DIR, "model.json")
    model.save_model(model_path)
    
    scaler_path = os.path.join(MODEL_DIR, "btc_3m_scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
        
    logging.info(f"\n✅ Model saved to {model_path}")
    logging.info(f"✅ Scaler saved to {scaler_path}")

if __name__ == "__main__":
    try:
        download_dataset()
        X, y = prepare_data()
        model, scaler = train_model(X, y)
        export_model(model, scaler)
        logging.info("\nTraining pipeline completed successfully!")
    except Exception as e:
        logging.error(f"Training failed: {str(e)}", exc_info=True)
        raise
