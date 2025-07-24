import os
import pickle
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from onnxruntime.quantization import quantize_dynamic, QuantType
import logging

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
    days = 365 * 3  # 3 years
    base_price = 30000
    daily_volatility = 0.02
    
    # Generate daily trend with momentum
    daily_changes = np.random.normal(0, daily_volatility, days)
    cumulative_trend = np.cumsum(daily_changes)
    
    # Convert to 15-minute intervals (96 periods/day)
    minute_prices = base_price * (1 + np.repeat(cumulative_trend, 96))
    
    # Add intraday volatility and noise
    minute_volatility = daily_volatility / np.sqrt(96)
    minute_prices *= (1 + np.random.normal(0, minute_volatility, len(minute_prices)))
    
    # Simulate market patterns
    for i in range(100, len(minute_prices), 1440):  # Add spikes every ~10 days
        minute_prices[i:i+100] *= 1.02  # 2% pump
    for i in range(500, len(minute_prices), 2880):  # Add dips every ~20 days
        minute_prices[i:i+150] *= 0.98  # 2% dump
    
    # Create DataFrame
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
            if len(df) < 1000:  # Fallback if data is incomplete
                raise ValueError("Insufficient data points")
            df.to_csv(CSV_FILE)
            logging.info(f"Downloaded {len(df)} live data points")
        except Exception as e:
            logging.warning(f"Download failed: {str(e)}. Generating synthetic data...")
            df = generate_synthetic_data()
            df.to_csv(CSV_FILE)
            logging.info(f"Generated {len(df)} synthetic data points")

def add_technical_indicators(df):
    """Calculate advanced technical indicators"""
    df = df.copy()
    
    # Price Features
    df['returns'] = df['Close'].pct_change()
    
    # Trend indicators
    df['ema_20'] = ta.trend.ema_indicator(df['Close'], window=20)
    df['macd'] = ta.trend.macd_diff(df['Close'])
    
    # Momentum indicators
    df['rsi'] = ta.momentum.rsi(df['Close'], window=14)
    
    # Volatility indicators
    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['bb_%b'] = (df['Close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
    df['atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    
    # Volume indicators
    df['obv'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    df['volume_ma'] = df['Volume'].rolling(20).mean()
    
    return df.dropna()

def prepare_data():
    """Load and prepare training data"""
    df = pd.read_csv(CSV_FILE, index_col=0, parse_dates=True)
    df = add_technical_indicators(df)
    
    # Target: Price movement in next period (1=up, 0=down)
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Selected features - using only available indicators
    features = [
        'Close', 'Volume',
        'ema_20', 'macd', 'rsi',
        'bb_%b', 'atr', 'obv',
        'volume_ma', 'returns'
    ]
    
    # Ensure all features exist
    available_features = [f for f in features if f in df.columns]
    X = df[available_features]
    y = df['target'].dropna()
    X = X.loc[y.index]  # Align features with targets
    
    logging.info(f"Using features: {available_features}")
    return X, y

def train_model(X, y):
    """Train and evaluate XGBoost model"""
    # Time-series split (no shuffling)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model configuration
    model = XGBClassifier(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=0.1,
        objective='binary:logistic',
        eval_metric='logloss',
        tree_method='hist',
        early_stopping_rounds=25,
        random_state=42
    )
    
    logging.info("Training model...")
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=20
    )
    
    # Evaluation
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    logging.info("\n=== Model Performance ===")
    logging.info(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    logging.info("Classification Report:")
    logging.info(classification_report(y_test, y_pred))
    
    # Feature importance
    logging.info("\nFeature Importances:")
    for name, imp in sorted(zip(X.columns, model.feature_importances_), 
                           key=lambda x: x[1], reverse=True):
        logging.info(f"{name}: {imp:.3f}")
    
    return model, scaler

def export_model(model, scaler):
    """Export model to ONNX format with quantization"""
    # Convert to ONNX
    initial_type = [('float_input', FloatTensorType([None, len(scaler.feature_names_in_)]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    
    # Save unquantized
    unquant_path = os.path.join(MODEL_DIR, "btc_3m_unquant.onnx")
    with open(unquant_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    
    # Quantize
    quant_path = os.path.join(MODEL_DIR, "btc_3m.onnx")
    quantize_dynamic(
        unquant_path,
        quant_path,
        weight_type=QuantType.QInt8
    )
    
    # Save scaler
    with open(os.path.join(MODEL_DIR, "btc_3m_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    
    logging.info(f"\nModel exported to {quant_path}")
    logging.info(f"Quantized size: {os.path.getsize(quant_path)/1024:.1f} KB")

if __name__ == "__main__":
    try:
        # Data pipeline
        download_dataset()
        X, y = prepare_data()
        
        # Training
        model, scaler = train_model(X, y)
        
        # Export
        export_model(model, scaler)
        
        logging.info("Training completed successfully!")
    except Exception as e:
        logging.error(f"Training failed: {str(e)}", exc_info=True)
        raise
