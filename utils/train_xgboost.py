
import pandas as pd
import pandas_ta as ta
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier
import joblib
import os

# === Config ===
symbol = "BTCUSDT"
interval = "3m"
csv_path = f"data/{symbol}_{interval}.csv"
model_path = f"models/{symbol}_{interval}_xgb.pkl"

# === Load and Prepare Data ===
df = pd.read_csv(csv_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.dropna()

# === Add Technical Indicators ===
df['rsi'] = ta.rsi(df['close'], length=14)
df['ema'] = ta.ema(df['close'], length=14)
macd = ta.macd(df['close'])
df['macd'] = macd['MACD_12_26_9']
df['macd_signal'] = macd['MACDs_12_26_9']
df['macd_hist'] = macd['MACDh_12_26_9']
df = df.dropna()

# === Create Target Column ===
df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

# === Feature Columns ===
features = ['rsi', 'ema', 'macd', 'macd_signal', 'macd_hist']
X = df[features]
y = df['target']

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# === Train Model ===
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# === Evaluate ===
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(f"✅ Accuracy: {acc * 100:.2f}%")
print("Confusion Matrix:")
print(cm)

# === Save Model ===
os.makedirs("models", exist_ok=True)
joblib.dump(model, model_path)
print(f"✅ Model saved to {model_path}")
