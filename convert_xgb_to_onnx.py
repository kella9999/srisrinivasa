import xgboost as xgb
import joblib
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# --- Load your trained XGBoost model (.pkl) ---
model = joblib.load("models/btc_3m_xgb.pkl")

# --- Define input shape: Adjust if needed (e.g., 30 features) ---
input_dim = 30
initial_type = [("input", FloatTensorType([None, input_dim]))]

# --- Convert to ONNX ---
onnx_model = convert_sklearn(model, initial_types=initial_type)

# --- Save ONNX model ---
with open("models/btc_3m.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("✅ Successfully converted XGBoost model to ONNX!")
