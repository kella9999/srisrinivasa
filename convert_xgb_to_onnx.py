import joblib
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
import os

# === PATHS ===
model_path = "models/btc_3m_xgb.pkl"
onnx_path = "models/btc_3m.onnx"

# === INPUT SHAPE ===
input_dim = 30  # ⚠️ Set this to the number of features your model was trained with

print("📦 Loading model...")
model = joblib.load(model_path)
print("✅ Model loaded:", type(model))

# === ONNX CONVERSION ===
print("🔄 Converting using onnxmltools...")
initial_type = [("input", FloatTensorType([None, input_dim]))]
onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_type)

# === SAVE FILE ===
with open(onnx_path, "wb") as f:
    f.write(onnx_model.SerializeToString())
print(f"✅ Saved ONNX model to {onnx_path}")
