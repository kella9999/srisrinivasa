import joblib
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
import os

# === SETTINGS ===
model_path = "models/btc_3m_xgb.pkl"
onnx_output_path = "models/btc_3m.onnx"
input_dim = 30  # <== CHANGE THIS TO MATCH your model's feature size

# === LOAD MODEL ===
print("📦 Loading model...")
model = joblib.load(model_path)
print("✅ Model loaded:", type(model))

# === CONVERT TO ONNX ===
print("🔄 Converting to ONNX...")
initial_type = [("input", FloatTensorType([None, input_dim]))]
onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_type)

# === SAVE ONNX MODEL ===
with open(onnx_output_path, "wb") as f:
    f.write(onnx_model.SerializeToString())
print(f"✅ ONNX model saved: {onnx_output_path}")
