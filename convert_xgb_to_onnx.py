import os
import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

pkl_path = "models/btc_3m_xgb.pkl"
onnx_path = "models/btc_3m.onnx"
input_dim = 30  # 🚨 Adjust this if your input has different number of features

# 1. Check if the model file exists
if not os.path.exists(pkl_path):
    print(f"❌ ERROR: {pkl_path} not found.")
    exit(1)

# 2. Load the model
print(f"📦 Loading model from {pkl_path}...")
try:
    model = joblib.load(pkl_path)
    print("✅ Model loaded:", type(model))
except Exception as e:
    print("❌ Failed to load model:", e)
    exit(1)

# 3. Convert to ONNX
print("🔄 Converting model to ONNX...")
try:
    initial_type = [("input", FloatTensorType([None, input_dim]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"✅ ONNX model saved to {onnx_path}")
except Exception as e:
    print("❌ ONNX conversion failed:", e)
