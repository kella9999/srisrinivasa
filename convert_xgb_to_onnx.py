import os
import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Set paths
pkl_path = "models/btc_3m_xgb.pkl"
onnx_path = "models/btc_3m.onnx"
input_dim = 30  # 🔁 Change if your model expects different input size

# 1. Check if model exists
if not os.path.exists(pkl_path):
    print(f"❌ ERROR: Model file not found at {pkl_path}")
    exit(1)

# 2. Load the model
print("📦 Loading model...")
try:
    model = joblib.load(pkl_path)
    print("✅ Model loaded:", type(model))
except Exception as e:
    print("❌ Failed to load model:", e)
    exit(1)

# 3. Convert to ONNX
print("🔄 Converting to ONNX format...")
try:
    initial_type = [("input", FloatTensorType([None, input_dim]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"✅ Saved ONNX model to {onnx_path}")
except Exception as e:
    print("❌ Conversion failed:", e)
