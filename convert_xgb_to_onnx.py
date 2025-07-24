import os
import joblib
import onnxmltools
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

# Set paths
pkl_path = "models/btc_3m_xgb.pkl"
onnx_path = "models/btc_3m.onnx"
input_dim = 30  # 🔁 Adjust if your model uses different number of features

# 1. Check model file exists
if not os.path.exists(pkl_path):
    print(f"❌ ERROR: Model file not found at {pkl_path}")
    exit(1)

# 2. Load model
print("📦 Loading model...")
try:
    model = joblib.load(pkl_path)
    print("✅ Model loaded:", type(model))
except Exception as e:
    print("❌ Failed to load model:", e)
    exit(1)

# 3. Convert to ONNX using onnxmltools
print("🔄 Converting to ONNX format with onnxmltools...")
try:
    initial_type = [("input", FloatTensorType([None, input_dim]))]
    onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_type)

    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"✅ ONNX model saved to {onnx_path}")
except Exception as e:
    print("❌ Conversion failed:", e)
