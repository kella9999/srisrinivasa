import os
import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

pkl_path = "models/btc_3m_xgb.pkl"
onnx_path = "models/btc_3m.onnx"
input_dim = 30  # ← adjust if your model uses a different number of features

# 1) Check that the .pkl exists
if not os.path.exists(pkl_path):
    print(f"❌ ERROR: {pkl_path} not found.")
    exit(1)
print(f"📦 Loading XGBoost model from {pkl_path} ...")
model = joblib.load(pkl_path)

# 2) Convert to ONNX
print("🔄 Converting to ONNX format …")
initial_type = [("input", FloatTensorType([None, input_dim]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

# 3) Save the ONNX file
with open(onnx_path, "wb") as f:
    f.write(onnx_model.SerializeToString())
print(f"✅ Successfully exported ONNX model to {onnx_path}")
