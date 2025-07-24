import pickle
from xgboost import XGBClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from onnxruntime.quantization import quantize_dynamic, QuantType
import os

# Paths
model_path = "models/btc_3m_xgb.pkl"
scaler_path = "models/btc_3m_scaler.pkl"
onnx_unquant_path = "models/btc_3m_unquant.onnx"
onnx_quant_path = "models/btc_3m_quant.onnx"

# Load model and scaler
with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

# Convert to ONNX
initial_type = [('float_input', FloatTensorType([None, len(scaler.feature_names_in_)]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

# Save unquantized model
with open(onnx_unquant_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

# Quantize and save
quantize_dynamic(onnx_unquant_path, onnx_quant_path, weight_type=QuantType.QInt8)
print("✅ Quantized ONNX model saved:", onnx_quant_path)
