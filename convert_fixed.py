# convert_fixed.py
import pickle
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

# Paths
model_path = "models/btc_3m_xgb.pkl"
scaler_path = "models/btc_3m_scaler.pkl"
onnx_path = "models/btc_3m.onnx"

# Load model
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Get feature count from model
feature_count = len(model.get_booster().feature_names)
print(f"Detected {feature_count} features in model")

# Create new scaler if needed
try:
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    print("Loaded existing scaler")
except:
    print("Creating new scaler")
    scaler = StandardScaler()
    # Fit with dummy data matching feature count
    scaler.fit(np.random.rand(100, feature_count))
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

# Convert to ONNX
initial_type = [('input', FloatTensorType([None, feature_count]))]
onnx_model = convert_xgboost(model, initial_types=initial_type)

# Save ONNX model
with open(onnx_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"✅ Successfully converted to ONNX: {onnx_path}")
print(f"Model size: {os.path.getsize(onnx_path)/1024:.1f} KB")
