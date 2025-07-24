import joblib
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType

# Load your trained model
model = joblib.load("models/btc_3m_xgb.pkl")

# Set input dimension (features used during training)
input_dim = 30  # ✅ UPDATE THIS if your model used more/less features

# Define input type for ONNX
initial_type = [("input", FloatTensorType([None, input_dim]))]

# Try converting to ONNX
try:
    onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_type)
    with open("models/btc_3m.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())
    print("✅ Successfully converted to ONNX and saved.")
except Exception as e:
    print("❌ Failed to convert:")
    print(e)
