# convert_fixed_features.py
import pickle
import numpy as np
from xgboost import XGBClassifier

# Load the model
with open("models/btc_3m_xgb_updated.pkl", "rb") as f:
    model = pickle.load(f)

# Fix feature names to be ONNX-compatible (f0, f1, f2,...)
booster = model.get_booster()
booster.feature_names = [f"f{i}" for i in range(model.n_features_in_)]
model._Booster = booster  # Update the booster in the model

# Save the fixed model
with open("models/btc_3m_xgb_fixed.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Fixed feature names in model")
print("Attempting ONNX conversion...")

try:
    from onnxmltools.convert import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType
    
    # Convert to ONNX
    initial_type = [('input', FloatTensorType([None, model.n_features_in_]))]
    onnx_model = convert_xgboost(model, initial_types=initial_type)
    
    # Save ONNX model
    with open("models/btc_3m.onnx", "wb") as f:
        onnx_model.SerializeToString()
    print("✅ ONNX conversion successful!")
    
except Exception as e:
    print(f"❌ ONNX conversion failed: {e}")
    print("Using native XGBoost format as fallback...")
    model.save_model("models/model.json")
    print("✅ Saved as models/model.json")
