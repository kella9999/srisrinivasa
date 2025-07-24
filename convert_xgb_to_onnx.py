# convert_xgb_to_onnx.py
import pickle
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from onnxruntime.quantization import quantize_dynamic, QuantType

# Workaround for XGBoost to ONNX conversion
from xgboost import XGBClassifier
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

# Paths
model_path = "models/btc_3m_xgb.pkl"
scaler_path = "models/btc_3m_scaler.pkl"
onnx_unquant_path = "models/btc_3m_unquant.onnx"
onnx_quant_path = "models/btc_3m_quant.onnx"

def convert_xgboost_to_onnx(model, feature_count):
    # Convert using onnxmltools
    initial_type = [('input', FloatTensorType([None, feature_count]))]
    onnx_model = convert_xgboost(model, initial_types=initial_type)
    return onnx_model

def main():
    # Load model
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    # Create or load scaler
    try:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    except:
        print("Creating new scaler...")
        scaler = StandardScaler()
        scaler.fit(np.random.rand(100, 10))  # Adjust 10 to your feature count
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
    
    # Convert to ONNX
    feature_count = len(scaler.feature_names_in_) if hasattr(scaler, 'feature_names_in_') else 10
    onnx_model = convert_xgboost_to_onnx(model, feature_count)
    
    # Save models
    with open(onnx_unquant_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    
    quantize_dynamic(onnx_unquant_path, onnx_quant_path, weight_type=QuantType.QInt8)
    print(f"✅ Model converted and saved to {onnx_quant_path}")

if __name__ == "__main__":
    main()
