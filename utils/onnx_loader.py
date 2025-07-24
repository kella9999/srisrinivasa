import os
import onnxruntime as ort

def load_onnx_model(model_path):
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        return None

    try:
        session = ort.InferenceSession(model_path)
        print(f"[INFO] Loaded ONNX model: {model_path}")
        return session
    except Exception as e:
        print(f"[ERROR] Failed to load ONNX model: {e}")
        return None
