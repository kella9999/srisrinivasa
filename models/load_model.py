import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "predictor.joblib")

def load_model():
    """Loads the model with verification"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file missing at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

# Test load
if __name__ == "__main__":
    model = load_model()
    print("✅ Model loaded successfully!")
