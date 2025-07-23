import joblib
import urllib.request
import os

MODEL_URL = "https://gist.githubusercontent.com/ai-temp-models/9cf5.../raw/predictor.joblib"

def load_model():
    if not os.path.exists("models/predictor.joblib"):
        urllib.request.urlretrieve(MODEL_URL, "models/predictor.joblib")
    return joblib.load("models/predictor.joblib")
