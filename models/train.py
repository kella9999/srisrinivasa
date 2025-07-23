from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import joblib

def train_model():
    """Basic starter model (we'll improve later)"""
    model = GradientBoostingRegressor(n_estimators=50, max_depth=3)
    
    # Mock training data (replace with real data later)
    X = pd.DataFrame([[i] for i in range(100)])
    y = [i*1.05 + 2 for i in range(100)]
    
    model.fit(X, y)
    joblib.dump(model, "models/BTCUSDT_3m.joblib")
    print("Model trained and saved")

if __name__ == "__main__":
    import os
    os.makedirs("models", exist_ok=True)
    train_model()
