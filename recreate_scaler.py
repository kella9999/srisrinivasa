# recreate_scaler.py
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Create new scaler for 30 features
scaler = StandardScaler()
scaler.fit(np.random.rand(100, 30))  # Match model's expected features

# Save it
with open("models/btc_3m_scaler_fixed.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Created new scaler for 30 features")
