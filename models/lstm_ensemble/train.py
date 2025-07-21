import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

def train_lstm_ensemble(data_path, n_models=3):
    df = pd.read_csv(data_path)
    closes = df['close'].values.reshape(-1, 1)
    
    # Normalize data
    scaler = MinMaxScaler()
    closes_scaled = scaler.fit_transform(closes)
    
    # Create sequences
    X, y = [], []
    seq_length = 60
    for i in range(seq_length, len(closes_scaled)):
        X.append(closes_scaled[i-seq_length:i])
        y.append(closes_scaled[i])
    X, y = np.array(X), np.array(y)
    
    # Train multiple LSTM models
    models = []
    for _ in range(n_models):
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
            Dropout(0.2),
            LSTM(50),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=10, batch_size=32, verbose=0)
        models.append(model)
    
    return models, scaler
