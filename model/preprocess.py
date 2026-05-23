import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle

WINDOW_SIZE = 30

def load_and_scale(csv_path="../data/aapl_1m.csv"):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    
    # Use only 5 OHLCV features
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = df[features].values
    
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)  # shape: (N, 5)
    
    # save scaler
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    return scaled, scaler

def create_windows(data, window_size=WINDOW_SIZE):
    X = []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])
    return np.array(X)  # shape: (N, 30, 5)

if __name__ == "__main__":
    scaled, scaler = load_and_scale()
    print(f"Scaled data shape: {scaled.shape}")
    
    X = create_windows(scaled)
    print(f"Windows shape: {X.shape}")
    
    # save for training
    np.save("X_windows.npy", X)
    print("Saved X_windows.npy")
