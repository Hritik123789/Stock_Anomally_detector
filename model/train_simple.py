import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from lstm_model import LSTMAutoencoder

EPOCHS = 20
BATCH_SIZE = 32
LR = 0.001
THRESHOLD_PERCENTILE = 90  # Lower = more sensitive

def train():
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    X = np.load("X_windows.npy").astype(np.float32)
    X_tensor = torch.tensor(X).to(device)
    
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = LSTMAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    print("Training...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for (batch,) in loader:
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{EPOCHS} — loss: {avg_loss:.6f}")
    
    # compute reconstruction error on all windows
    model.eval()
    with torch.no_grad():
        reconstructed = model(X_tensor)
        errors = torch.mean((X_tensor - reconstructed) ** 2, dim=(1, 2))
        errors = errors.cpu().numpy()
    
    # set threshold at specified percentile
    threshold = float(np.percentile(errors, THRESHOLD_PERCENTILE))
    print(f"\nAnomaly threshold ({THRESHOLD_PERCENTILE}th percentile): {threshold:.6f}")
    print(f"Windows flagged as anomalies: {(errors > threshold).sum()}")
    
    # save model and threshold
    torch.save({
        "model_state": model.state_dict(),
        "threshold": threshold
    }, "lstm_autoencoder.pt")
    print("Model saved to lstm_autoencoder.pt (trained on cuda)")

if __name__ == "__main__":
    train()
