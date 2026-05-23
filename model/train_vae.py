import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pickle
from lstm_vae import LSTMVAE

EPOCHS = 30
BATCH_SIZE = 32
LR = 0.001

def vae_loss(recon_x, x, mu, logvar):
    """VAE loss = reconstruction loss + KL divergence."""
    recon_loss = nn.MSELoss()(recon_x, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss / x.size(0)  # Normalize by batch size
    return recon_loss + 0.001 * kl_loss  # Weight KL term

def train():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    X = np.load("X_windows.npy").astype(np.float32)
    X_tensor = torch.tensor(X).to(device)
    
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = LSTMVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    print("Training LSTM-VAE...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for (batch,) in loader:
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss = vae_loss(recon, batch, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{EPOCHS} — loss: {avg_loss:.6f}")
    
    # Compute reconstruction errors on training data
    print("\nComputing reconstruction errors...")
    model.eval()
    all_errors = []
    
    with torch.no_grad():
        for i in range(0, len(X_tensor), BATCH_SIZE):
            batch = X_tensor[i:i+BATCH_SIZE]
            errors, _, _ = model.get_reconstruction_error(batch)
            all_errors.extend(errors.cpu().numpy())
    
    all_errors = np.array(all_errors)
    
    # Compute dynamic thresholds
    mean_error = np.mean(all_errors)
    std_error = np.std(all_errors)
    
    # Multiple threshold levels
    threshold_95 = np.percentile(all_errors, 95)
    threshold_99 = np.percentile(all_errors, 99)
    threshold_3std = mean_error + 3 * std_error
    
    print(f"\nError Statistics:")
    print(f"Mean: {mean_error:.6f}")
    print(f"Std: {std_error:.6f}")
    print(f"95th percentile: {threshold_95:.6f}")
    print(f"99th percentile: {threshold_99:.6f}")
    print(f"Mean + 3*std: {threshold_3std:.6f}")
    
    # Use 95th percentile as primary threshold
    threshold = threshold_95
    anomalies = (all_errors > threshold).sum()
    print(f"\nAnomalies at 95th percentile: {anomalies} / {len(all_errors)}")
    
    # Save model and thresholds
    torch.save({
        'model_state': model.state_dict(),
        'threshold': threshold,
        'mean_error': mean_error,
        'std_error': std_error,
        'threshold_95': threshold_95,
        'threshold_99': threshold_99
    }, "lstm_vae.pt")
    
    print(f"\nModel saved to lstm_vae.pt")
    print(f"Threshold: {threshold:.6f}")

if __name__ == "__main__":
    train()
