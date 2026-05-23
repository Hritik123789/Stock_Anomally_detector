import torch
import torch.nn as nn

class LSTMVAE(nn.Module):
    """LSTM Variational Autoencoder for anomaly detection."""
    
    def __init__(self, input_size=10, hidden_size=64, latent_size=32, num_layers=2):
        super(LSTMVAE, self).__init__()
        
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        
        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_size, latent_size)
        self.fc_logvar = nn.Linear(hidden_size, latent_size)
        
        # Decoder
        self.fc_decode = nn.Linear(latent_size, hidden_size)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=input_size,
            num_layers=num_layers,
            batch_first=True
        )
    
    def encode(self, x):
        """Encode input to latent distribution."""
        _, (hidden, _) = self.encoder_lstm(x)
        h = hidden[-1]
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, seq_len):
        """Decode latent vector to sequence."""
        h = self.fc_decode(z)
        h = h.unsqueeze(1).repeat(1, seq_len, 1)
        out, _ = self.decoder_lstm(h)
        return out
    
    def forward(self, x):
        """Forward pass."""
        seq_len = x.size(1)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, seq_len)
        return recon, mu, logvar
    
    def get_reconstruction_error(self, x):
        """Get reconstruction error for anomaly detection."""
        recon, mu, logvar = self.forward(x)
        mse = torch.mean((x - recon) ** 2, dim=(1, 2))
        return mse, mu, logvar
