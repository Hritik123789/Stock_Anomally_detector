import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2):
        super(LSTMAutoencoder, self).__init__()
        
        # encoder — compresses the sequence
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # decoder — reconstructs the sequence
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=input_size,
            num_layers=num_layers,
            batch_first=True
        )
    
    def forward(self, x):
        # x shape: (batch, 30, 5)
        
        # encode
        _, (hidden, _) = self.encoder(x)
        
        # repeat hidden state across all 30 timesteps
        repeated = hidden[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        
        # decode
        out, _ = self.decoder(repeated)
        
        # out shape: (batch, 30, 5) — same as input
        return out