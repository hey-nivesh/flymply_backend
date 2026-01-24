"""PyTorch LSTM Autoencoder model for turbulence detection."""
import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    """LSTM Autoencoder for time-series anomaly detection."""
    
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, dropout=0.1):
        """
        Initialize LSTM Autoencoder.
        
        Args:
            input_size: Number of input features (default: 6)
            hidden_size: Hidden dimension size (default: 64)
            num_layers: Number of LSTM layers (default: 2)
            dropout: Dropout rate (default: 0.1)
        """
        super(LSTMAutoencoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Encoder
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Decoder
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection
        self.output_layer = nn.Linear(hidden_size, input_size)
        
    def forward(self, x):
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Reconstructed tensor of shape (batch_size, sequence_length, input_size)
        """
        batch_size = x.size(0)
        
        # Encode
        encoded, (hidden, cell) = self.encoder(x)
        
        # Use the last hidden state as initial state for decoder
        # Repeat it for each time step
        decoder_input = encoded[:, -1:, :]  # Take last time step
        decoder_input = decoder_input.repeat(1, x.size(1), 1)  # Repeat for all time steps
        
        # Decode
        decoded, _ = self.decoder(decoder_input, (hidden, cell))
        
        # Project to output dimension
        output = self.output_layer(decoded)
        
        return output
    
    def encode(self, x):
        """
        Encode input to latent representation.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Encoded tensor
        """
        encoded, _ = self.encoder(x)
        return encoded
    
    def compute_reconstruction_error(self, x):
        """
        Compute MSE reconstruction error.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            MSE error per sample
        """
        reconstructed = self.forward(x)
        mse = nn.functional.mse_loss(reconstructed, x, reduction='none')
        # Average over sequence length and features, keep batch dimension
        mse = mse.mean(dim=(1, 2))
        return mse

