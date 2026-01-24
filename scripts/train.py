"""Training script for LSTM Autoencoder."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
from config import (
    MODEL_PATH, PROCESSED_DATA_DIR, WINDOW_SIZE, BATCH_SIZE, 
    EPOCHS, LEARNING_RATE, HIDDEN_SIZE, NUM_LAYERS, NUM_FEATURES, NORM_STATS_PATH
)
from model import LSTMAutoencoder
from utils import load_all_training_data, save_normalization_stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model():
    """Train the LSTM Autoencoder model."""
    logger.info("Starting training...")
    
    # Load data
    logger.info("Loading training data...")
    windows, normalization_stats = load_all_training_data()
    
    # Save normalization stats for inference
    save_normalization_stats(normalization_stats, Path(NORM_STATS_PATH))
    logger.info(f"Saved normalization stats to {NORM_STATS_PATH}")
    
    if len(windows) == 0:
        raise ValueError("No training windows found")
    
    logger.info(f"Loaded {len(windows)} training windows")
    
    # Convert to PyTorch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    windows_tensor = torch.FloatTensor(windows)
    dataset = TensorDataset(windows_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize model
    model = LSTMAutoencoder(
        input_size=NUM_FEATURES,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    logger.info(f"Training for {EPOCHS} epochs...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            x = batch[0].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            reconstructed = model(x)
            loss = criterion(reconstructed, x)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.6f}")
    
    # Save model
    torch.save(model.state_dict(), MODEL_PATH)
    logger.info(f"Model saved to {MODEL_PATH}")
    
    return model, avg_loss, len(windows)


if __name__ == "__main__":
    try:
        model, final_loss, num_windows = train_model()
        logger.info(f"Training completed. Final loss: {final_loss:.6f}, Windows: {num_windows}")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

