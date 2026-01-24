"""Compute training scores for anomaly detection."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import logging
from config import (
    MODEL_PATH, SCORES_PATH, WINDOW_SIZE, BATCH_SIZE,
    HIDDEN_SIZE, NUM_LAYERS, NUM_FEATURES
)
from model import LSTMAutoencoder
from utils import load_all_training_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_scores():
    """Compute anomaly scores for all training windows."""
    logger.info("Computing training scores...")
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model = LSTMAutoencoder(
        input_size=NUM_FEATURES,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        logger.info(f"Loaded model from {MODEL_PATH}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please train the model first.")
    
    model.eval()
    
    # Load training data
    windows, _ = load_all_training_data()
    logger.info(f"Loaded {len(windows)} training windows")
    
    # Compute scores in batches
    scores = []
    windows_tensor = torch.FloatTensor(windows)
    dataset = torch.utils.data.TensorDataset(windows_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device)
            reconstructed = model(x)
            
            # Compute MSE per sample
            mse = torch.nn.functional.mse_loss(
                reconstructed, x, reduction='none'
            )
            mse = mse.mean(dim=(1, 2))  # Average over sequence and features
            scores.extend(mse.cpu().numpy())
    
    scores = np.array(scores)
    
    # Save scores
    np.save(SCORES_PATH, scores)
    logger.info(f"Saved {len(scores)} training scores to {SCORES_PATH}")
    logger.info(f"Score statistics: min={scores.min():.6f}, max={scores.max():.6f}, "
                f"mean={scores.mean():.6f}, median={np.median(scores):.6f}")
    
    return scores


if __name__ == "__main__":
    try:
        scores = compute_scores()
        logger.info("Training scores computation completed")
    except Exception as e:
        logger.error(f"Failed to compute training scores: {str(e)}")
        raise

