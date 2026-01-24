"""Preprocessing script for training data."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import logging
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURE_COLUMNS
from utils import load_all_training_data, save_normalization_stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Preprocess all CSV files and save processed data."""
    logger.info("Starting preprocessing...")
    
    try:
        windows, normalization_stats = load_all_training_data()
        
        # Save processed windows
        processed_file = PROCESSED_DATA_DIR / "windows.npy"
        np.save(processed_file, windows)
        logger.info(f"Saved {len(windows)} windows to {processed_file}")
        
        # Save normalization stats
        stats_file = PROCESSED_DATA_DIR / "normalization_stats.npz"
        save_normalization_stats(normalization_stats, stats_file)
        logger.info(f"Saved normalization stats to {stats_file}")
        
        logger.info("Preprocessing completed successfully")
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()

