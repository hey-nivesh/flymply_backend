"""Configuration module for turbulence prediction backend."""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
DATA_FOR_MODEL_DIR = BASE_DIR / "data_for_model"  # For .tar files
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models" / "saved"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Server configuration
PORT = int(os.getenv("PORT", 5000))
HOST = os.getenv("HOST", "0.0.0.0")
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# Model paths
MODEL_PATH = os.getenv("MODEL_PATH", str(MODELS_DIR / "lstm_autoencoder.pth"))
SCORES_PATH = os.getenv("SCORES_PATH", str(MODELS_DIR / "training_scores.npy"))
NORM_STATS_PATH = os.getenv("NORM_STATS_PATH", str(MODELS_DIR / "normalization_stats.npz"))

# Model hyperparameters
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", 50))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
EPOCHS = int(os.getenv("EPOCHS", 50))
LEARNING_RATE = float(os.getenv("LR", 0.001))
HIDDEN_SIZE = int(os.getenv("HIDDEN_SIZE", 64))
NUM_LAYERS = int(os.getenv("NUM_LAYERS", 2))

# Feature configuration
FEATURE_COLUMNS = ["altitude", "velocity", "vertical_rate", "u_wind", "v_wind", "temperature"]
NUM_FEATURES = len(FEATURE_COLUMNS)

# Training configuration
TRAIN_TEST_SPLIT = 0.8
RANDOM_SEED = 42

# LLM configuration
GEMMA_MODEL_ID = os.getenv("GEMMA_MODEL_ID", "google/gemma-2b-it")
LORA_ADAPTER_PATH = os.getenv("LORA_ADAPTER_PATH", str(BASE_DIR / "llm" / "adapters" / "gemma_turbulence_advisor"))
HF_TOKEN = os.getenv("HF_TOKEN")

