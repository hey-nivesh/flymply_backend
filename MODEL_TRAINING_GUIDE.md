# LSTM Model Training & Fine-Tuning Guide

## Overview

This guide explains how to train, fine-tune, and manage the LSTM Autoencoder model for turbulence prediction.

## Current Status

- ✅ **Normalization Stats**: Available at `models/saved/normalization_stats.npz`
- ❌ **LSTM Model**: Missing at `models/saved/lstm_autoencoder.pth`
- ✅ **Fallback**: App uses hackathon ML service when LSTM model is unavailable

## Model Structure

```
flymply_backend/
├── models/
│   ├── saved/
│   │   ├── lstm_autoencoder.pth      ← Trained LSTM model (TO BE CREATED)
│   │   ├── normalization_stats.npz   ← Normalization stats (EXISTS)
│   │   └── training_scores.npy      ← Training scores (TO BE CREATED)
│   └── turbulence_model.pkl         ← Hackathon RandomForest model
├── model.py                          ← LSTM model architecture
├── scripts/
│   ├── preprocess.py                 ← Data preprocessing
│   ├── train.py                      ← Training script
│   └── compute_training_scores.py    ← Score computation
└── utils.py                          ← Utility functions
```

## Step 1: Prepare Training Data

### Option A: Use Existing Data

If you have flight tracking data in `data/raw/`:

```bash
# Data should be in one of these formats:
# - CSV files: *.csv
# - Compressed CSV: *.csv.tar
# - Compressed JSON: *.json.tar
```

### Option B: Generate Synthetic Data (For Testing)

Create a test data generator:

```python
# scripts/generate_test_data.py
import numpy as np
import pandas as pd
from pathlib import Path

def generate_synthetic_data(n_samples=10000, output_path="data/raw/synthetic_data.csv"):
    """Generate synthetic flight tracking data for testing."""
    np.random.seed(42)
    
    data = {
        'altitude': np.random.uniform(30000, 42000, n_samples),  # feet
        'velocity': np.random.uniform(400, 550, n_samples),       # knots
        'vertical_rate': np.random.normal(0, 500, n_samples),      # fpm
        'u_wind': np.random.normal(0, 20, n_samples),            # m/s
        'v_wind': np.random.normal(0, 20, n_samples),            # m/s
        'temperature': np.random.uniform(-60, -30, n_samples)      # Celsius
    }
    
    df = pd.DataFrame(data)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {n_samples} samples to {output_path}")

if __name__ == "__main__":
    generate_synthetic_data()
```

## Step 2: Preprocess Data

```bash
cd flymply_backend
python scripts/preprocess.py
```

This will:
- Load data from `data/raw/`
- Map columns to features
- Create sliding windows
- Save processed windows to `data/processed/windows.npy`
- Save normalization stats to `models/saved/normalization_stats.npz`

## Step 3: Train the LSTM Model

### Basic Training

```bash
python scripts/train.py
```

This will:
- Load preprocessed windows
- Train LSTM autoencoder
- Save model to `models/saved/lstm_autoencoder.pth`
- Save normalization stats (if not already saved)

### Training Parameters

Edit `config.py` or set environment variables:

```python
# config.py
WINDOW_SIZE = 50          # Sequence length
BATCH_SIZE = 32           # Batch size
EPOCHS = 50               # Number of epochs
LEARNING_RATE = 0.001     # Learning rate
HIDDEN_SIZE = 64          # LSTM hidden size
NUM_LAYERS = 2            # Number of LSTM layers
```

### Training Output

Expected output:
```
INFO:__main__:Starting training...
INFO:__main__:Loading training data...
INFO:__main__:Loaded 1000 training windows
INFO:__main__:Using device: cuda (or cpu)
INFO:__main__:Training for 50 epochs...
INFO:__main__:Epoch [1/50], Loss: 0.123456
INFO:__main__:Epoch [10/50], Loss: 0.045678
...
INFO:__main__:Model saved to models/saved/lstm_autoencoder.pth
INFO:__main__:Training completed. Final loss: 0.012345, Windows: 1000
```

## Step 4: Compute Training Scores

After training, compute anomaly scores for all training data:

```bash
python scripts/compute_training_scores.py
```

This will:
- Load the trained model
- Compute anomaly scores for all training windows
- Save scores to `models/saved/training_scores.npy`

**Why this is needed**: The scores are used to convert anomaly scores to probabilities during inference.

## Step 5: Verify Model Works

Test the model:

```bash
python test_predict.py
```

Or test via API:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"window": [[...50 rows of 6 features...]]}'
```

## Fine-Tuning the Model

### Method 1: Continue Training (Transfer Learning)

Create a fine-tuning script:

```python
# scripts/finetune.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from config import MODEL_PATH, LEARNING_RATE, BATCH_SIZE
from model import LSTMAutoencoder
from utils import load_all_training_data

def finetune_model(
    pretrained_path=MODEL_PATH,
    new_data_path=None,
    epochs=10,
    learning_rate=0.0001,  # Lower LR for fine-tuning
    save_path=None
):
    """Fine-tune a pre-trained model on new data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load pre-trained model
    model = LSTMAutoencoder().to(device)
    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    print(f"Loaded pre-trained model from {pretrained_path}")
    
    # Load new training data
    if new_data_path:
        windows = np.load(new_data_path)
    else:
        windows, _ = load_all_training_data()
    
    windows_tensor = torch.FloatTensor(windows)
    dataset = TensorDataset(windows_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Use lower learning rate for fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Fine-tuning loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            reconstructed = model(x)
            loss = criterion(reconstructed, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
    
    # Save fine-tuned model
    save_path = save_path or pretrained_path.replace('.pth', '_finetuned.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Fine-tuned model saved to {save_path}")
    
    return model

if __name__ == "__main__":
    finetune_model(epochs=10, learning_rate=0.0001)
```

### Method 2: Hyperparameter Tuning

Create a hyperparameter tuning script:

```python
# scripts/hyperparameter_tune.py
import itertools
from scripts.train import train_model
from config import HIDDEN_SIZE, NUM_LAYERS, LEARNING_RATE

def grid_search():
    """Perform grid search over hyperparameters."""
    hidden_sizes = [32, 64, 128]
    num_layers = [1, 2, 3]
    learning_rates = [0.001, 0.0005, 0.0001]
    
    best_loss = float('inf')
    best_params = None
    
    for hidden_size, num_layers, lr in itertools.product(hidden_sizes, num_layers, learning_rates):
        print(f"\nTesting: hidden_size={hidden_size}, num_layers={num_layers}, lr={lr}")
        
        # Temporarily modify config
        import config
        config.HIDDEN_SIZE = hidden_size
        config.NUM_LAYERS = num_layers
        config.LEARNING_RATE = lr
        
        try:
            model, loss, _ = train_model()
            if loss < best_loss:
                best_loss = loss
                best_params = (hidden_size, num_layers, lr)
                print(f"New best: Loss={loss:.6f}")
        except Exception as e:
            print(f"Error: {e}")
    
    print(f"\nBest parameters: {best_params}, Loss: {best_loss:.6f}")

if __name__ == "__main__":
    grid_search()
```

### Method 3: Incremental Learning

Train on new data while preserving old knowledge:

```python
# scripts/incremental_train.py
def incremental_train(new_data_path, old_model_path, epochs=5):
    """Train on new data while preserving old model knowledge."""
    # Load old model
    model = load_model(old_model_path)
    
    # Freeze early layers (optional)
    for param in list(model.encoder.parameters())[:-2]:  # Freeze all but last 2 layers
        param.requires_grad = False
    
    # Train only on new data with lower learning rate
    # ... (similar to finetune.py)
```

## Model Management Best Practices

### 1. Version Control

Save models with version numbers:

```python
# Save with version
version = "v1.0"
model_path = f"models/saved/lstm_autoencoder_{version}.pth"
torch.save(model.state_dict(), model_path)
```

### 2. Model Checkpoints

Save checkpoints during training:

```python
# In training loop
if (epoch + 1) % 10 == 0:
    checkpoint_path = f"models/checkpoints/checkpoint_epoch_{epoch+1}.pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
```

### 3. Model Evaluation

Create evaluation metrics:

```python
# scripts/evaluate.py
def evaluate_model(model_path, test_data_path):
    """Evaluate model on test data."""
    model = load_model(model_path)
    test_windows = np.load(test_data_path)
    
    model.eval()
    with torch.no_grad():
        scores = []
        for window in test_windows:
            score = compute_anomaly_score(model, window)
            scores.append(score)
    
    print(f"Mean score: {np.mean(scores):.6f}")
    print(f"Std score: {np.std(scores):.6f}")
    print(f"Min score: {np.min(scores):.6f}")
    print(f"Max score: {np.max(scores):.6f}")
```

## Using Normalization Stats

The `normalization_stats.npz` file contains:
- `mean`: Mean values for each feature
- `std`: Standard deviation for each feature

**How it's used**:
1. During training: Normalize training data
2. During inference: Normalize input windows before prediction

**Location**: `models/saved/normalization_stats.npz`

## Quick Start: Train Model Now

```bash
# 1. Ensure you have data in data/raw/
cd flymply_backend

# 2. Preprocess (if not done)
python scripts/preprocess.py

# 3. Train model
python scripts/train.py

# 4. Compute scores
python scripts/compute_training_scores.py

# 5. Test
python test_predict.py
```

## Troubleshooting

### Error: "No training windows found"
- **Solution**: Run `preprocess.py` first or add data to `data/raw/`

### Error: "Model not found"
- **Solution**: Train the model first using `scripts/train.py`

### Error: "CUDA out of memory"
- **Solution**: Reduce `BATCH_SIZE` in `config.py` or use CPU

### Low accuracy after training
- **Solution**: 
  - Increase `EPOCHS`
  - Adjust `LEARNING_RATE`
  - Add more training data
  - Try different `HIDDEN_SIZE` or `NUM_LAYERS`

## Next Steps

1. ✅ Train the LSTM model: `python scripts/train.py`
2. ✅ Compute training scores: `python scripts/compute_training_scores.py`
3. ✅ Test the model: `python test_predict.py`
4. ✅ Fine-tune if needed: Use `scripts/finetune.py`
5. ✅ Deploy: The app will automatically use the trained model

## Current Workaround

Until the LSTM model is trained, the app automatically falls back to the hackathon ML service (RandomForest + Gemini). This allows the API to work immediately while you train the LSTM model.

