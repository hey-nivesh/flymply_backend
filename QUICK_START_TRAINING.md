# Quick Start: Train Your LSTM Model

## Current Status ✅

- ✅ **Normalization Stats**: `models/saved/normalization_stats.npz` (EXISTS)
- ❌ **LSTM Model**: `models/saved/lstm_autoencoder.pth` (MISSING - needs training)
- ✅ **App Fallback**: Uses hackathon ML service when LSTM model is missing

## Quick Fix: Train the Model Now

### Step 1: Check Your Data

```bash
cd flymply_backend
ls data/raw/  # Should have CSV or .tar files
```

### Step 2: Preprocess Data (if needed)

```bash
python scripts/preprocess.py
```

This creates:
- `data/processed/windows.npy` - Training windows
- `models/saved/normalization_stats.npz` - Normalization stats (already exists)

### Step 3: Train the Model

```bash
python scripts/train.py
```

**Expected Output:**
```
INFO:__main__:Starting training...
INFO:__main__:Loaded 1000 training windows
INFO:__main__:Training for 50 epochs...
INFO:__main__:Epoch [1/50], Loss: 0.123456
...
INFO:__main__:Model saved to models/saved/lstm_autoencoder.pth
```

### Step 4: Compute Training Scores

```bash
python scripts/compute_training_scores.py
```

This creates:
- `models/saved/training_scores.npy` - Used for probability calculation

### Step 5: Test It

```bash
python test_predict.py
```

Or restart your Flask app - it will now use the LSTM model instead of the fallback!

## What Happens Now?

**Before Training:**
- `/predict` endpoint → Uses hackathon ML service (RandomForest + Gemini)
- Works but uses simpler model

**After Training:**
- `/predict` endpoint → Uses trained LSTM autoencoder
- More accurate predictions using your actual flight data

## If You Don't Have Training Data

Generate synthetic data for testing:

```python
# Create scripts/generate_test_data.py
import numpy as np
import pandas as pd
from pathlib import Path

def generate_synthetic_data(n_samples=10000):
    np.random.seed(42)
    data = {
        'altitude': np.random.uniform(30000, 42000, n_samples),
        'velocity': np.random.uniform(400, 550, n_samples),
        'vertical_rate': np.random.normal(0, 500, n_samples),
        'u_wind': np.random.normal(0, 20, n_samples),
        'v_wind': np.random.normal(0, 20, n_samples),
        'temperature': np.random.uniform(-60, -30, n_samples)
    }
    df = pd.DataFrame(data)
    Path("data/raw/synthetic_data.csv").parent.mkdir(parents=True, exist_ok=True)
    df.to_csv("data/raw/synthetic_data.csv", index=False)
    print(f"Generated {n_samples} samples")

if __name__ == "__main__":
    generate_synthetic_data()
```

Then run:
```bash
python scripts/generate_test_data.py
python scripts/preprocess.py
python scripts/train.py
python scripts/compute_training_scores.py
```

## File Structure After Training

```
models/saved/
├── lstm_autoencoder.pth      ← Created by train.py
├── normalization_stats.npz  ← Already exists
└── training_scores.npy      ← Created by compute_training_scores.py
```

## Fine-Tuning Later

See `MODEL_TRAINING_GUIDE.md` for:
- Hyperparameter tuning
- Fine-tuning on new data
- Model evaluation
- Best practices

## Need Help?

Check `MODEL_TRAINING_GUIDE.md` for detailed instructions.

