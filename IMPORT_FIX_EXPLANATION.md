# Import Error Fix - Explanation

## The Problem

You were getting this error:
```
ImportError: cannot import name 'compute_anomaly_score' from 'utils'
```

## Root Cause

The issue was caused by a **naming conflict**:

1. **Original Setup**: There was a `utils.py` file (a Python module) containing functions like:
   - `compute_anomaly_score()`
   - `compute_probability()`
   - `get_severity()`
   - `get_confidence()`
   - `load_normalization_stats()`

2. **The Conflict**: A `utils/` directory was created with an `__init__.py` file, which made Python treat `utils` as a **package** instead of a **module**.

3. **What Happened**: When `app.py` tried to import from `utils`, Python found the `utils/` directory first and tried to import from `utils/__init__.py` instead of `utils.py`. Since `__init__.py` didn't have those functions, the import failed.

## The Solution

1. **Removed the conflicting `utils/` directory** - Now Python can properly import from `utils.py` module
2. **Renamed hackathon utilities** - Moved `utils/feature_engineering.py` to `hackathon_utils/feature_engineering.py` to avoid conflicts
3. **Updated imports** - Changed `from utils.feature_engineering` to `from hackathon_utils.feature_engineering` in `app.py`

## Current Structure

```
flymply_backend/
├── utils.py                    ← Original module with compute_anomaly_score, etc.
├── hackathon_utils/            ← New hackathon utilities (no conflict)
│   ├── __init__.py
│   └── feature_engineering.py
└── app.py                      ← Imports from both utils.py and hackathon_utils/
```

## How It Works Now

### Original Functions (from utils.py)
```python
from utils import (
    compute_anomaly_score,    # ✓ Works - imports from utils.py
    compute_probability,      # ✓ Works - imports from utils.py
    get_severity,            # ✓ Works - imports from utils.py
    get_confidence,          # ✓ Works - imports from utils.py
    load_normalization_stats  # ✓ Works - imports from utils.py
)
```

### Hackathon Functions (from hackathon_utils/)
```python
from hackathon_utils.feature_engineering import FeatureEngineer  # ✓ Works
```

## Where `compute_anomaly_score` Comes From

The `compute_anomaly_score` function is defined in **`utils.py`** at line 418:

```python
def compute_anomaly_score(model: torch.nn.Module, window: np.ndarray, device: str = "cpu") -> float:
    """
    Compute anomaly score for a single window.
    
    Args:
        model: Trained PyTorch model
        window: Input window of shape (window_size, 6)
        device: Device to run computation on
        
    Returns:
        Anomaly score (MSE)
    """
    model.eval()
    
    # Convert to tensor
    window_tensor = torch.FloatTensor(window).unsqueeze(0).to(device)
    
    with torch.no_grad():
        reconstructed = model(window_tensor)
        mse = torch.nn.functional.mse_loss(reconstructed, window_tensor, reduction='mean')
        score = mse.item()
    
    return score
```

## Execution Flow

1. **App starts** → `app.py` imports from `utils.py` ✓
2. **Prediction request** → Uses `compute_anomaly_score()` from `utils.py` ✓
3. **Hackathon endpoints** → Use `FeatureEngineer` from `hackathon_utils/` ✓

## Testing

To verify the fix works:

```bash
# Test import
python -c "from utils import compute_anomaly_score; print('✓ Import works!')"

# Test app import (if dependencies installed)
python -c "import app; print('✓ App imports successfully!')"
```

## Next Steps

If you still see errors, they're likely dependency-related (not import structure):

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Activate virtual environment** (if using one):
   ```bash
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

The import structure is now correct! ✅

