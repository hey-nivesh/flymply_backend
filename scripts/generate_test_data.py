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