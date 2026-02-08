import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

class TurbulenceMLModel:
    def __init__(self):
        self.model_path = "models/turbulence_model.pkl"
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Load or create ML model"""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        else:
            self._train_initial_model()
    
    def _train_initial_model(self):
        """Create simple model with synthetic training data"""
        # Generate synthetic training data (hackathon placeholder)
        np.random.seed(42)
        n_samples = 1000
        
        # Features: wind_change, temp_gradient, pressure_drop, wind_speed
        X = np.random.randn(n_samples, 4)
        
        # Label: turbulence probability (0 or 1)
        # High turbulence if: wind_change > 1 OR temp_gradient > 1.5
        y = ((X[:, 0] > 1) | (X[:, 1] > 1.5)).astype(int)
        
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.model.fit(X, y)
        
        # Save model
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.model, self.model_path)
    
    def predict(self, features):
        """
        Predict turbulence probability
        features: dict with wind_change, temp_gradient, pressure_drop, wind_speed
        Returns: probability (0-1)
        """
        X = np.array([[
            features.get('wind_change', 0),
            features.get('temp_gradient', 0),
            features.get('pressure_drop', 0),
            features.get('wind_speed', 0)
        ]])
        
        prob = self.model.predict_proba(X)[0][1]
        return prob

