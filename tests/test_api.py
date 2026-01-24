"""Tests for the Flask API."""
import pytest
import numpy as np
from flask import Flask
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import app
from config import WINDOW_SIZE, NUM_FEATURES


@pytest.fixture
def client():
    """Create a test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_health_endpoint(client):
    """Test /health endpoint returns ok."""
    response = client.get('/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'ok'


def test_predict_missing_window(client):
    """Test /predict returns 400 for missing window."""
    response = client.post('/predict', json={})
    assert response.status_code == 400
    data = response.get_json()
    assert 'error' in data
    assert 'window' in data['error'].lower()


def test_predict_invalid_window_type(client):
    """Test /predict returns 400 for invalid window type."""
    response = client.post('/predict', json={"window": "not a list"})
    assert response.status_code == 400
    data = response.get_json()
    assert 'error' in data


def test_predict_empty_window(client):
    """Test /predict returns 400 for empty window."""
    response = client.post('/predict', json={"window": []})
    assert response.status_code == 400
    data = response.get_json()
    assert 'error' in data


def test_predict_wrong_window_size(client):
    """Test /predict returns 400 for wrong window size."""
    # Create window with wrong size
    wrong_size = WINDOW_SIZE - 1
    window = [[1.0] * NUM_FEATURES for _ in range(wrong_size)]
    
    response = client.post('/predict', json={"window": window})
    assert response.status_code == 400
    data = response.get_json()
    assert 'error' in data
    assert str(WINDOW_SIZE) in data['error']


def test_predict_wrong_feature_count(client):
    """Test /predict returns 400 for wrong feature count."""
    # Create window with wrong feature count
    window = [[1.0] * (NUM_FEATURES - 1) for _ in range(WINDOW_SIZE)]
    
    response = client.post('/predict', json={"window": window})
    assert response.status_code == 400
    data = response.get_json()
    assert 'error' in data
    assert str(NUM_FEATURES) in data['error']


def test_predict_invalid_data_type(client):
    """Test /predict returns 400 for non-numeric data."""
    window = [["not", "numeric", "data", "here", "please", "fail"] for _ in range(WINDOW_SIZE)]
    
    response = client.post('/predict', json={"window": window})
    assert response.status_code == 400
    data = response.get_json()
    assert 'error' in data


def test_predict_not_json(client):
    """Test /predict returns 400 for non-JSON request."""
    response = client.post('/predict', data="not json")
    assert response.status_code == 400
    data = response.get_json()
    assert 'error' in data


def test_predict_valid_window_structure(client):
    """Test /predict validates window structure correctly."""
    # Create valid window structure (will fail if model not trained, but tests validation)
    window = [[float(i + j) for j in range(NUM_FEATURES)] for i in range(WINDOW_SIZE)]
    
    response = client.post('/predict', json={"window": window})
    
    # This will return 500 if model not found, or 200 if model exists
    # We're mainly testing that the validation passes for correct structure
    assert response.status_code in [200, 500]
    
    # If model exists, check response structure
    if response.status_code == 200:
        data = response.get_json()
        assert 'turbulence_probability' in data
        assert 'severity' in data
        assert 'confidence' in data
        assert 'anomaly_score' in data
        assert isinstance(data['turbulence_probability'], float)
        assert data['severity'] in ['Low', 'Moderate', 'High']
        assert data['confidence'] in ['Low', 'Medium', 'High']
    else:
        # Model not found is acceptable for testing
        data = response.get_json()
        assert 'error' in data


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

