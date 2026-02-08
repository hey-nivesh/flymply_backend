"""Flask application for turbulence prediction API."""
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any

from config import (
    MODEL_PATH, SCORES_PATH, WINDOW_SIZE, NUM_FEATURES,
    HIDDEN_SIZE, NUM_LAYERS, PORT, HOST, NORM_STATS_PATH,
    GEMMA_MODEL_ID, LORA_ADAPTER_PATH
)
from model import LSTMAutoencoder
from utils import (
    compute_anomaly_score, compute_probability,
    get_severity, get_confidence, load_normalization_stats
)

# Try to import LLM module (may fail if dependencies not installed)
try:
    from llm.gemma_infer import generate_advisory
    LLM_AVAILABLE = True
except ImportError as e:
    import logging
    logging.warning(f"LLM module not available: {e}. Advisory generation will be disabled.")
    LLM_AVAILABLE = False
    def generate_advisory(*args, **kwargs):
        raise RuntimeError("LLM module not available. Install transformers, peft, and bitsandbytes.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import new hackathon services
try:
    from services.weather_service import WeatherService
    from services.ml_service import TurbulenceMLModel
    from services.gemma_service import GemmaService
    from hackathon_utils.feature_engineering import FeatureEngineer
    HACKATHON_SERVICES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Hackathon services not available: {e}")
    HACKATHON_SERVICES_AVAILABLE = False
    WeatherService = None
    TurbulenceMLModel = None
    GemmaService = None
    FeatureEngineer = None

# Initialize hackathon services if available
if HACKATHON_SERVICES_AVAILABLE:
    try:
        weather_service = WeatherService()
        ml_model = TurbulenceMLModel()
        gemma_service = GemmaService()
        feature_engineer = FeatureEngineer()
        logger.info("Hackathon services initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize hackathon services: {e}")
        HACKATHON_SERVICES_AVAILABLE = False

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables for model and training scores
model = None
training_scores = None
normalization_stats = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    """Load the trained model."""
    global model
    if model is None:
        try:
            model = LSTMAutoencoder(
                input_size=NUM_FEATURES,
                hidden_size=HIDDEN_SIZE,
                num_layers=NUM_LAYERS
            ).to(device)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model.eval()
            logger.info(f"Model loaded from {MODEL_PATH}")
        except FileNotFoundError:
            logger.warning(f"LSTM model not found at {MODEL_PATH}. Using hackathon ML service as fallback.")
            model = None  # Will use hackathon service instead
        except Exception as e:
            logger.warning(f"Error loading LSTM model: {str(e)}. Using hackathon ML service as fallback.")
            model = None  # Will use hackathon service instead
    return model


def load_training_scores():
    """Load training scores."""
    global training_scores
    if training_scores is None:
        try:
            training_scores = np.load(SCORES_PATH)
            logger.info(f"Training scores loaded from {SCORES_PATH}")
        except FileNotFoundError:
            logger.warning(f"Training scores not found at {SCORES_PATH}")
            training_scores = np.array([])
    return training_scores


def load_norm_stats():
    """Load normalization statistics."""
    global normalization_stats
    if normalization_stats is None:
        try:
            normalization_stats = load_normalization_stats(Path(NORM_STATS_PATH))
            logger.info(f"Normalization stats loaded from {NORM_STATS_PATH}")
        except FileNotFoundError:
            logger.warning(f"Normalization stats not found at {NORM_STATS_PATH}")
            normalization_stats = None
    return normalization_stats


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"})


@app.route('/predict', methods=['POST'])
def predict():
    """Predict turbulence probability from input window."""
    try:
        # Try to load LSTM model, fallback to hackathon service if not available
        lstm_model = load_model()
        scores = load_training_scores()
        
        # If LSTM model is not available, use hackathon service
        if lstm_model is None:
            if HACKATHON_SERVICES_AVAILABLE:
                logger.info("LSTM model not available, using hackathon ML service")
                return predict_with_hackathon_service()
            else:
                return jsonify({
                    "error": "LSTM model not found and hackathon services unavailable. Please train the model first."
                }), 503
        
        # Validate request
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.get_json()
        
        if "window" not in data:
            return jsonify({"error": "Missing 'window' field in request"}), 400
        
        window = data["window"]
        
        # Validate window format
        if not isinstance(window, list):
            return jsonify({"error": "Window must be a list"}), 400
        
        if len(window) == 0:
            return jsonify({"error": "Window cannot be empty"}), 400
        
        # Convert to numpy array
        try:
            window_array = np.array(window, dtype=np.float32)
        except (ValueError, TypeError) as e:
            return jsonify({"error": f"Invalid window data: {str(e)}"}), 400
        
        # Validate shape
        if window_array.ndim != 2:
            return jsonify({
                "error": f"Window must be 2D array, got {window_array.ndim}D"
            }), 400
        
        if window_array.shape[0] != WINDOW_SIZE:
            return jsonify({
                "error": f"Window length must be {WINDOW_SIZE}, got {window_array.shape[0]}"
            }), 400
        
        if window_array.shape[1] != NUM_FEATURES:
            return jsonify({
                "error": f"Window must have {NUM_FEATURES} features, got {window_array.shape[1]}"
            }), 400
        
        # Check for NaN or infinite values
        if not np.isfinite(window_array).all():
            return jsonify({"error": "Window contains NaN or infinite values"}), 400
        
        # Normalize the window using training statistics
        norm_stats = load_norm_stats()
        if norm_stats is not None:
            mean = norm_stats["mean"]
            std = norm_stats["std"]
            window_array = (window_array - mean) / std
        else:
            logger.warning("No normalization stats found, using raw input")
        
        # Compute anomaly score
        anomaly_score = compute_anomaly_score(lstm_model, window_array, device=str(device))
        
        # Compute probability
        probability = compute_probability(anomaly_score, scores)
        
        # Get severity and confidence
        severity = get_severity(probability)
        confidence = get_confidence(anomaly_score, scores)
        
        response = {
            "turbulence_probability": float(probability),
            "severity": severity,
            "confidence": confidence,
            "anomaly_score": float(anomaly_score)
        }
        
        # Optionally generate advisory if requested
        include_advisory = data.get("include_advisory", False)
        if include_advisory:
            if not LLM_AVAILABLE:
                logger.warning("Advisory requested but LLM module not available")
                response["advisory_error"] = "LLM module not available"
            else:
                try:
                    # Extract time_horizon and altitude_band from request or use defaults
                    time_horizon_min = data.get("time_horizon_min", 10)
                    altitude_band = data.get("altitude_band", "FL360")
                    model_id = data.get("model_id")
                    hf_token = data.get("hf_token")
                    
                    advisory = generate_advisory(
                        probability=probability,
                        severity=severity,
                        confidence=confidence,
                        time_horizon_min=time_horizon_min,
                        altitude_band=altitude_band,
                        model_id=model_id,
                        token=hf_token
                    )
                    response["advisory"] = advisory
                    response["model"] = "gemma-2b-it+lora"
                except Exception as e:
                    logger.error(f"Error generating advisory: {str(e)}")
                    response["advisory_error"] = f"Failed to generate advisory: {str(e)}"
        
        logger.info(f"Prediction: probability={probability:.4f}, severity={severity}, "
                   f"confidence={confidence}, score={anomaly_score:.6f}")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


def predict_with_hackathon_service():
    """Fallback prediction using hackathon ML service when LSTM model is unavailable."""
    try:
        data = request.get_json() or {}
        
        # Check if we have window data (from /predict endpoint) or lat/lon (from /predict-cat)
        if 'window' in data:
            # Extract features from window data
            window = data.get('window', [])
            if not window or len(window) == 0:
                return jsonify({"error": "Window data is required"}), 400
            
            try:
                window_array = np.array(window, dtype=np.float32)
            except (ValueError, TypeError) as e:
                return jsonify({"error": f"Invalid window data: {str(e)}"}), 400
            
            if window_array.ndim != 2 or window_array.shape[1] != NUM_FEATURES:
                return jsonify({"error": f"Window must be 2D array with {NUM_FEATURES} features per row"}), 400
            
            # Extract features from window: [altitude, velocity, vertical_rate, u_wind, v_wind, temperature]
            # Calculate statistics from the window
            u_wind_values = window_array[:, 3]  # u_wind column
            v_wind_values = window_array[:, 4]  # v_wind column
            temp_values = window_array[:, 5]     # temperature column
            vertical_rate_values = window_array[:, 2]  # vertical_rate column
            
            # Calculate wind speed from u and v components
            wind_speeds = np.sqrt(u_wind_values**2 + v_wind_values**2)
            avg_wind_speed = float(np.mean(wind_speeds))
            wind_change = float(np.std(wind_speeds))  # Variability in wind speed
            
            # Calculate temperature gradient (variation)
            temp_gradient = float(np.std(temp_values))
            
            # Calculate pressure drop (using vertical rate as proxy for pressure changes)
            pressure_drop = float(np.std(vertical_rate_values))  # High vertical rate variance = pressure changes
            
            # Create features dict for ML model
            features = {
                'wind_speed': avg_wind_speed,
                'wind_change': wind_change,
                'temp_gradient': temp_gradient,
                'pressure_drop': pressure_drop
            }
            
            # Create mock weather data for Gemma service
            weather_data = {
                'wind_speed': avg_wind_speed,
                'temperature': float(np.mean(temp_values)),
                'pressure': 1013.25,  # Standard sea level pressure
                'humidity': 50.0  # Default
            }
            
        elif 'lat' in data or 'lon' in data:
            # Original behavior: use lat/lon to get weather data
            lat = data.get('lat', 40.7128)
            lon = data.get('lon', -74.0060)
            
            # 1. Get weather data
            weather_data = weather_service.get_weather_data(lat, lon)
            
            # 2. Extract features
            features = feature_engineer.extract_features(weather_data)
        else:
            return jsonify({"error": "Either 'window' or 'lat'/'lon' must be provided"}), 400
        
        # 3. ML prediction
        ml_probability = ml_model.predict(features)
        
        # 4. Gemma reasoning (if available)
        try:
            gemma_result = gemma_service.analyze_turbulence(weather_data, ml_probability)
            final_probability = gemma_result['adjusted_probability']
            gemma_adjustment = gemma_result['raw_adjustment']
            reasoning = gemma_result['reasoning']
        except Exception as gemma_error:
            logger.warning(f"Gemma service unavailable: {gemma_error}. Using ML prediction only.")
            final_probability = ml_probability
            gemma_adjustment = 0
            reasoning = "ML model prediction only (Gemma service unavailable)"
        
        # 5. Final probability
        turbulence_percentage = round(final_probability * 100)
        
        # Determine severity based on probability
        if final_probability < 0.3:
            severity = "Low"
        elif final_probability < 0.7:
            severity = "Moderate"
        else:
            severity = "High"
        
        # Determine confidence based on feature variance
        feature_variance = wind_change + temp_gradient + pressure_drop
        if feature_variance > 3.0:
            confidence = "High"
        elif feature_variance > 1.5:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        # Generate advisory based on severity
        advisory = generate_advisory_from_prediction(final_probability, severity, confidence, 
                                                     data.get('time_horizon_min', 15),
                                                     data.get('altitude_band', 'FL380'))
        
        return jsonify({
            "turbulence_probability": float(final_probability),  # 0-1 range
            "severity": severity,
            "confidence": confidence,
            "anomaly_score": float(feature_variance / 10.0),  # Normalized feature variance
            "advisory": advisory,
            "model_type": "hackathon_ml",
            "ml_score": round(ml_probability * 100),
            "gemma_adjustment": gemma_adjustment,
            "reasoning": reasoning
        })
        
    except Exception as e:
        logger.error(f"Hackathon service prediction error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Hackathon service error: {str(e)}"}), 500


def generate_advisory_from_prediction(probability, severity, confidence, time_horizon_min, altitude_band):
    """Generate advisory text from prediction data."""
    prob_percent = round(probability * 100)
    
    if severity == "High":
        return f"SEVERE CAT WARNING: {prob_percent}% turbulence probability at {altitude_band}. Immediate altitude change recommended. High confidence. Expected within {time_horizon_min} minutes."
    elif severity == "Moderate":
        return f"Moderate turbulence likely ahead in ~{time_horizon_min} minutes at {altitude_band}. Probability {prob_percent}%. Seatbelt ON. {confidence} confidence."
    else:
        return f"Conditions stable. Turbulence probability {prob_percent}% at {altitude_band}. Low risk expected for next {time_horizon_min} minutes. {confidence} confidence."


@app.route('/advisory', methods=['POST'])
def advisory():
    """Generate turbulence advisory from prediction inputs."""
    try:
        if not LLM_AVAILABLE:
            return jsonify({
                "error": "LLM module not available. Install transformers, peft, and bitsandbytes."
            }), 503
        
        # Validate request
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.get_json()
        
        # Validate required fields
        required_fields = ["probability", "severity", "confidence", "time_horizon_min", "altitude_band"]
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                "error": f"Missing required fields: {', '.join(missing_fields)}"
            }), 400
        
        # Validate field types and values
        probability = data["probability"]
        if not isinstance(probability, (int, float)) or not (0 <= probability <= 1):
            return jsonify({"error": "probability must be a float between 0 and 1"}), 400
        
        severity = data["severity"]
        if severity not in ["Low", "Moderate", "High"]:
            return jsonify({"error": "severity must be 'Low', 'Moderate', or 'High'"}), 400
        
        confidence = data["confidence"]
        if confidence not in ["Low", "Medium", "High"]:
            return jsonify({"error": "confidence must be 'Low', 'Medium', or 'High'"}), 400
        
        time_horizon_min = data["time_horizon_min"]
        if not isinstance(time_horizon_min, int) or time_horizon_min < 0:
            return jsonify({"error": "time_horizon_min must be a non-negative integer"}), 400
        
        altitude_band = data["altitude_band"]
        if not isinstance(altitude_band, str):
            return jsonify({"error": "altitude_band must be a string"}), 400
        
        # Optional model config
        model_id = data.get("model_id")
        hf_token = data.get("hf_token")

        # Generate advisory
        advisory_text = generate_advisory(
            probability=float(probability),
            severity=severity,
            confidence=confidence,
            time_horizon_min=int(time_horizon_min),
            altitude_band=altitude_band,
            model_id=model_id,
            token=hf_token
        )
        
        response = {
            "advisory": advisory_text,
            "model": "gemma-2b-it+lora"
        }
        
        logger.info(f"Generated advisory for probability={probability:.2f}, severity={severity}")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Advisory generation error: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route('/train', methods=['POST'])
def train():
    """Trigger training pipeline."""
    try:
        import subprocess
        import sys
        from pathlib import Path
        
        logger.info("Starting training pipeline...")
        
        # Run preprocessing
        logger.info("Running preprocessing...")
        preprocess_script = Path(__file__).parent / "scripts" / "preprocess.py"
        result = subprocess.run(
            [sys.executable, str(preprocess_script)],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            logger.error(f"Preprocessing failed: {result.stderr}")
            return jsonify({"error": f"Preprocessing failed: {result.stderr}"}), 500
        
        # Run training
        logger.info("Running training...")
        train_script = Path(__file__).parent / "scripts" / "train.py"
        result = subprocess.run(
            [sys.executable, str(train_script)],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            logger.error(f"Training failed: {result.stderr}")
            return jsonify({"error": f"Training failed: {result.stderr}"}), 500
        
        # Extract loss from output
        output_lines = result.stdout.split('\n')
        loss = None
        num_windows = None
        for line in output_lines:
            if "Final loss:" in line:
                try:
                    loss = float(line.split("Final loss:")[1].split(",")[0].strip())
                except:
                    pass
            if "Windows:" in line:
                try:
                    num_windows = int(line.split("Windows:")[1].strip())
                except:
                    pass
        
        # Run score computation
        logger.info("Computing training scores...")
        scores_script = Path(__file__).parent / "scripts" / "compute_training_scores.py"
        result = subprocess.run(
            [sys.executable, str(scores_script)],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            logger.error(f"Score computation failed: {result.stderr}")
            return jsonify({"error": f"Score computation failed: {result.stderr}"}), 500
        
        # Reload model, scores, and normalization stats
        global model, training_scores, normalization_stats
        model = None
        training_scores = None
        normalization_stats = None
        load_model()
        load_training_scores()
        load_norm_stats()
        
        logger.info("Training pipeline completed successfully")
        
        return jsonify({
            "status": "trained",
            "num_windows": num_windows or 0,
            "loss": loss or 0.0
        }), 200
        
    except Exception as e:
        logger.error(f"Training pipeline error: {str(e)}")
        return jsonify({"error": f"Training failed: {str(e)}"}), 500


# ============================================================================
# HACKATHON ENDPOINTS - Simple CAT Prediction System
# ============================================================================

@app.route('/weather', methods=['GET'])
def get_weather():
    """Fetch current weather data"""
    if not HACKATHON_SERVICES_AVAILABLE:
        return jsonify({"error": "Hackathon services not available"}), 503
    
    try:
        lat = request.args.get('lat', 40.7128, type=float)
        lon = request.args.get('lon', -74.0060, type=float)
        
        weather_data = weather_service.get_weather_data(lat, lon)
        return jsonify(weather_data)
    except Exception as e:
        logger.error(f"Weather endpoint error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/predict-cat', methods=['POST'])
def predict_turbulence():
    """
    Predict turbulence risk (Hackathon version)
    Input: JSON with optional lat/lon and flight_altitude
    Returns: turbulence probability and risk level
    """
    if not HACKATHON_SERVICES_AVAILABLE:
        return jsonify({"error": "Hackathon services not available"}), 503
    
    try:
        data = request.get_json() or {}
        lat = data.get('lat', 40.7128)
        lon = data.get('lon', -74.0060)
        
        # 1. Get weather data
        weather_data = weather_service.get_weather_data(lat, lon)
        
        # 2. Extract features
        features = feature_engineer.extract_features(weather_data)
        
        # 3. ML prediction
        ml_probability = ml_model.predict(features)
        
        # 4. Gemma reasoning
        gemma_result = gemma_service.analyze_turbulence(weather_data, ml_probability)
        
        # 5. Final probability
        final_probability = gemma_result['adjusted_probability']
        turbulence_percentage = round(final_probability * 100)
        
        # Determine risk level
        if turbulence_percentage < 40:
            risk_level = "Low"
        elif turbulence_percentage < 70:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return jsonify({
            "turbulence_probability": turbulence_percentage,
            "risk_level": risk_level,
            "weather": weather_data,
            "features": features,
            "ml_score": round(ml_probability * 100),
            "gemma_adjustment": gemma_result['raw_adjustment'],
            "reasoning": gemma_result['reasoning']
        })
        
    except Exception as e:
        logger.error(f"CAT prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    logger.info(f"Starting Flask server on {HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=False)

