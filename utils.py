"""Utility functions for data processing and model operations."""
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import logging
import tarfile
import io
import json
import gzip
from typing import List, Tuple, Optional

from config import RAW_DATA_DIR, DATA_FOR_MODEL_DIR, PROCESSED_DATA_DIR, FEATURE_COLUMNS, WINDOW_SIZE

logger = logging.getLogger(__name__)


def map_flight_data_to_features(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Map flight tracking data columns to expected feature columns.
    
    Handles ADS-B flight tracking data format:
    - Maps baroaltitude/geoaltitude -> altitude
    - Maps velocity -> velocity
    - Maps vertrate -> vertical_rate
    - Sets default/derived values for missing weather data (u_wind, v_wind, temperature)
    
    Args:
        df: DataFrame with flight tracking data
        
    Returns:
        DataFrame with mapped features or None if error
    """
    try:
        # Create a copy to avoid modifying original
        mapped_df = df.copy()
        
        # Map altitude (prefer baroaltitude, fallback to geoaltitude)
        if 'baroaltitude' in mapped_df.columns:
            mapped_df['altitude'] = mapped_df['baroaltitude']
        elif 'geoaltitude' in mapped_df.columns:
            mapped_df['altitude'] = mapped_df['geoaltitude']
        elif 'altitude' not in mapped_df.columns:
            logger.warning("No altitude column found (baroaltitude, geoaltitude, or altitude)")
            return None
        
        # Map velocity
        if 'velocity' not in mapped_df.columns:
            logger.warning("No velocity column found")
            return None
        
        # Map vertical_rate
        if 'vertrate' in mapped_df.columns:
            mapped_df['vertical_rate'] = mapped_df['vertrate']
        elif 'vertical_rate' not in mapped_df.columns:
            # Set to 0 if not available
            mapped_df['vertical_rate'] = 0.0
            logger.debug("No vertrate/vertical_rate found, using 0.0")
        
        # Handle missing weather data - use default/derived values
        # u_wind and v_wind: Can derive from velocity and heading if available
        if 'u_wind' not in mapped_df.columns:
            if 'heading' in mapped_df.columns and 'velocity' in mapped_df.columns:
                # Derive wind components from heading (simplified - assumes no wind for now)
                # In reality, this would need actual weather data
                mapped_df['u_wind'] = 0.0
                mapped_df['v_wind'] = 0.0
            else:
                mapped_df['u_wind'] = 0.0
                mapped_df['v_wind'] = 0.0
            logger.debug("No u_wind/v_wind found, using 0.0")
        
        # Temperature: Use standard atmospheric model based on altitude
        if 'temperature' not in mapped_df.columns:
            # Standard atmospheric temperature model: T = 288.15 - 0.0065 * altitude (in meters)
            # Convert altitude from feet to meters if needed (assuming altitude is in feet)
            altitude_m = mapped_df['altitude'] * 0.3048  # feet to meters
            mapped_df['temperature'] = 288.15 - 0.0065 * altitude_m  # Kelvin
            # Convert to Celsius for more intuitive values (or keep Kelvin)
            mapped_df['temperature'] = mapped_df['temperature'] - 273.15
            logger.debug("No temperature found, using standard atmospheric model")
        
        # Select only the required feature columns
        if not all(col in mapped_df.columns for col in FEATURE_COLUMNS):
            missing = set(FEATURE_COLUMNS) - set(mapped_df.columns)
            logger.error(f"Missing required columns after mapping: {missing}")
            return None
        
        return mapped_df[FEATURE_COLUMNS]
        
    except Exception as e:
        logger.error(f"Error mapping flight data to features: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def create_sliding_windows(data: np.ndarray, window_size: int, stride: int = 1) -> np.ndarray:
    """
    Create sliding windows from time-series data.
    
    Args:
        data: Array of shape (N, features)
        window_size: Size of each window
        stride: Step size for sliding window
        
    Returns:
        Array of shape (num_windows, window_size, features)
    """
    if len(data) < window_size:
        logger.warning(f"Data length ({len(data)}) is less than window_size ({window_size})")
        return np.array([]).reshape(0, window_size, data.shape[1])
    
    windows = []
    for i in range(0, len(data) - window_size + 1, stride):
        windows.append(data[i:i + window_size])
    
    return np.array(windows)


def extract_and_read_tar_csv(tar_path: Path) -> Optional[pd.DataFrame]:
    """
    Extract and read CSV from a .csv.tar file.
    Handles both plain CSV and gzipped CSV (.csv.gz) files.
    
    Args:
        tar_path: Path to .csv.tar file
        
    Returns:
        DataFrame or None if error
    """
    try:
        with tarfile.open(tar_path, 'r') as tar:
            # Find CSV file inside tar (check for both .csv and .csv.gz)
            csv_members = [m for m in tar.getmembers() 
                          if m.name.endswith('.csv') or m.name.endswith('.csv.gz')]
            if not csv_members:
                logger.warning(f"No CSV file found in {tar_path.name}")
                return None
            
            # Read the first CSV file found
            csv_member = csv_members[0]
            csv_file = tar.extractfile(csv_member)
            if csv_file is None:
                logger.error(f"Could not extract {csv_member.name} from {tar_path.name}")
                return None
            
            # Check if it's gzipped
            if csv_member.name.endswith('.gz'):
                # Decompress gzip
                with gzip.open(csv_file, 'rt', encoding='utf-8') as gz_file:
                    df = pd.read_csv(gz_file)
            else:
                # Plain CSV
                df = pd.read_csv(io.TextIOWrapper(csv_file))
            
            logger.info(f"Extracted and loaded CSV from {tar_path.name} ({csv_member.name})")
            return df
            
    except Exception as e:
        logger.error(f"Error extracting CSV from {tar_path}: {str(e)}")
        return None


def extract_and_read_tar_json(tar_path: Path) -> Optional[pd.DataFrame]:
    """
    Extract and read JSON from a .json.tar file and convert to DataFrame.
    Handles both plain JSON and gzipped JSON (.json.gz) files.
    
    Args:
        tar_path: Path to .json.tar file
        
    Returns:
        DataFrame or None if error
    """
    try:
        with tarfile.open(tar_path, 'r') as tar:
            # Find JSON file inside tar (check for both .json and .json.gz)
            json_members = [m for m in tar.getmembers() 
                           if m.name.endswith('.json') or m.name.endswith('.json.gz')]
            if not json_members:
                logger.warning(f"No JSON file found in {tar_path.name}")
                return None
            
            # Read the first JSON file found
            json_member = json_members[0]
            json_file = tar.extractfile(json_member)
            if json_file is None:
                logger.error(f"Could not extract {json_member.name} from {tar_path.name}")
                return None
            
            # Check if it's gzipped
            if json_member.name.endswith('.gz'):
                # Decompress gzip
                with gzip.open(json_file, 'rt', encoding='utf-8') as gz_file:
                    json_data = json.load(gz_file)
            else:
                # Plain JSON
                json_data = json.load(io.TextIOWrapper(json_file))
            
            # Convert JSON to DataFrame
            # Handle different JSON structures
            if isinstance(json_data, list):
                # If it's a list of records
                if len(json_data) > 0 and isinstance(json_data[0], dict):
                    df = pd.json_normalize(json_data)
                else:
                    df = pd.DataFrame(json_data)
            elif isinstance(json_data, dict):
                # Try to find data array
                if 'data' in json_data:
                    if isinstance(json_data['data'], list):
                        df = pd.json_normalize(json_data['data'])
                    else:
                        df = pd.DataFrame([json_data['data']])
                elif 'states' in json_data:
                    if isinstance(json_data['states'], list):
                        df = pd.json_normalize(json_data['states'])
                    else:
                        df = pd.DataFrame([json_data['states']])
                elif 'time' in json_data or 'altitude' in json_data:
                    # Single record dict
                    df = pd.json_normalize([json_data])
                else:
                    # Try to flatten the dict
                    df = pd.json_normalize([json_data])
            else:
                logger.error(f"Unexpected JSON structure in {tar_path.name}: {type(json_data)}")
                return None
            
            logger.info(f"Extracted and loaded JSON from {tar_path.name} ({json_member.name})")
            return df
            
    except Exception as e:
        logger.error(f"Error extracting JSON from {tar_path}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def load_and_preprocess_csv(file_path: Path) -> Optional[np.ndarray]:
    """
    Load and preprocess a single CSV file or extract from tar.
    
    Args:
        file_path: Path to CSV file or .csv.tar file
        
    Returns:
        Preprocessed numpy array of shape (N, 6) or None if error
    """
    try:
        # Handle tar files
        if file_path.suffixes == ['.csv', '.tar'] or str(file_path).endswith('.csv.tar'):
            df = extract_and_read_tar_csv(file_path)
        else:
            df = pd.read_csv(file_path)
        
        if df is None:
            return None
        
        # Map flight tracking data to expected features
        mapped_df = map_flight_data_to_features(df)
        if mapped_df is None:
            logger.error(f"Could not map columns in {file_path}")
            logger.debug(f"Available columns: {list(df.columns)}")
            return None
        
        # Extract features in correct order
        features = mapped_df.values
        
        # Remove rows with NaN or infinite values
        valid_mask = np.isfinite(features).all(axis=1)
        features = features[valid_mask]
        
        if len(features) == 0:
            logger.warning(f"No valid data in {file_path}")
            return None
        
        # Normalize features (z-score normalization)
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True)
        std = np.where(std == 0, 1, std)  # Avoid division by zero
        features = (features - mean) / std
        
        logger.info(f"Loaded {len(features)} samples from {file_path.name}")
        return features
        
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}")
        return None


def load_all_training_data() -> Tuple[np.ndarray, dict]:
    """
    Load all CSV/JSON files from data/raw/ or data_for_model/ and create windows.
    Supports .csv, .csv.tar, and .json.tar files.
    
    Returns:
        Tuple of (windows array, normalization stats dict)
    """
    # Check both directories
    raw_dir = Path(RAW_DATA_DIR)
    data_for_model_dir = Path(DATA_FOR_MODEL_DIR) if DATA_FOR_MODEL_DIR.exists() else None
    
    # Collect all data files
    data_files = []
    
    # Regular CSV files from data/raw/
    if raw_dir.exists():
        data_files.extend(raw_dir.glob("*.csv"))
        # CSV tar files from data/raw/
        data_files.extend(raw_dir.glob("*.csv.tar"))
        # JSON tar files from data/raw/
        data_files.extend(raw_dir.glob("*.json.tar"))
    
    # Tar files from data_for_model/ (if it exists)
    if data_for_model_dir and data_for_model_dir.exists():
        # CSV tar files
        data_files.extend(data_for_model_dir.glob("*.csv.tar"))
        # JSON tar files
        data_files.extend(data_for_model_dir.glob("*.json.tar"))
    
    if not data_files:
        error_msg = f"No data files found in {RAW_DATA_DIR}"
        if data_for_model_dir and data_for_model_dir.exists():
            error_msg += f" or {DATA_FOR_MODEL_DIR}"
        raise ValueError(error_msg)
    
    logger.info(f"Found {len(data_files)} data files")
    
    all_features = []
    all_stats = []
    
    for data_file in data_files:
        df = None
        
        # Handle different file types
        if str(data_file).endswith('.csv.tar'):
            df = extract_and_read_tar_csv(data_file)
        elif str(data_file).endswith('.json.tar'):
            df = extract_and_read_tar_json(data_file)
        elif data_file.suffix == '.csv':
            try:
                df = pd.read_csv(data_file)
            except Exception as e:
                logger.error(f"Error reading {data_file}: {str(e)}")
                continue
        
        if df is None:
            continue
        
        # Map flight tracking data to expected features
        # This handles ADS-B data format and maps columns appropriately
        mapped_df = map_flight_data_to_features(df)
        if mapped_df is None:
            logger.warning(f"Could not map columns for {data_file.name}")
            logger.debug(f"Available columns: {list(df.columns)}")
            continue
        
        # Extract features
        features = mapped_df.values
        
        # Remove invalid rows
        valid_mask = np.isfinite(features).all(axis=1)
        features = features[valid_mask]
        
        if len(features) == 0:
            logger.warning(f"No valid data in {data_file.name}")
            continue
        
        # Store stats for normalization
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        std = np.where(std == 0, 1, std)
        all_stats.append({"mean": mean, "std": std})
        
        # Normalize
        features = (features - mean) / std
        all_features.append(features)
        
        logger.info(f"Processed {len(features)} samples from {data_file.name}")
    
    if not all_features:
        raise ValueError("No valid data found in any data files")
    
    # Concatenate all data
    combined_data = np.vstack(all_features)
    
    # Compute global normalization stats (for inference)
    global_mean = np.mean([s["mean"] for s in all_stats], axis=0)
    global_std = np.mean([s["std"] for s in all_stats], axis=0)
    global_std = np.where(global_std == 0, 1, global_std)
    
    normalization_stats = {
        "mean": global_mean,
        "std": global_std
    }
    
    # Create sliding windows
    windows = create_sliding_windows(combined_data, WINDOW_SIZE, stride=1)
    
    logger.info(f"Created {len(windows)} windows of size {WINDOW_SIZE}")
    
    return windows, normalization_stats


def save_normalization_stats(stats: dict, file_path: Path):
    """Save normalization statistics to file."""
    np.savez(file_path, **stats)


def load_normalization_stats(file_path: Path) -> dict:
    """Load normalization statistics from file."""
    data = np.load(file_path, allow_pickle=True)
    return {key: data[key] for key in data.keys()}


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


def compute_probability(score: float, training_scores: np.ndarray) -> float:
    """
    Compute turbulence probability from anomaly score.
    
    Args:
        score: Current anomaly score
        training_scores: Array of training anomaly scores
        
    Returns:
        Probability as float between 0 and 1
    """
    if len(training_scores) == 0:
        return 0.5  # Default if no training scores
    
    # Fraction of training scores that are less than current score
    probability = np.mean(training_scores < score)
    return float(probability)


def get_severity(probability: float) -> str:
    """
    Map probability to severity level.
    
    Args:
        probability: Turbulence probability (0-1)
        
    Returns:
        Severity string: "Low", "Moderate", or "High"
    """
    if probability < 0.40:
        return "Low"
    elif probability < 0.70:
        return "Moderate"
    else:
        return "High"


def get_confidence(score: float, training_scores: np.ndarray) -> str:
    """
    Map score to confidence level based on training distribution.
    
    Args:
        score: Current anomaly score
        training_scores: Array of training anomaly scores
        
    Returns:
        Confidence string: "Low", "Medium", or "High"
    """
    if len(training_scores) == 0:
        return "Medium"
    
    percentile_50 = np.percentile(training_scores, 50)
    percentile_90 = np.percentile(training_scores, 90)
    
    if score <= percentile_50:
        return "High"
    elif score <= percentile_90:
        return "Medium"
    else:
        return "Low"

