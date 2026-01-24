"""Build synthetic training dataset for Gemma fine-tuning."""
import json
import random
from pathlib import Path
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


def generate_advisory_text(probability: float, severity: str, confidence: str,
                           time_horizon_min: int, altitude_band: str) -> str:
    """
    Generate synthetic advisory text based on inputs.
    
    Args:
        probability: Turbulence probability (0-1)
        severity: "Low", "Moderate", or "High"
        confidence: "Low", "Medium", or "High"
        time_horizon_min: Time horizon in minutes
        altitude_band: Altitude band (e.g., "FL360")
        
    Returns:
        Advisory text string
    """
    # Base templates for different severity levels
    if severity == "Low":
        templates = [
            f"Light turbulence possible at {altitude_band} within {time_horizon_min} minutes. "
            f"Probability {probability:.0%}. Monitor conditions.",
            f"Minimal turbulence expected at {altitude_band} in next {time_horizon_min} minutes. "
            f"Probability {probability:.0%}. Continue normal operations.",
            f"Low-level turbulence advisory for {altitude_band}. "
            f"Probability {probability:.0%} over {time_horizon_min} minutes. Standard precautions.",
        ]
    elif severity == "Moderate":
        templates = [
            f"Moderate turbulence expected at {altitude_band} within {time_horizon_min} minutes. "
            f"Probability {probability:.0%}. Consider altitude change or route adjustment.",
            f"Moderate turbulence advisory for {altitude_band}. "
            f"Probability {probability:.0%} over {time_horizon_min} minutes. Secure cabin and prepare.",
            f"Moderate turbulence conditions at {altitude_band} in {time_horizon_min} minutes. "
            f"Probability {probability:.0%}. Exercise caution.",
        ]
    else:  # High
        templates = [
            f"High turbulence expected at {altitude_band} within {time_horizon_min} minutes. "
            f"Probability {probability:.0%}. Strongly consider altitude change or route deviation.",
            f"Severe turbulence advisory for {altitude_band}. "
            f"Probability {probability:.0%} over {time_horizon_min} minutes. Immediate action recommended.",
            f"High turbulence conditions at {altitude_band} in {time_horizon_min} minutes. "
            f"Probability {probability:.0%}. Consider alternative routing.",
        ]
    
    # Add confidence qualifiers
    if confidence == "Low":
        qualifiers = ["Uncertain conditions.", "Limited confidence.", "Variable conditions."]
    elif confidence == "Medium":
        qualifiers = ["Moderate confidence.", "Fair conditions.", "Standard confidence."]
    else:  # High
        qualifiers = ["High confidence.", "Reliable forecast.", "Strong confidence."]
    
    # Select random template and qualifier
    base_text = random.choice(templates)
    qualifier = random.choice(qualifiers)
    
    # Combine (keep max 2 lines)
    advisory = f"{base_text} {qualifier}"
    
    # Ensure it's not too long (max ~200 chars for 2 lines)
    if len(advisory) > 200:
        advisory = advisory[:197] + "..."
    
    return advisory


def generate_training_examples(num_examples: int = 300) -> List[Dict]:
    """
    Generate synthetic training examples.
    
    Args:
        num_examples: Number of examples to generate
        
    Returns:
        List of training example dictionaries
    """
    examples = []
    
    severities = ["Low", "Moderate", "High"]
    confidences = ["Low", "Medium", "High"]
    altitude_bands = ["FL200", "FL250", "FL300", "FL330", "FL360", "FL390", "FL410", "FL450"]
    time_horizons = [5, 8, 10, 12, 15, 20, 30, 45, 60]
    
    random.seed(42)  # For reproducibility
    
    for i in range(num_examples):
        # Generate realistic probability based on severity
        if random.random() < 0.33:
            severity = "Low"
            probability = random.uniform(0.15, 0.40)
        elif random.random() < 0.66:
            severity = "Moderate"
            probability = random.uniform(0.40, 0.70)
        else:
            severity = "High"
            probability = random.uniform(0.70, 0.95)
        
        # Select other parameters
        confidence = random.choice(confidences)
        altitude_band = random.choice(altitude_bands)
        time_horizon_min = random.choice(time_horizons)
        
        # Generate advisory text
        advisory_text = generate_advisory_text(
            probability, severity, confidence, time_horizon_min, altitude_band
        )
        
        example = {
            "input": {
                "probability": round(probability, 2),
                "severity": severity,
                "confidence": confidence,
                "time_horizon_min": time_horizon_min,
                "altitude_band": altitude_band
            },
            "output": advisory_text
        }
        
        examples.append(example)
    
    return examples


def build_dataset(output_path: Path, num_examples: int = 300):
    """
    Build and save training dataset.
    
    Args:
        output_path: Path to save JSONL file
        num_examples: Number of examples to generate
    """
    logger.info(f"Generating {num_examples} training examples...")
    examples = generate_training_examples(num_examples)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write JSONL file
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')
    
    logger.info(f"Dataset saved to {output_path} with {len(examples)} examples")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    logging.basicConfig(level=logging.INFO)
    
    # Default output path
    base_dir = Path(__file__).parent.parent
    output_path = base_dir / "data" / "llm_train.jsonl"
    
    # Allow override via command line
    if len(sys.argv) > 1:
        output_path = Path(sys.argv[1])
    
    build_dataset(output_path, num_examples=300)
    print(f"Dataset built: {output_path}")

