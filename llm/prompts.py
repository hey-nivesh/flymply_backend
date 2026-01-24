"""Prompt templates for Gemma turbulence advisory generation."""

def format_training_prompt(probability: float, severity: str, confidence: str, 
                          time_horizon_min: int, altitude_band: str) -> str:
    """
    Format input prompt for training examples.
    
    Args:
        probability: Turbulence probability (0-1)
        severity: "Low", "Moderate", or "High"
        confidence: "Low", "Medium", or "High"
        time_horizon_min: Time horizon in minutes
        altitude_band: Altitude band (e.g., "FL360")
        
    Returns:
        Formatted prompt string
    """
    return (
        f"<start_of_turn>user\n"
        f"Generate a brief aviation turbulence advisory. "
        f"Probability: {probability:.2f}, Severity: {severity}, "
        f"Confidence: {confidence}, Time horizon: {time_horizon_min} minutes, "
        f"Altitude: {altitude_band}. "
        f"Keep it professional, concise (max 2 lines), and cockpit-safe.<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )


def format_inference_prompt(probability: float, severity: str, confidence: str,
                            time_horizon_min: int, altitude_band: str) -> str:
    """
    Format input prompt for inference.
    
    Args:
        probability: Turbulence probability (0-1)
        severity: "Low", "Moderate", or "High"
        confidence: "Low", "Medium", or "High"
        time_horizon_min: Time horizon in minutes
        altitude_band: Altitude band (e.g., "FL360")
        
    Returns:
        Formatted prompt string
    """
    return (
        f"<start_of_turn>user\n"
        f"Generate a brief aviation turbulence advisory. "
        f"Probability: {probability:.2f}, Severity: {severity}, "
        f"Confidence: {confidence}, Time horizon: {time_horizon_min} minutes, "
        f"Altitude: {altitude_band}. "
        f"Keep it professional, concise (max 2 lines), and cockpit-safe.<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )

