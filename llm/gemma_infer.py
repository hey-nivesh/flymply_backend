"""Inference module for Gemma turbulence advisory generation."""
import os
import logging
from pathlib import Path
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GemmaAdvisor:
    """Gemma model wrapper for generating turbulence advisories."""
    
    def __init__(
        self,
        model_id: str = "google/gemma-2b-it",
        adapter_path: Optional[Path] = None,
        device: Optional[str] = None
    ):
        """
        Initialize Gemma advisor.
        
        Args:
            model_id: HuggingFace model ID
            adapter_path: Path to LoRA adapter directory
            device: Device to use ("cuda", "cpu", or None for auto)
        """
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self._loaded = False
    
    def _load_model(self):
        """Lazy load model and tokenizer."""
        if self._loaded:
            return
        
        logger.info(f"Loading model {self.model_id} on {self.device}")
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
        
        # Load base model
        try:
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
            else:
                logger.info("Using CPU inference")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                ).to(self.device)
        except Exception as e:
            logger.error(f"Failed to load base model: {e}")
            raise
        
        # Load LoRA adapter if provided
        if self.adapter_path and Path(self.adapter_path).exists():
            try:
                logger.info(f"Loading LoRA adapter from {self.adapter_path}")
                self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
                logger.info("LoRA adapter loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load LoRA adapter: {e}")
                logger.warning("Continuing with base model only")
        elif self.adapter_path:
            logger.warning(f"Adapter path {self.adapter_path} does not exist. Using base model.")
        
        self.model.eval()
        self._loaded = True
        logger.info("Model loaded successfully")
    
    def generate_advisory(
        self,
        probability: float,
        severity: str,
        confidence: str,
        time_horizon_min: int,
        altitude_band: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate turbulence advisory.
        
        Args:
            probability: Turbulence probability (0-1)
            severity: "Low", "Moderate", or "High"
            confidence: "Low", "Medium", or "High"
            time_horizon_min: Time horizon in minutes
            altitude_band: Altitude band (e.g., "FL360")
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated advisory text
        """
        if not self._loaded:
            self._load_model()
        
        # Format prompt
        prompt = (
            f"<start_of_turn>user\n"
            f"Generate a brief aviation turbulence advisory. "
            f"Probability: {probability:.2f}, Severity: {severity}, "
            f"Confidence: {confidence}, Time horizon: {time_horizon_min} minutes, "
            f"Altitude: {altitude_band}. "
            f"Keep it professional, concise (max 2 lines), and cockpit-safe.<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Extract only the model's response (after <start_of_turn>model\n)
        if "<start_of_turn>model\n" in full_text:
            response = full_text.split("<start_of_turn>model\n")[-1]
            # Remove <end_of_turn> if present
            response = response.split("<end_of_turn>")[0].strip()
        else:
            # Fallback: just take after the prompt
            response = full_text[len(prompt):].strip()
        
        # Clean up and ensure max 2 lines
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        if len(lines) > 2:
            response = '\n'.join(lines[:2])
        else:
            response = '\n'.join(lines)
        
        # Ensure safe tone (no panic words)
        panic_words = ["emergency", "immediate danger", "crash", "fatal", "death"]
        response_lower = response.lower()
        for word in panic_words:
            if word in response_lower:
                logger.warning(f"Detected potentially unsafe word: {word}. Sanitizing response.")
                # Replace with safer alternative
                response = response.replace(word, "caution")
        
        return response


# Global instance for lazy loading
_advisor_instance = None


def get_advisor(
    model_id: Optional[str] = None,
    adapter_path: Optional[Path] = None
) -> GemmaAdvisor:
    """
    Get or create global Gemma advisor instance.
    
    Args:
        model_id: HuggingFace model ID (uses env var if None)
        adapter_path: Path to LoRA adapter (uses env var if None)
        
    Returns:
        GemmaAdvisor instance
    """
    global _advisor_instance
    
    if _advisor_instance is None:
        if model_id is None:
            model_id = os.getenv("GEMMA_MODEL_ID", "google/gemma-2b-it")
        
        if adapter_path is None:
            adapter_str = os.getenv("LORA_ADAPTER_PATH", "llm/adapters/gemma_turbulence_advisor/")
            adapter_path = Path(adapter_str)
        
        _advisor_instance = GemmaAdvisor(
            model_id=model_id,
            adapter_path=adapter_path
        )
    
    return _advisor_instance


def generate_advisory(
    probability: float,
    severity: str,
    confidence: str,
    time_horizon_min: int,
    altitude_band: str
) -> str:
    """
    Generate turbulence advisory (convenience function).
    
    Args:
        probability: Turbulence probability (0-1)
        severity: "Low", "Moderate", or "High"
        confidence: "Low", "Medium", or "High"
        time_horizon_min: Time horizon in minutes
        altitude_band: Altitude band (e.g., "FL360")
        
    Returns:
        Generated advisory text
    """
    advisor = get_advisor()
    return advisor.generate_advisory(
        probability, severity, confidence, time_horizon_min, altitude_band
    )

