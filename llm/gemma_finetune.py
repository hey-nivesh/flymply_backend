"""Fine-tune Gemma model using LoRA for turbulence advisory generation."""
import os
import json
import logging
from pathlib import Path
from typing import Optional
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import jsonlines

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(jsonl_path: Path) -> Dataset:
    """
    Load training dataset from JSONL file.
    
    Args:
        jsonl_path: Path to JSONL file
        
    Returns:
        HuggingFace Dataset
    """
    examples = []
    with jsonlines.open(jsonl_path) as reader:
        for obj in reader:
            # Format as instruction-following prompt
            input_data = obj["input"]
            output_text = obj["output"]
            
            # Format prompt using Gemma chat template
            prompt = (
                f"<start_of_turn>user\n"
                f"Generate a brief aviation turbulence advisory. "
                f"Probability: {input_data['probability']:.2f}, Severity: {input_data['severity']}, "
                f"Confidence: {input_data['confidence']}, Time horizon: {input_data['time_horizon_min']} minutes, "
                f"Altitude: {input_data['altitude_band']}. "
                f"Keep it professional, concise (max 2 lines), and cockpit-safe.<end_of_turn>\n"
                f"<start_of_turn>model\n{output_text}<end_of_turn>"
            )
            
            examples.append({"text": prompt})
    
    return Dataset.from_list(examples)


def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize examples for training."""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )


def fine_tune_gemma(
    model_id: str = "google/gemma-2b-it",
    dataset_path: Path = None,
    output_dir: Path = None,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    use_4bit: bool = True,
    max_length: int = 512
):
    """
    Fine-tune Gemma model using LoRA.
    
    Args:
        model_id: HuggingFace model ID
        dataset_path: Path to training JSONL file
        output_dir: Directory to save LoRA adapters
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        use_4bit: Whether to use 4-bit quantization
        max_length: Maximum sequence length
    """
    # Set default paths
    if dataset_path is None:
        base_dir = Path(__file__).parent.parent
        dataset_path = base_dir / "data" / "llm_train.jsonl"
    
    if output_dir is None:
        base_dir = Path(__file__).parent.parent
        output_dir = base_dir / "llm" / "adapters" / "gemma_turbulence_advisor"
    
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if dataset exists
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            f"Run llm/dataset_builder.py first to generate training data."
        )
    
    logger.info(f"Loading dataset from {dataset_path}")
    dataset = load_dataset(dataset_path)
    logger.info(f"Loaded {len(dataset)} examples")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer for {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with quantization if requested
    logger.info(f"Loading model {model_id}")
    device_map = "auto"
    
    if use_4bit and torch.cuda.is_available():
        logger.info("Using 4-bit quantization (QLoRA)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=device_map,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        model = prepare_model_for_kbit_training(model)
    else:
        logger.info("Using full precision (CPU or no quantization support)")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Tokenize dataset
    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        warmup_steps=50,
        report_to="none",
        remove_unused_columns=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save model
    logger.info(f"Saving LoRA adapters to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune Gemma for turbulence advisories")
    parser.add_argument("--model-id", type=str, default="google/gemma-2b-it",
                       help="HuggingFace model ID")
    parser.add_argument("--dataset", type=str, default=None,
                       help="Path to training JSONL file")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for LoRA adapters")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--no-4bit", action="store_true",
                       help="Disable 4-bit quantization")
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset) if args.dataset else None
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    fine_tune_gemma(
        model_id=args.model_id,
        dataset_path=dataset_path,
        output_dir=output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_4bit=not args.no_4bit
    )

