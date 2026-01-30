import os
import torch
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Model-related configuration"""
    base_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    model_sft_path: str = "./llama3.2-3b-ultramedical-lora"
    model_b_path: str = "./llama3.2-3b-ultramedical-model-b"
    
    # CPU-specific settings
    use_cpu: bool = True  # Force CPU usage
    load_in_8bit: bool = True  # Use 8-bit instead of 4-bit for CPU
    device_map: str = "cpu"  # Force CPU device map
    torch_dtype: str = "float32"  # CPU works better with float32


@dataclass
class GenerationConfig:
    """Text generation parameters - Optimized for CPU"""
    max_new_tokens: int = 256  # Reduced for faster CPU generation
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    num_beams: int = 1  # Beam search is slow on CPU


@dataclass
class TrainingConfig:
    """Incremental training configuration - CPU Optimized"""
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8  # Increased for CPU
    gradient_checkpointing: bool = True
    optim: str = "adamw_torch"  # Standard optimizer for CPU
    learning_rate: float = 1e-4
    lr_scheduler_type: str = "constant"
    warmup_ratio: float = 0.0
    logging_steps: int = 5
    save_steps: int = 100
    save_total_limit: int = 2
    max_grad_norm: float = 1.0
    fp16: bool = False  # No mixed precision on CPU
    bf16: bool = False  # No bfloat16 on CPU
    report_to: str = "none"
    max_seq_length: int = 1024  # Reduced for CPU


@dataclass
class FeedbackConfig:
    """Feedback collection settings"""
    feedback_data_path: str = "./human_feedback_data.jsonl"
    retrain_threshold: int = 20  # Retrain after N samples
    auto_retrain: bool = True


@dataclass
class AppConfig:
    """Main application configuration"""
    model: ModelConfig = None
    generation: GenerationConfig = None
    training: TrainingConfig = None
    feedback: FeedbackConfig = None
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.generation is None:
            self.generation = GenerationConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.feedback is None:
            self.feedback = FeedbackConfig()
        
        # Detect if CUDA is available and warn if forcing CPU
        if self.model.use_cpu and torch.cuda.is_available():
            print("⚠️  CUDA detected but CPU mode is forced in config")
            print("   To use GPU, set use_cpu=False in config.py")
    
    def to_training_args_dict(self, output_dir: str) -> dict:
        """Convert training config to TrainingArguments dict"""
        return {
            "output_dir": output_dir,
            "num_train_epochs": self.training.num_train_epochs,
            "per_device_train_batch_size": self.training.per_device_train_batch_size,
            "gradient_accumulation_steps": self.training.gradient_accumulation_steps,
            "gradient_checkpointing": self.training.gradient_checkpointing,
            "optim": self.training.optim,
            "learning_rate": self.training.learning_rate,
            "lr_scheduler_type": self.training.lr_scheduler_type,
            "warmup_ratio": self.training.warmup_ratio,
            "logging_steps": self.training.logging_steps,
            "save_steps": self.training.save_steps,
            "save_total_limit": self.training.save_total_limit,
            "max_grad_norm": self.training.max_grad_norm,
            "fp16": self.training.fp16,
            "bf16": self.training.bf16,
            "report_to": self.training.report_to,
        }


# Default configuration instance
config = AppConfig()