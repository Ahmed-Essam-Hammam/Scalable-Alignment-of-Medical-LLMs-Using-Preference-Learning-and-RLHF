import torch
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel
from typing import Optional
from config import AppConfig
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class ModelManager:
    """Manages model loading and text generation"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[PeftModel] = None
        self._is_loaded = False
    
    def load_model(self):
        """Load base model and LoRA adapter - CPU Optimized"""
        print("="*80)
        print("LOADING MODEL (CPU MODE)")
        print("="*80)
        
        # Get HuggingFace token from environment
        token = os.getenv("HF_TOKEN")
        
        if not token:
            print("❌ ERROR: HF_TOKEN not found!")
            print("💡 Please create a .env file with your HuggingFace token:")
            print("   1. Create a file named '.env' in the project root")
            print("   2. Add this line: HF_TOKEN=hf_your_token_here")
            print("   3. Get your token from: https://huggingface.co/settings/tokens")
            raise RuntimeError("HuggingFace token not found in .env file")
        
        print(f"✅ HuggingFace token loaded from .env")
        print(f"   Token starts with: {token[:10]}...")
        
        # Check device
        if torch.cuda.is_available() and not self.config.model.use_cpu:
            device = "cuda"
            print(f"✅ CUDA available - using GPU")
        else:
            device = "cpu"
            print(f"⚠️  Running on CPU - this will be slower")
            print(f"   Inference: ~30-60 seconds per response")
            print(f"   Training: ~10-30 minutes per iteration")
        
        # Load tokenizer
        print(f"\nLoading tokenizer from {self.config.model.base_model}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model.base_model,
                token=token,
                trust_remote_code=True
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
            print("✅ Tokenizer loaded successfully!")
        except Exception as e:
            print(f"\n❌ Failed to load tokenizer: {str(e)}")
            print("\n💡 Possible solutions:")
            print("   1. Check your token is valid: https://huggingface.co/settings/tokens")
            print("   2. Accept Llama 3.1 license: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct")
            print("   3. Ensure token has 'Read' permissions")
            print("   4. Check token in .env file is correct (starts with 'hf_')")
            raise
        
        # Load base model (CPU-optimized)
        print(f"\nLoading base model: {self.config.model.base_model}...")
        print("⏳ This may take 5-10 minutes on CPU...")
        print("   Model size: ~8GB, loading in 8-bit to save memory")
        
        try:
            # Load in 8-bit for CPU (more stable than 4-bit)
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.model.base_model,
                load_in_8bit=self.config.model.load_in_8bit,
                device_map=self.config.model.device_map,
                torch_dtype=getattr(torch, self.config.model.torch_dtype),
                trust_remote_code=True,
                token=token,
                low_cpu_mem_usage=True,  # Reduce memory spikes
            )
            print("✅ Base model loaded!")
        except Exception as e:
            print(f"\n❌ Failed to load base model: {str(e)}")
            print("\n💡 If you're running out of memory, try:")
            print("   1. Close other applications")
            print("   2. Use a smaller model (Llama 3.2 3B)")
            print("   3. Use Google Colab with free GPU")
            raise
        
        # Load LoRA adapter
        print(f"\nLoading LoRA adapter from {self.config.model.model_sft_path}...")
        try:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(
                base_model, 
                self.config.model.model_sft_path,
                device_map=self.config.model.device_map
            )
            self.model.eval()
            print("✅ LoRA adapter loaded!")
        except Exception as e:
            print(f"\n❌ Failed to load LoRA adapter: {str(e)}")
            print(f"\n💡 Make sure the path exists: {self.config.model.model_sft_path}")
            raise
        
        self._is_loaded = True
        print("\n" + "="*80)
        print("✅ MODEL LOADED SUCCESSFULLY!")
        print(f"   Device: {device.upper()}")
        print(f"   Memory efficient: 8-bit quantization")
        print("="*80)
    
    def generate_response(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """
        Generate response for a given prompt
        
        Args:
            prompt: User's input prompt
            max_new_tokens: Maximum tokens to generate (uses config default if None)
            temperature: Sampling temperature (uses config default if None)
            top_p: Nucleus sampling parameter (uses config default if None)
        
        Returns:
            Generated response text
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Use config defaults if not specified
        max_new_tokens = max_new_tokens or self.config.generation.max_new_tokens
        temperature = temperature or self.config.generation.temperature
        top_p = top_p or self.config.generation.top_p
        
        # Format prompt in Llama 3.1 chat format
        formatted_prompt = self._format_prompt(prompt)
        
        # Tokenize
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=self.config.generation.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        response = self._extract_response(full_response)
        
        return response
    
    def _format_prompt(self, prompt: str) -> str:
        """Format prompt in Llama 3.1 chat format"""
        formatted = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        formatted += f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        return formatted
    
    def _extract_response(self, full_text: str) -> str:
        """Extract assistant's response from full generated text"""
        # Split by assistant header and get the last part
        parts = full_text.split("<|start_header_id|>assistant<|end_header_id|>")
        if len(parts) > 1:
            return parts[-1].strip()
        return full_text.strip()
    
    def set_eval_mode(self):
        """Set model to evaluation mode"""
        if self.model:
            self.model.eval()
    
    def set_train_mode(self):
        """Set model to training mode"""
        if self.model:
            self.model.train()
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._is_loaded