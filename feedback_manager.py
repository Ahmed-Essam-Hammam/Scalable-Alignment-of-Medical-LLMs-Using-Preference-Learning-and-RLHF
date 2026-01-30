import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from datasets import Dataset
from config import AppConfig

class FeedbackManager:
    """Manages human feedback data collection and storage"""

    def __init__(self, config: AppConfig):
        self.config = config
        self.feedback_path = config.feedback.feedback_data_path
        self.feedback_count = 0
        self.iteration = 0
        self._load_existing_feedback()

    def _load_existing_feedback(self):
        """Load existing feedback count and iteration"""
        if os.path.exists(self.feedback_path):
            with open(self.feedback_path, 'r', encoding='utf-8') as f:
                self.feedback_count = sum(1 for _ in f)

            if self.feedback_count > 0:
                self.iteration = self.feedback_count // self.config.feedback.retrain_threshold
            
            print(f"📊 Loaded {self.feedback_count} existing feedback samples")
            print(f"📊 Current iteration: {self.iteration}")

    def add_feedback(
        self,
        prompt: str,
        generated_response: str,
        preferred_response: str,
        feedback_type: str
    ) -> int:
        """
        Save feedback to JSONL file
        
        Args:
            prompt: Original user prompt
            generated_response: Model's generated response
            preferred_response: Human's preferred response (same as generated if accepted)
            feedback_type: 'accepted' or 'edited'
        
        Returns:
            Total feedback count after adding this sample
        """
        if feedback_type not in ['accepted', 'edited']:
            raise ValueError("feedback_type must be 'accepted' or 'edited'")
        
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "iteration": self.iteration,
            "prompt": prompt,
            "generated_response": generated_response,
            "preferred_response": preferred_response,
            "feedback_type": feedback_type,
        }

        # Append to JSONL file
        with open(self.feedback_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(feedback_entry, ensure_ascii=False) + '\n')

        self.feedback_count += 1
        print(f"✅ Feedback saved! Total: {self.feedback_count} (Type: {feedback_type})")
        
        return self.feedback_count
    
    def load_feedback_dataset(self) -> Optional[Dataset]:
        """
        Load all feedback as a HuggingFace Dataset for training
        
        Returns:
            Dataset object or None if no feedback exists
        """

        if not os.path.exists(self.feedback_path):
            return None
        
        data= []
        with open(self.feedback_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                # Format as training example
                formatted = {
                    "text": self._format_for_training(
                        entry["prompt"],
                        entry["preferred_response"]
                    )
                }
                data.append(formatted)

        if len(data) == 0:
            return None
        return Dataset.from_list(data)
    
    def _format_for_training(self, prompt: str, response: str) -> str:
        """Format feedback into Llama 3.1 chat format for training"""
        formatted = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        formatted += f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        formatted += f"{response}<|eot_id|>"
        return formatted
    
    def should_retrain(self) -> bool:
        """Check if we've reached the threshold for retraining"""
        if not self.config.feedback.auto_retrain:
            return False
        
        threshold = self.config.feedback.retrain_threshold
        return self.feedback_count > 0 and self.feedback_count % threshold == 0

    def increment_iteration(self):
        """Increment the training iteration counter"""
        self.iteration += 1
        print(f"📈 Advanced to iteration {self.iteration}")

    def get_statistics(self) -> Dict[str, any]:
        """Get feedback statistics"""
        stats = {
            "total_feedback": self.feedback_count,
            "current_iteration": self.iteration,
            "retrain_threshold": self.config.feedback.retrain_threshold,
            "next_retrain_at": ((self.feedback_count // self.config.feedback.retrain_threshold) + 1) * self.config.feedback.retrain_threshold,
            "samples_until_retrain": self.config.feedback.retrain_threshold - (self.feedback_count % self.config.feedback.retrain_threshold),
        }

        if os.path.exists(self.feedback_path):
            accepted = 0
            edited = 0
            with open(self.feedback_path, 'r', encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line)
                    if entry.get("feedback_type") == "accepted":
                        accepted += 1
                    elif entry.get("feedback_type") == "edited":
                        edited += 1

            stats["accepted_count"] = accepted
            stats["edited_count"] = edited
            stats["edit_rate"] = edited / self.feedback_count if self.feedback_count > 0 else 0
        
        return stats
    
    def export_feedback(self, output_path: str, format: str = "jsonl"):
        """
        Export feedback data to a file
        
        Args:
            output_path: Path to save the exported data
            format: 'jsonl' or 'csv'
        """
        if not os.path.exists(self.feedback_path):
            print("⚠️  No feedback data to export")
            return
        
        if format == "jsonl":
            # Copy the existing JSONL file
            import shutil
            shutil.copy(self.feedback_path, output_path)
            print(f"✅ Exported {self.feedback_count} samples to {output_path}")
        
        elif format == "csv":
            # Convert to CSV
            import pandas as pd
            data = []
            with open(self.feedback_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))
            
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)
            print(f"✅ Exported {self.feedback_count} samples to {output_path}")
        
        else:
            raise ValueError("format must be 'jsonl' or 'csv'")


