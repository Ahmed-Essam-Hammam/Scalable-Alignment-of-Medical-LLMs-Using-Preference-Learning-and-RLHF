import os
import json
from datetime import datetime
from transformers import TrainingArguments
from trl import SFTTrainer
from config import AppConfig
from model_manager import ModelManager
from feedback_manager import FeedbackManager

class IncrementalTrainer:
    """Handles incremental training of the model with human feedback"""
    
    def __init__(
        self,
        config: AppConfig,
        model_manager: ModelManager,
        feedback_manager: FeedbackManager
    ):
        self.config = config
        self.model_manager = model_manager
        self.feedback_manager = feedback_manager

    def train(self) -> bool:
        """
        Perform incremental training with accumulated feedback
        
        Returns:
            True if training succeeded, False if no data available
        """
        print("\n" + "="*80)
        print(f"INCREMENTAL RETRAINING - Iteration {self.feedback_manager.iteration + 1}")
        print("="*80)
        
        # Load feedback dataset
        feedback_dataset = self.feedback_manager.load_feedback_dataset()
        
        if feedback_dataset is None or len(feedback_dataset) == 0:
            print("⚠️  No feedback data available for training")
            return False
        
        print(f"Training on {len(feedback_dataset)} feedback samples...")


        # Set model to training mode
        self.model_manager.set_train_mode()

        # Prepare output directory for this iteration
        iteration_num = self.feedback_manager.iteration + 1
        iteration_output_dir = f"{self.config.model.model_b_path}_iter_{iteration_num}"

        # Create training arguments
        training_args_dict = self.config.to_training_args_dict(iteration_output_dir)
        training_args = TrainingArguments(**training_args_dict)
        
        # Create trainer
        trainer = SFTTrainer(
            model=self.model_manager.model,
            args=training_args,
            train_dataset=feedback_dataset,
            tokenizer=self.model_manager.tokenizer,
            dataset_text_field="text",
            max_seq_length=self.config.training.max_seq_length,
            packing=False,
        )

        # Train
        print("🚀 Starting training...")
        try:
            trainer.train()

            # Save the trained model
            print(f"💾 Saving Model_B iteration {iteration_num}...")
            
            # Save iteration-specific checkpoint
            trainer.save_model(iteration_output_dir)
            print(f"✅ Saved iteration checkpoint: {iteration_output_dir}")
            
            # Also save as the main Model_B (latest version)
            trainer.save_model(self.config.model.model_b_path)
            print(f"✅ Saved as latest Model_B: {self.config.model.model_b_path}")
            
            # Save training metadata
            self._save_training_metadata(iteration_num, len(feedback_dataset))
            
            # Increment iteration counter
            self.feedback_manager.increment_iteration()
            
            # Set model back to eval mode
            self.model_manager.set_eval_mode()
            
            print("="*80)
            print("✅ TRAINING COMPLETED SUCCESSFULLY")
            print("="*80)
            
            return True
            
        except Exception as e:
            print(f"\n❌ Training failed with error: {str(e)}")
            # Set model back to eval mode even if training fails
            self.model_manager.set_eval_mode()
            return False
        

    def _save_training_metadata(self, iteration: int, num_samples: int):
        """Save metadata about this training iteration"""
        metadata_path = os.path.join(
            self.config.model.model_b_path,
            "training_metadata.jsonl"
        )

        metadata = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "num_training_samples": num_samples,
            "total_feedback_count": self.feedback_manager.feedback_count,
            "config": {
                "learning_rate": self.config.training.learning_rate,
                "num_epochs": self.config.training.num_train_epochs,
                "batch_size": self.config.training.per_device_train_batch_size,
            }
        }

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

        # Append metadata
        with open(metadata_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(metadata, ensure_ascii=False) + '\n')
        
        print(f"📝 Training metadata saved to {metadata_path}")

    def get_training_history(self):
        """Load and return training history"""
        metadata_path = os.path.join(
            self.config.model.model_b_path,
            "training_metadata.jsonl"
        )
        
        if not os.path.exists(metadata_path):
            return []
        
        import json
        history = []
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                history.append(json.loads(line))
        
        return history



