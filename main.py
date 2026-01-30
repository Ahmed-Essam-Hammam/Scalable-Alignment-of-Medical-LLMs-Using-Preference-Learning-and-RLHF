import argparse
import sys
import os
from config import AppConfig
from model_manager import ModelManager
from feedback_manager import FeedbackManager
from trainer import IncrementalTrainer
from gradio_ui import GradioInterface

# Optional: Set HuggingFace token directly (if not using CLI login)
# Uncomment and add your token if you prefer
# os.environ["HF_TOKEN"] = "hf_your_token_here"


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Model B - Human-in-the-Loop Training System"
    )
    
    parser.add_argument(
        "--model-sft-path",
        type=str,
        default="./llama3.2-3b-ultramedical-lora",
        help="Path to Model_SFT (base LoRA model)"
    )
    
    parser.add_argument(
        "--model-b-path",
        type=str,
        default="./llama3.2-3b-ultramedical-model-b",
        help="Path to save Model_B"
    )
    
    parser.add_argument(
        "--feedback-path",
        type=str,
        default="./human_feedback_data.jsonl",
        help="Path to feedback data file"
    )
    
    parser.add_argument(
        "--retrain-threshold",
        type=int,
        default=20,
        help="Number of feedback samples before auto-retraining"
    )
    
    parser.add_argument(
        "--no-auto-retrain",
        action="store_true",
        help="Disable automatic retraining"
    )
    
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link for Gradio"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for Gradio interface"
    )
    
    return parser.parse_args()


def main():
    """Main application function"""
    
    # Parse arguments
    args = parse_args()
    
    # Create configuration
    config = AppConfig()
    config.model.model_sft_path = args.model_sft_path
    config.model.model_b_path = args.model_b_path
    config.feedback.feedback_data_path = args.feedback_path
    config.feedback.retrain_threshold = args.retrain_threshold
    config.feedback.auto_retrain = not args.no_auto_retrain
    
    print("="*80)
    print("MODEL B - HUMAN-IN-THE-LOOP TRAINING SYSTEM (CPU MODE)")
    print("="*80)
    
    # CPU warning
    import torch
    if not torch.cuda.is_available():
        print("\n⚠️  WARNING: Running on CPU")
        print("   • Model loading: ~5-10 minutes")
        print("   • Generation: ~30-60 seconds per response")
        print("   • Training: ~10-30 minutes per iteration")
        print("   • Consider using Google Colab for faster performance")
        print()
    
    print(f"\n📋 Configuration:")
    print(f"   Model_SFT Path: {config.model.model_sft_path}")
    print(f"   Model_B Path: {config.model.model_b_path}")
    print(f"   Feedback Path: {config.feedback.feedback_data_path}")
    print(f"   Retrain Threshold: {config.feedback.retrain_threshold}")
    print(f"   Auto-retrain: {config.feedback.auto_retrain}")
    print(f"   Device: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
    print("="*80 + "\n")
    
    # Initialize components
    print("🔧 Initializing components...")
    
    try:
        # Initialize model manager
        model_manager = ModelManager(config)
        model_manager.load_model()
        
        # Initialize feedback manager
        feedback_manager = FeedbackManager(config)
        
        # Initialize trainer
        trainer = IncrementalTrainer(config, model_manager, feedback_manager)
        
        # Initialize Gradio interface
        gradio_interface = GradioInterface(
            config, model_manager, feedback_manager, trainer
        )
        
        print("\n✅ All components initialized successfully!")
        print("="*80)
        
        # Launch Gradio interface
        print("\n🚀 Launching Gradio interface...")
        print(f"   Port: {args.port}")
        print(f"   Share: {args.share}")
        print("="*80 + "\n")
        
        gradio_interface.launch(
            share=args.share,
            server_port=args.port,
            show_error=True,
            debug=True
        )
        
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down gracefully...")
        sys.exit(0)
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()