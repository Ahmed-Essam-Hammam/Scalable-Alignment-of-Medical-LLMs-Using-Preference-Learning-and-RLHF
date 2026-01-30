"""
gradio_ui.py
============
Gradio web interface for human-in-the-loop feedback
"""

import gradio as gr
from typing import Tuple
from config import AppConfig
from model_manager import ModelManager
from feedback_manager import FeedbackManager
from trainer import IncrementalTrainer


class GradioInterface:
    """Gradio web interface for Model B HITL system"""
    
    def __init__(
        self,
        config: AppConfig,
        model_manager: ModelManager,
        feedback_manager: FeedbackManager,
        trainer: IncrementalTrainer
    ):
        self.config = config
        self.model_manager = model_manager
        self.feedback_manager = feedback_manager
        self.trainer = trainer
        
        # State variables
        self.current_prompt = ""
        self.current_response = ""
    
    def on_generate(self, prompt: str) -> Tuple[str, str, str]:
        """
        Generate response from the model
        
        Returns:
            (response, current_prompt, current_response)
        """
        if not prompt.strip():
            return "⚠️  Please enter a prompt", "", ""
        
        try:
            response = self.model_manager.generate_response(prompt)
            self.current_prompt = prompt
            self.current_response = response
            return response, prompt, response
        except Exception as e:
            return f"❌ Error generating response: {str(e)}", "", ""
    
    def on_accept(self, prompt: str, response: str) -> Tuple[str, int]:
        """
        Accept the generated response
        
        Returns:
            (status_message, feedback_count)
        """
        if not prompt or not response:
            return "⚠️  No response to accept", self.feedback_manager.feedback_count
        
        try:
            count = self.feedback_manager.add_feedback(
                prompt, response, response, "accepted"
            )
            
            # Check if we should retrain
            if self.feedback_manager.should_retrain():
                self.trainer.train()
                return f"✅ Accepted! Auto-retrained at {count} samples", count
            
            samples_until_retrain = self.feedback_manager.get_statistics()["samples_until_retrain"]
            return f"✅ Accepted! ({samples_until_retrain} samples until next retrain)", count
            
        except Exception as e:
            return f"❌ Error: {str(e)}", self.feedback_manager.feedback_count
    
    def on_edit_and_save(
        self,
        prompt: str,
        generated_response: str,
        edited_response: str
    ) -> Tuple[str, int]:
        """
        Save edited response as preferred
        
        Returns:
            (status_message, feedback_count)
        """
        if not prompt or not edited_response.strip():
            return "⚠️  Please provide both prompt and edited response", self.feedback_manager.feedback_count
        
        try:
            count = self.feedback_manager.add_feedback(
                prompt, generated_response, edited_response, "edited"
            )
            
            # Check if we should retrain
            if self.feedback_manager.should_retrain():
                self.trainer.train()
                return f"✅ Edited response saved! Auto-retrained at {count} samples", count
            
            samples_until_retrain = self.feedback_manager.get_statistics()["samples_until_retrain"]
            return f"✅ Edited response saved! ({samples_until_retrain} samples until next retrain)", count
            
        except Exception as e:
            return f"❌ Error: {str(e)}", self.feedback_manager.feedback_count
    
    def manual_retrain(self) -> str:
        """Manually trigger retraining"""
        try:
            success = self.trainer.train()
            if success:
                return f"✅ Retraining completed! Now at iteration {self.feedback_manager.iteration}"
            else:
                return "⚠️  No feedback data available for training"
        except Exception as e:
            return f"❌ Training failed: {str(e)}"
    
    def get_statistics(self) -> str:
        """Get formatted statistics"""
        stats = self.feedback_manager.get_statistics()
        
        stats_text = f"""
        📊 **Current Statistics**
        
        - Total Feedback: {stats['total_feedback']}
        - Current Iteration: {stats['current_iteration']}
        - Samples Until Next Retrain: {stats['samples_until_retrain']}
        - Next Retrain At: {stats['next_retrain_at']} samples
        """
        
        if 'accepted_count' in stats:
            stats_text += f"""
        - Accepted: {stats['accepted_count']} ({100*stats['accepted_count']/stats['total_feedback']:.1f}%)
        - Edited: {stats['edited_count']} ({100*stats['edited_count']/stats['total_feedback']:.1f}%)
        - Edit Rate: {100*stats['edit_rate']:.1f}%
            """
        
        return stats_text
    
    def build_interface(self) -> gr.Blocks:
        """Build and return the Gradio interface"""
        
        with gr.Blocks(
            title="Model B - Human-in-the-Loop Training",
            theme=gr.themes.Soft()
        ) as demo:
            
            gr.Markdown("# 🔄 Model B: Human-in-the-Loop Training Interface")
            gr.Markdown("""
            This interface allows you to:
            1. **Generate** responses from the current model
            2. **Accept** good responses or **Edit** poor ones
            3. **Automatically retrain** after collecting enough feedback
            """)
            
            # Main interaction area
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📝 Input")
                    prompt_input = gr.Textbox(
                        label="Medical Question/Prompt",
                        placeholder="Enter a medical question...",
                        lines=5
                    )
                    generate_btn = gr.Button("🤖 Generate Response", variant="primary", size="lg")
                    
                with gr.Column(scale=1):
                    gr.Markdown("### 🎯 Generated Response")
                    response_output = gr.Textbox(
                        label="Model's Response",
                        lines=10,
                        interactive=False
                    )
            
            # Feedback buttons
            with gr.Row():
                accept_btn = gr.Button("✅ Accept Response", variant="secondary", size="lg")
                gr.Markdown("**OR**", elem_id="or-text")
            
            # Edit area
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### ✏️ Edit Response (if needed)")
                    edited_response = gr.Textbox(
                        label="Preferred Response",
                        placeholder="Edit the response or write your own preferred response...",
                        lines=10
                    )
                    edit_save_btn = gr.Button("💾 Save Edited Response", variant="primary", size="lg")
            
            gr.Markdown("---")
            
            # Status and controls
            with gr.Row():
                with gr.Column(scale=2):
                    feedback_status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        lines=2
                    )
                with gr.Column(scale=1):
                    feedback_counter = gr.Number(
                        label="Total Feedback Samples",
                        value=self.feedback_manager.feedback_count,
                        interactive=False
                    )
            
            with gr.Row():
                manual_retrain_btn = gr.Button("🔄 Manual Retrain Now", variant="stop")
                stats_btn = gr.Button("📊 Show Statistics", variant="secondary")
            
            stats_display = gr.Markdown(self.get_statistics())
            
            # Info section
            gr.Markdown(f"""
            ---
            ### ℹ️ Configuration
            - **Auto-retrain threshold**: Every {self.config.feedback.retrain_threshold} feedback samples
            - **Current iteration**: {self.feedback_manager.iteration}
            - **Feedback saved to**: `{self.config.feedback.feedback_data_path}`
            - **Model saved to**: `{self.config.model.model_b_path}`
            """)
            
            # Hidden states to track current interaction
            current_prompt_state = gr.State("")
            current_response_state = gr.State("")
            
            # Event handlers
            generate_btn.click(
                fn=self.on_generate,
                inputs=[prompt_input],
                outputs=[response_output, current_prompt_state, current_response_state]
            )
            
            accept_btn.click(
                fn=self.on_accept,
                inputs=[current_prompt_state, current_response_state],
                outputs=[feedback_status, feedback_counter]
            )
            
            edit_save_btn.click(
                fn=self.on_edit_and_save,
                inputs=[current_prompt_state, current_response_state, edited_response],
                outputs=[feedback_status, feedback_counter]
            )
            
            manual_retrain_btn.click(
                fn=self.manual_retrain,
                outputs=[feedback_status]
            )
            
            stats_btn.click(
                fn=self.get_statistics,
                outputs=[stats_display]
            )
        
        return demo
    
    def launch(self, **kwargs):
        """Launch the Gradio interface"""
        demo = self.build_interface()
        demo.launch(**kwargs)