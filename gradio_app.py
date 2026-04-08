"""
Gradio interface for Smart Delivery Route Optimization Environment.
Compatible with Hugging Face Spaces deployment.
"""

import gradio as gr
from typing import Dict, Any, Optional
import json

from app.env import DeliveryRouteEnv
from app.tasks import TASK_REGISTRY, get_task
from app.grader import TrajectoryTracker, grade_task


class DeliveryRouteGradioApp:
    """Gradio wrapper for the Delivery Route Environment."""
    
    def __init__(self):
        self.env: Optional[DeliveryRouteEnv] = None
        self.tracker: Optional[TrajectoryTracker] = None
        self.current_task_id: Optional[str] = None
    
    def reset_environment(self, task_id: str, seed: int) -> Dict[str, Any]:
        """Reset the environment with selected task."""
        try:
            task = get_task(task_id)
            config = task.config.copy()
            config["seed"] = seed
            
            self.env = DeliveryRouteEnv(task_config=config)
            state = self.env.reset(task_id=task_id)
            self.current_task_id = task_id
            
            self.tracker = TrajectoryTracker()
            self.tracker.finalize(task_id, config)
            
            return self._format_state_display(state)
        except Exception as e:
            return {"error": str(e)}
    
    def take_step(self, action_str: str) -> Dict[str, Any]:
        """Execute a step in the environment."""
        if self.env is None:
            return {"error": "Environment not initialized. Click 'Reset Environment' first."}
        
        try:
            action = int(action_str)
            state, reward, done, info = self.env.step(action)
            
            if self.tracker:
                self.tracker.record_step(
                    state=state.to_dict(),
                    action=action,
                    reward=reward,
                    info=info
                )
            
            result = self._format_state_display(state)
            result["last_reward"] = f"{reward:.4f}"
            result["done"] = done
            result["action_taken"] = action
            
            if done:
                grading = grade_task(
                    self.tracker.to_trajectory_dict(),
                    self.current_task_id
                )
                result["final_score"] = f"{grading.score:.4f}"
            
            return result
        except ValueError as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": f"Step failed: {str(e)}"}
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get the current environment state."""
        if self.env is None:
            return {"error": "Environment not initialized"}
        
        return self._format_state_display(self.env.get_state())
    
    def get_score(self) -> str:
        """Get the current trajectory score."""
        if self.tracker is None:
            return "No trajectory recorded"
        
        trajectory = self.tracker.to_trajectory_dict()
        grading = grade_task(trajectory, self.current_task_id)
        
        breakdown = (
            f"Score: {grading.score:.4f}\n\n"
            f"Completion: {grading.completion_score:.4f}\n"
            f"Time Efficiency: {grading.time_score:.4f}\n"
            f"Fuel Efficiency: {grading.fuel_score:.4f}"
        )
        
        return breakdown
    
    def _format_state_display(self, state) -> Dict[str, Any]:
        """Format state for display."""
        current = state.current_location
        remaining = state.remaining_deliveries
        completed = state.completed_deliveries
        
        remaining_str = "\n".join([
            f"  [{i}] x={loc.x:.1f}, y={loc.y:.1f}, traffic={loc.traffic_multiplier:.2f}"
            for i, loc in enumerate(remaining)
        ]) if remaining else "  None"
        
        completed_str = "\n".join([
            f"  x={loc.x:.1f}, y={loc.y:.1f}"
            for loc in completed
        ]) if completed else "  none"
        
        return {
            "current_location": f"x={current.x:.1f}, y={current.y:.1f}",
            "remaining_deliveries": remaining_str,
            "completed_deliveries": completed_str,
            "time_elapsed": f"{state.time_elapsed:.2f}",
            "fuel_used": f"{state.fuel_used:.2f}",
            "total_distance": f"{state.total_distance:.2f}",
            "traffic_mode": state.traffic_conditions.get("mode", "unknown"),
            "deliveries_remaining": len(remaining),
            "deliveries_completed": len(completed),
            "action_space_size": len(remaining)
        }


app_instance = DeliveryRouteGradioApp()

tasks_info = "\n".join([
    f"- **{t.task_id.upper()}**: {t.description[:60]}..."
    for t in TASK_REGISTRY.get_all_tasks()
])


with gr.Blocks(
    title="Smart Delivery Route Optimization",
    theme=gr.themes.Soft()
) as demo:
    gr.Markdown("# Smart Delivery Route Optimization Environment")
    gr.Markdown(f"""
    This is an OpenEnv-compatible reinforcement learning environment for optimizing delivery routes.
    
    **Available Tasks:**
    {tasks_info}
    
    ## How to Use
    1. Select a task and click "Reset Environment"
    2. Review the remaining deliveries
    3. Enter an action (index of delivery to visit next)
    4. Click "Take Step" to execute the action
    5. Track your progress and final score
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            task_dropdown = gr.Dropdown(
                choices=["easy", "medium", "hard"],
                value="easy",
                label="Task"
            )
            seed_slider = gr.Number(value=42, label="Random Seed")
            reset_btn = gr.Button("Reset Environment", variant="primary")
        
        with gr.Column(scale=2):
            state_display = gr.JSON(label="Current State")
    
    with gr.Row():
        with gr.Column():
            action_input = gr.Number(
                value=0,
                label="Action (Delivery Index)",
                info="Enter the index of the delivery to visit next"
            )
            step_btn = gr.Button("Take Step", variant="primary")
        
        with gr.Column():
            reward_display = gr.Textbox(label="Last Reward", interactive=False)
            done_display = gr.Textbox(label="Episode Status", interactive=False)
            score_btn = gr.Button("Get Score")
            score_display = gr.Textbox(label="Score Breakdown", interactive=False)
    
    reset_btn.click(
        app_instance.reset_environment,
        inputs=[task_dropdown, seed_slider],
        outputs=state_display
    )
    
    step_btn.click(
        app_instance.take_step,
        inputs=[action_input],
        outputs=[state_display, reward_display, done_display]
    )
    
    score_btn.click(
        app_instance.get_score,
        outputs=score_display
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
