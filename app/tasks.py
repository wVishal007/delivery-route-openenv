from typing import Dict, Any, Callable, List
from dataclasses import dataclass
import numpy as np


@dataclass
class TaskDefinition:
    task_id: str
    name: str
    description: str
    difficulty: str
    config: Dict[str, Any]
    grader_fn: Callable[[Dict[str, Any]], float]
    max_steps: int
    reward_threshold: float


class TaskRegistry:
    """Registry for all available delivery route optimization tasks."""
    
    def __init__(self):
        self._tasks: Dict[str, TaskDefinition] = {}
        self._register_default_tasks()
    
    def _register_default_tasks(self):
        self.register_task(self._create_easy_task())
        self.register_task(self._create_medium_task())
        self.register_task(self._create_hard_task())
    
    def _create_easy_task(self) -> TaskDefinition:
        def easy_grader(trajectory: Dict[str, Any]) -> float:
            from app.grader import grade_easy_task
            return grade_easy_task(trajectory)
        
        config = {
            "num_locations": 3,
            "map_size": 50.0,
            "traffic_enabled": False,
            "time_limit": 200.0,
            "fuel_limit": 100.0,
            "base_speed": 1.5,
            "fuel_per_distance": 0.08,
            "seed": 42,
            "depot": {"x": 25.0, "y": 25.0}
        }
        
        return TaskDefinition(
            task_id="easy",
            name="Campus Delivery",
            description=(
                "A simple campus delivery route with 3 locations spread across a small area. "
                "No traffic delays. Agent must learn basic route selection."
            ),
            difficulty="easy",
            config=config,
            grader_fn=easy_grader,
            max_steps=10,
            reward_threshold=0.6
        )
    
    def _create_medium_task(self) -> TaskDefinition:
        def medium_grader(trajectory: Dict[str, Any]) -> float:
            from app.grader import grade_medium_task
            return grade_medium_task(trajectory)
        
        config = {
            "num_locations": 6,
            "map_size": 100.0,
            "traffic_enabled": True,
            "dynamic_traffic": False,
            "time_limit": 300.0,
            "fuel_limit": 150.0,
            "base_speed": 1.2,
            "fuel_per_distance": 0.1,
            "seed": 123,
            "depot": {"x": 50.0, "y": 50.0}
        }
        
        return TaskDefinition(
            task_id="medium",
            name="City Logistics",
            description=(
                "Urban delivery route with 6 locations across a city grid. "
                "Moderate traffic congestion increases travel time. "
                "Agent must balance speed and fuel efficiency under time pressure."
            ),
            difficulty="medium",
            config=config,
            grader_fn=medium_grader,
            max_steps=20,
            reward_threshold=0.5
        )
    
    def _create_hard_task(self) -> TaskDefinition:
        def hard_grader(trajectory: Dict[str, Any]) -> float:
            from app.grader import grade_hard_task
            return grade_hard_task(trajectory)
        
        config = {
            "num_locations": 12,
            "map_size": 200.0,
            "traffic_enabled": True,
            "dynamic_traffic": True,
            "time_limit": 400.0,
            "fuel_limit": 180.0,
            "base_speed": 1.0,
            "fuel_per_distance": 0.12,
            "seed": 456,
            "depot": {"x": 100.0, "y": 100.0}
        }
        
        return TaskDefinition(
            task_id="hard",
            name="Metro Distribution",
            description=(
                "Complex multi-zone distribution network with 12 delivery points. "
                "Dynamic traffic conditions that change throughout the route. "
                "Strict fuel and time constraints require optimal route planning. "
                "Agent must adapt to changing conditions and prioritize efficiency."
            ),
            difficulty="hard",
            config=config,
            grader_fn=hard_grader,
            max_steps=40,
            reward_threshold=0.4
        )
    
    def register_task(self, task: TaskDefinition):
        """Register a new task in the registry."""
        self._tasks[task.task_id] = task
    
    def get_task(self, task_id: str) -> TaskDefinition:
        """Get a task by its ID."""
        if task_id not in self._tasks:
            raise ValueError(f"Unknown task: {task_id}")
        return self._tasks[task_id]
    
    def get_all_tasks(self) -> List[TaskDefinition]:
        """Get all registered tasks."""
        return list(self._tasks.values())
    
    def get_tasks_by_difficulty(self, difficulty: str) -> List[TaskDefinition]:
        """Get tasks filtered by difficulty level."""
        return [t for t in self._tasks.values() if t.difficulty == difficulty]
    
    def list_task_ids(self) -> List[str]:
        """List all available task IDs."""
        return list(self._tasks.keys())


TASK_REGISTRY = TaskRegistry()


def get_task(task_id: str) -> TaskDefinition:
    """Convenience function to get a task from the registry."""
    return TASK_REGISTRY.get_task(task_id)


def get_all_tasks() -> List[TaskDefinition]:
    """Convenience function to get all tasks."""
    return TASK_REGISTRY.get_all_tasks()


def get_task_config(task_id: str) -> Dict[str, Any]:
    """Get the configuration for a specific task."""
    return TASK_REGISTRY.get_task(task_id).config
