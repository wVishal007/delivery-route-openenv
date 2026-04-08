"""Smart Delivery Route Optimization Environment."""

from app.env import DeliveryRouteEnv, DeliveryState, Location
from app.tasks import TASK_REGISTRY, get_task, get_all_tasks, get_task_config
from app.grader import grade_task, TrajectoryTracker

__version__ = "1.0.0"
__all__ = [
    "DeliveryRouteEnv",
    "DeliveryState",
    "Location",
    "TASK_REGISTRY",
    "get_task",
    "get_all_tasks",
    "get_task_config",
    "grade_task",
    "TrajectoryTracker"
]
