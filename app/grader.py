from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class GradingResult:
    score: float
    completion_score: float
    time_score: float
    fuel_score: float
    details: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "completion_score": round(self.completion_score, 4),
            "time_score": round(self.time_score, 4),
            "fuel_score": round(self.fuel_score, 4),
            "details": self.details
        }


class TrajectoryTracker:
    """Track trajectory data for grading."""
    
    def __init__(self):
        self.states: List[Dict[str, Any]] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.infos: List[Dict[str, Any]] = []
        self.task_id: Optional[str] = None
        self.config: Optional[Dict[str, Any]] = None
    
    def record_step(
        self,
        state: Dict[str, Any],
        action: int,
        reward: float,
        info: Dict[str, Any]
    ):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.infos.append(info)
    
    def finalize(self, task_id: str, config: Dict[str, Any]):
        self.task_id = task_id
        self.config = config
    
    def to_trajectory_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "config": self.config,
            "states": self.states,
            "actions": self.actions,
            "rewards": self.rewards,
            "infos": self.infos,
            "total_steps": len(self.actions),
            "total_reward": sum(self.rewards) if self.rewards else 0.0
        }


def grade_delivery_completion(trajectory: Dict[str, Any]) -> float:
    """Grade based on delivery completion rate."""
    config = trajectory.get("config", {})
    expected_deliveries = config.get("num_locations", 0)
    
    if expected_deliveries == 0:
        return 0.0
    
    final_state = trajectory.get("states", [{}])[-1] if trajectory.get("states") else {}
    completed = final_state.get("deliveries_completed", 0)
    
    return completed / expected_deliveries


def grade_time_efficiency(trajectory: Dict[str, Any]) -> float:
    """Grade based on time efficiency."""
    config = trajectory.get("config", {})
    time_limit = config.get("time_limit", 1000.0)
    
    if time_limit <= 0:
        return 0.5
    
    final_state = trajectory.get("states", [{}])[-1] if trajectory.get("states") else {}
    time_elapsed = final_state.get("time_elapsed", time_limit)
    
    if time_elapsed <= time_limit:
        efficiency = 1.0 - (time_elapsed / time_limit) * 0.5
    else:
        over_time = (time_elapsed - time_limit) / time_limit
        efficiency = max(0.0, 0.5 - over_time * 0.3)
    
    return max(0.0, min(1.0, efficiency))


def grade_fuel_efficiency(trajectory: Dict[str, Any]) -> float:
    """Grade based on fuel efficiency."""
    config = trajectory.get("config", {})
    fuel_limit = config.get("fuel_limit", 500.0)
    
    if fuel_limit <= 0:
        return 0.5
    
    final_state = trajectory.get("states", [{}])[-1] if trajectory.get("states") else {}
    fuel_used = final_state.get("fuel_used", 0.0)
    
    if fuel_used <= fuel_limit:
        efficiency = 1.0 - (fuel_used / fuel_limit) * 0.4
    else:
        over_fuel = (fuel_used - fuel_limit) / fuel_limit
        efficiency = max(0.0, 0.6 - over_fuel * 0.4)
    
    return max(0.0, min(1.0, efficiency))


def grade_easy_task(trajectory: Dict[str, Any]) -> float:
    """
    Grade performance on the easy task (Campus Delivery).
    
    Scoring weights:
    - Completion: 50%
    - Time efficiency: 30%
    - Fuel efficiency: 20%
    """
    completion = grade_delivery_completion(trajectory)
    time_score = grade_time_efficiency(trajectory)
    fuel_score = grade_fuel_efficiency(trajectory)
    
    score = (
        0.50 * completion +
        0.30 * time_score +
        0.20 * fuel_score
    )
    
    result = GradingResult(
        score=score,
        completion_score=completion,
        time_score=time_score,
        fuel_score=fuel_score,
        details={
            "task": "easy",
            "weights": {"completion": 0.50, "time": 0.30, "fuel": 0.20},
            "total_reward": trajectory.get("total_reward", 0.0),
            "steps": trajectory.get("total_steps", 0)
        }
    )
    
    return round(result.score, 4)


def grade_medium_task(trajectory: Dict[str, Any]) -> float:
    """
    Grade performance on the medium task (City Logistics).
    
    Scoring weights:
    - Completion: 40%
    - Time efficiency: 35%
    - Fuel efficiency: 25%
    """
    completion = grade_delivery_completion(trajectory)
    time_score = grade_time_efficiency(trajectory)
    fuel_score = grade_fuel_efficiency(trajectory)
    
    score = (
        0.40 * completion +
        0.35 * time_score +
        0.25 * fuel_score
    )
    
    result = GradingResult(
        score=score,
        completion_score=completion,
        time_score=time_score,
        fuel_score=fuel_score,
        details={
            "task": "medium",
            "weights": {"completion": 0.40, "time": 0.35, "fuel": 0.25},
            "total_reward": trajectory.get("total_reward", 0.0),
            "steps": trajectory.get("total_steps", 0)
        }
    )
    
    return round(result.score, 4)


def grade_hard_task(trajectory: Dict[str, Any]) -> float:
    """
    Grade performance on the hard task (Metro Distribution).
    
    Scoring weights:
    - Completion: 35%
    - Time efficiency: 40%
    - Fuel efficiency: 25%
    
    Additional penalties for constraint violations.
    """
    completion = grade_delivery_completion(trajectory)
    time_score = grade_time_efficiency(trajectory)
    fuel_score = grade_fuel_efficiency(trajectory)
    
    infos = trajectory.get("infos", [])
    has_violation = any(info.get("constraint_violation", False) for info in infos)
    
    violation_penalty = 0.15 if has_violation else 0.0
    
    score = (
        0.35 * completion +
        0.40 * time_score +
        0.25 * fuel_score -
        violation_penalty
    )
    
    result = GradingResult(
        score=score,
        completion_score=completion,
        time_score=time_score,
        fuel_score=fuel_score,
        details={
            "task": "hard",
            "weights": {"completion": 0.35, "time": 0.40, "fuel": 0.25},
            "violation_penalty": violation_penalty,
            "has_constraint_violation": has_violation,
            "total_reward": trajectory.get("total_reward", 0.0),
            "steps": trajectory.get("total_steps", 0)
        }
    )
    
    return round(max(0.0, min(1.0, result.score)), 4)


def grade_task(trajectory: Dict[str, Any], task_id: str) -> GradingResult:
    """
    Grade a trajectory for a specific task.
    
    Args:
        trajectory: Dictionary containing states, actions, rewards, etc.
        task_id: The task identifier
        
    Returns:
        GradingResult with score breakdown
    """
    if task_id == "easy":
        score = grade_easy_task(trajectory)
    elif task_id == "medium":
        score = grade_medium_task(trajectory)
    elif task_id == "hard":
        score = grade_hard_task(trajectory)
    else:
        raise ValueError(f"Unknown task: {task_id}")
    
    completion = grade_delivery_completion(trajectory)
    time_score = grade_time_efficiency(trajectory)
    fuel_score = grade_fuel_efficiency(trajectory)
    
    return GradingResult(
        score=score,
        completion_score=completion,
        time_score=time_score,
        fuel_score=fuel_score,
        details={"task_id": task_id}
    )


def get_score_breakdown(trajectory: Dict[str, Any]) -> Dict[str, float]:
    """Get detailed score breakdown for a trajectory."""
    return {
        "completion": grade_delivery_completion(trajectory),
        "time": grade_time_efficiency(trajectory),
        "fuel": grade_fuel_efficiency(trajectory)
    }
