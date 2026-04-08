from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import numpy as np
import math


@dataclass
class Location:
    x: float
    y: float
    traffic_multiplier: float = 1.0
    
    def distance_to(self, other: 'Location') -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
    
    def to_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y}


@dataclass
class DeliveryState:
    current_location: Location
    remaining_deliveries: List[Location]
    completed_deliveries: List[Location]
    time_elapsed: float
    fuel_used: float
    total_distance: float
    traffic_conditions: Dict[str, Any]
    depot_location: Location
    task_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_location": self.current_location.to_dict(),
            "remaining_deliveries": [loc.to_dict() for loc in self.remaining_deliveries],
            "completed_deliveries": [loc.to_dict() for loc in self.completed_deliveries],
            "time_elapsed": round(self.time_elapsed, 2),
            "fuel_used": round(self.fuel_used, 2),
            "total_distance": round(self.total_distance, 2),
            "traffic_conditions": self.traffic_conditions,
            "depot_location": self.depot_location.to_dict(),
            "task_id": self.task_id,
            "deliveries_completed": len(self.completed_deliveries),
            "deliveries_remaining": len(self.remaining_deliveries)
        }


class DeliveryRouteEnv:
    """
    Smart Delivery Route Optimization Environment.
    
    Simulates a logistics system where an AI agent optimizes delivery routes
    to minimize time, fuel consumption, and traffic delays.
    """
    
    VERSION = "1.0.0"
    
    def __init__(self, task_config: Optional[Dict[str, Any]] = None):
        self.task_config = task_config or self._default_config()
        self.state: Optional[DeliveryState] = None
        self.action_space_size: int = 0
        self._initialize_from_config()
    
    def _default_config(self) -> Dict[str, Any]:
        return {
            "num_locations": 5,
            "map_size": 100.0,
            "traffic_enabled": False,
            "time_limit": 1000.0,
            "fuel_limit": 500.0,
            "depot": {"x": 50.0, "y": 50.0}
        }
    
    def _initialize_from_config(self):
        config = self.task_config
        
        depot = Location(
            x=config.get("depot", {}).get("x", 50.0),
            y=config.get("depot", {}).get("y", 50.0)
        )
        
        self.depot_location = depot
        self.num_locations = config.get("num_locations", 5)
        self.map_size = config.get("map_size", 100.0)
        self.traffic_enabled = config.get("traffic_enabled", False)
        self.time_limit = config.get("time_limit", 1000.0)
        self.fuel_limit = config.get("fuel_limit", 500.0)
        self.base_speed = config.get("base_speed", 1.0)
        self.fuel_per_distance = config.get("fuel_per_distance", 0.1)
        
        self._generate_locations()
    
    def _generate_locations(self):
        np.random.seed(self.task_config.get("seed", 42))
        
        self.delivery_locations: List[Location] = []
        
        for _ in range(self.num_locations):
            x = np.random.uniform(0, self.map_size)
            y = np.random.uniform(0, self.map_size)
            
            if self.traffic_enabled:
                traffic = np.random.uniform(1.0, 2.5)
            else:
                traffic = 1.0
            
            self.delivery_locations.append(Location(x=x, y=y, traffic_multiplier=traffic))
    
    def reset(self, task_id: str = "easy") -> DeliveryState:
        """Reset the environment to initial state for the given task."""
        self.task_id = task_id
        
        self.state = DeliveryState(
            current_location=Location(x=self.depot_location.x, y=self.depot_location.y),
            remaining_deliveries=self.delivery_locations.copy(),
            completed_deliveries=[],
            time_elapsed=0.0,
            fuel_used=0.0,
            total_distance=0.0,
            traffic_conditions=self._get_traffic_conditions(),
            depot_location=self.depot_location,
            task_id=task_id
        )
        
        self.action_space_size = len(self.state.remaining_deliveries)
        
        return self.state
    
    def _get_traffic_conditions(self) -> Dict[str, Any]:
        if not self.traffic_enabled:
            return {
                "mode": "static",
                "global_multiplier": 1.0,
                "congestion_level": "none"
            }
        
        congestion = np.random.choice(["low", "moderate", "high", "severe"])
        congestion_factors = {
            "low": 1.2,
            "moderate": 1.5,
            "high": 2.0,
            "severe": 2.5
        }
        
        return {
            "mode": "dynamic" if self.task_config.get("dynamic_traffic", False) else "static",
            "global_multiplier": congestion_factors.get(congestion, 1.5),
            "congestion_level": congestion
        }
    
    def step(self, action: int) -> Tuple[DeliveryState, float, bool, Dict[str, Any]]:
        """
        Execute an action in the environment.
        
        Args:
            action: Index of the delivery location to visit next
            
        Returns:
            Tuple of (new_state, reward, done, info)
        """
        if self.state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        if action < 0 or action >= len(self.state.remaining_deliveries):
            raise ValueError(f"Invalid action {action}. Valid range: 0-{len(self.state.remaining_deliveries)-1}")
        
        target_location = self.state.remaining_deliveries[action]
        
        distance = self.state.current_location.distance_to(target_location)
        
        traffic_multiplier = target_location.traffic_multiplier
        if self.state.traffic_conditions.get("mode") == "dynamic":
            traffic_multiplier *= self.state.traffic_conditions.get("global_multiplier", 1.0)
        
        travel_time = (distance / self.base_speed) * traffic_multiplier
        fuel_consumed = distance * self.fuel_per_distance
        
        self.state.time_elapsed += travel_time
        self.state.fuel_used += fuel_consumed
        self.state.total_distance += distance
        
        completed_location = self.state.remaining_deliveries.pop(action)
        self.state.current_location = completed_location
        self.state.completed_deliveries.append(completed_location)
        
        reward = self._calculate_reward(distance, travel_time, fuel_consumed)
        
        done = len(self.state.remaining_deliveries) == 0
        if done:
            reward += self._calculate_completion_bonus()
        
        info = {
            "distance_traveled": round(distance, 2),
            "time_taken": round(travel_time, 2),
            "fuel_consumed": round(fuel_consumed, 2),
            "deliveries_completed": len(self.state.completed_deliveries),
            "deliveries_remaining": len(self.state.remaining_deliveries),
            "traffic_multiplier": traffic_multiplier
        }
        
        if self.state.time_elapsed > self.time_limit or self.state.fuel_used > self.fuel_limit:
            done = True
            reward = max(0.0, reward - 0.3)
            info["constraint_violation"] = True
        
        return self.state, reward, done, info
    
    def _calculate_reward(
        self, 
        distance: float, 
        travel_time: float, 
        fuel_consumed: float
    ) -> float:
        delivery_reward = 0.15
        
        max_acceptable_time = (distance / self.base_speed) * 1.5
        if travel_time <= max_acceptable_time:
            efficiency_bonus = 0.1 * (1 - (travel_time / max_acceptable_time))
        else:
            efficiency_bonus = -0.05 * ((travel_time / max_acceptable_time) - 1)
        
        fuel_penalty = 0.02 * (fuel_consumed / self.fuel_limit) if self.fuel_limit > 0 else 0
        
        traffic_penalty = 0.0
        if self.traffic_enabled:
            traffic_mult = self.state.traffic_conditions.get("global_multiplier", 1.0)
            if traffic_mult > 1.3:
                traffic_penalty = 0.03 * (traffic_mult - 1.3)
        
        raw_reward = (
            delivery_reward +
            efficiency_bonus -
            fuel_penalty -
            traffic_penalty
        )
        
        return max(0.0, min(1.0, raw_reward))
    
    def _calculate_completion_bonus(self) -> float:
        if self.num_locations == 0:
            return 0.0
        
        completion_rate = len(self.state.completed_deliveries) / self.num_locations
        
        time_efficiency = 1.0 - (self.state.time_elapsed / self.time_limit) if self.time_limit > 0 else 0.5
        time_efficiency = max(0.0, min(1.0, time_efficiency))
        
        fuel_efficiency = 1.0 - (self.state.fuel_used / self.fuel_limit) if self.fuel_limit > 0 else 0.5
        fuel_efficiency = max(0.0, min(1.0, fuel_efficiency))
        
        bonus = 0.2 * completion_rate + 0.15 * time_efficiency + 0.1 * fuel_efficiency
        
        return min(0.45, bonus)
    
    def get_state(self) -> DeliveryState:
        """Get the current state of the environment."""
        if self.state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self.state
    
    def get_action_space(self) -> int:
        """Get the current action space size."""
        if self.state is None:
            return self.num_locations
        return len(self.state.remaining_deliveries)
    
    def get_observation(self) -> Dict[str, Any]:
        """Get a normalized observation for the agent."""
        if self.state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        state = self.state
        
        current_norm = {
            "x": state.current_location.x / self.map_size,
            "y": state.current_location.y / self.map_size
        }
        
        remaining_norm = []
        for loc in state.remaining_deliveries:
            remaining_norm.append({
                "x": loc.x / self.map_size,
                "y": loc.y / self.map_size,
                "traffic": loc.traffic_multiplier / 3.0
            })
        
        return {
            "current_location": current_norm,
            "remaining_deliveries": remaining_norm,
            "time_elapsed": state.time_elapsed / self.time_limit,
            "fuel_used": state.fuel_used / self.fuel_limit,
            "progress": len(state.completed_deliveries) / self.num_locations if self.num_locations > 0 else 0,
            "action_space_size": self.get_action_space()
        }
