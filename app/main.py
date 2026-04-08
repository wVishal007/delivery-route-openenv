from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from app.env import DeliveryRouteEnv, DeliveryState
from app.tasks import TASK_REGISTRY, get_task, get_task_config
from app.grader import TrajectoryTracker, grade_task, get_score_breakdown


app = FastAPI(
    title="Smart Delivery Route Optimization Environment",
    description="OpenEnv-compatible RL environment for delivery route optimization",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_current_env: Optional[DeliveryRouteEnv] = None
_trajectory_tracker: Optional[TrajectoryTracker] = None


class ResetRequest(BaseModel):
    task_id: str = Field(default="easy", description="Task ID to reset the environment with")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")


class StepRequest(BaseModel):
    action: int = Field(..., description="Action (index of delivery location to visit)")


class EnvResponse(BaseModel):
    state: Dict[str, Any]
    observation: Dict[str, Any]
    action_space_size: int


class StepResponse(BaseModel):
    state: Dict[str, Any]
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]
    action_space_size: int


class TaskInfo(BaseModel):
    task_id: str
    name: str
    description: str
    difficulty: str
    max_steps: int
    reward_threshold: float


class TaskListResponse(BaseModel):
    tasks: List[TaskInfo]
    total: int


class GradingResponse(BaseModel):
    score: float
    completion_score: float
    time_score: float
    fuel_score: float
    details: Dict[str, Any]


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Smart Delivery Route Optimization Environment",
        "version": "1.0.0",
        "openenv_compatible": True
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "environment_initialized": _current_env is not None,
        "trajectory_tracking": _trajectory_tracker is not None
    }


@app.post("/reset", response_model=EnvResponse)
async def reset_post(request: ResetRequest):
    """
    Reset the environment to initial state.
    
    This initializes or reinitializes the environment with the specified task.
    """
    return await _do_reset(request.task_id, request.seed)


@app.get("/reset", response_model=EnvResponse)
async def reset_get(task_id: str = "easy", seed: Optional[int] = None):
    """
    Reset the environment to initial state (GET method).
    
    This initializes or reinitializes the environment with the specified task.
    """
    return await _do_reset(task_id, seed)


async def _do_reset(task_id: str, seed: Optional[int]) -> EnvResponse:
    """Internal reset logic shared by both GET and POST."""
    global _current_env, _trajectory_tracker
    
    try:
        task = get_task(task_id)
        config = task.config.copy()
        
        if seed is not None:
            config["seed"] = seed
        
        _current_env = DeliveryRouteEnv(task_config=config)
        state = _current_env.reset(task_id=task_id)
        
        _trajectory_tracker = TrajectoryTracker()
        _trajectory_tracker.finalize(task_id, config)
        
        return EnvResponse(
            state=state.to_dict(),
            observation=_current_env.get_observation(),
            action_space_size=_current_env.get_action_space()
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


@app.get("/state", response_model=EnvResponse)
async def state():
    """
    Get the current state of the environment.
    
    Returns the full state dictionary, observation, and action space size.
    """
    global _current_env, _trajectory_tracker
    
    if _current_env is None or _current_env.get_state() is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first."
        )
    
    current_state = _current_env.get_state()
    
    return EnvResponse(
        state=current_state.to_dict(),
        observation=_current_env.get_observation(),
        action_space_size=_current_env.get_action_space()
    )


@app.post("/step", response_model=StepResponse)
async def step(request: StepRequest):
    """
    Execute an action in the environment.
    
    Takes an action (delivery location index) and returns the new state,
    reward, done flag, and additional information.
    """
    global _current_env, _trajectory_tracker
    
    if _current_env is None or _current_env.get_state() is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first."
        )
    
    try:
        action = request.action
        state, reward, done, info = _current_env.step(action)
        
        if _trajectory_tracker is not None:
            _trajectory_tracker.record_step(
                state=state.to_dict(),
                action=action,
                reward=reward,
                info=info
            )
        
        return StepResponse(
            state=state.to_dict(),
            observation=_current_env.get_observation(),
            reward=reward,
            done=done,
            info=info,
            action_space_size=_current_env.get_action_space()
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step failed: {str(e)}")


@app.get("/tasks", response_model=TaskListResponse)
async def list_tasks():
    """
    List all available tasks.
    
    Returns information about each task including difficulty and thresholds.
    """
    tasks = TASK_REGISTRY.get_all_tasks()
    
    task_infos = [
        TaskInfo(
            task_id=t.task_id,
            name=t.name,
            description=t.description,
            difficulty=t.difficulty,
            max_steps=t.max_steps,
            reward_threshold=t.reward_threshold
        )
        for t in tasks
    ]
    
    return TaskListResponse(tasks=task_infos, total=len(task_infos))


@app.get("/tasks/{task_id}", response_model=TaskInfo)
async def get_task_info(task_id: str):
    """Get detailed information about a specific task."""
    try:
        task = get_task(task_id)
        return TaskInfo(
            task_id=task.task_id,
            name=task.name,
            description=task.description,
            difficulty=task.difficulty,
            max_steps=task.max_steps,
            reward_threshold=task.reward_threshold
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/grade", response_model=GradingResponse)
async def grade_trajectory():
    """
    Grade the current trajectory.
    
    Returns the score based on delivery completion, time efficiency,
    and fuel efficiency metrics.
    """
    global _trajectory_tracker
    
    if _trajectory_tracker is None:
        raise HTTPException(
            status_code=400,
            detail="No trajectory to grade. Complete episodes first."
        )
    
    trajectory = _trajectory_tracker.to_trajectory_dict()
    result = grade_task(trajectory, trajectory["task_id"])
    
    return GradingResponse(
        score=result.score,
        completion_score=result.completion_score,
        time_score=result.time_score,
        fuel_score=result.fuel_score,
        details=result.details
    )


@app.get("/trajectory")
async def get_trajectory():
    """Get the current trajectory data."""
    global _trajectory_tracker
    
    if _trajectory_tracker is None:
        raise HTTPException(
            status_code=400,
            detail="No trajectory recorded. Complete episodes first."
        )
    
    return _trajectory_tracker.to_trajectory_dict()


@app.get("/score-breakdown")
async def get_trajectory_breakdown():
    """Get detailed score breakdown for the current trajectory."""
    global _trajectory_tracker
    
    if _trajectory_tracker is None:
        raise HTTPException(
            status_code=400,
            detail="No trajectory to analyze. Complete episodes first."
        )
    
    trajectory = _trajectory_tracker.to_trajectory_dict()
    breakdown = get_score_breakdown(trajectory)
    
    return {
        "completion": round(breakdown["completion"], 4),
        "time": round(breakdown["time"], 4),
        "fuel": round(breakdown["fuel"], 4)
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
