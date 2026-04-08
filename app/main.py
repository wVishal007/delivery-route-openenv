from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from app.env import DeliveryRouteEnv
from app.tasks import TASK_REGISTRY, get_task, get_all_tasks
from app.grader import TrajectoryTracker, grade_task, get_score_breakdown


app = FastAPI(
    title="Smart Delivery Route Optimization",
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

env: Optional[DeliveryRouteEnv] = None
trajectory_tracker: Optional[TrajectoryTracker] = None
current_task_id: Optional[str] = None


class ResetRequest(BaseModel):
    task_id: str = Field(default="easy")
    seed: Optional[int] = Field(default=None)


class StepRequest(BaseModel):
    action: int = Field(..., ge=0)


class EnvResponse(BaseModel):
    state: Dict[str, Any]
    action_space_size: int


class StepResponse(BaseModel):
    state: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]
    action_space_size: int


@app.get("/")
def root():
    return {"status": "ok", "message": "Delivery Route Optimization API is running"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "environment_initialized": env is not None,
        "task_id": current_task_id
    }


@app.get("/reset", response_model=EnvResponse)
def reset_get(
    task_id: str = Query(default="easy", description="Task ID: easy, medium, or hard"),
    seed: Optional[int] = Query(default=None, description="Random seed")
):
    return _do_reset(task_id, seed)


@app.post("/reset", response_model=EnvResponse)
def reset_post(request: ResetRequest):
    return _do_reset(request.task_id, request.seed)


def _do_reset(task_id: str, seed: Optional[int]) -> EnvResponse:
    global env, trajectory_tracker, current_task_id
    
    try:
        task = get_task(task_id)
        config = task.config.copy()
        
        if seed is not None:
            config["seed"] = seed
        
        env = DeliveryRouteEnv(task_config=config)
        state = env.reset(task_id=task_id)
        
        trajectory_tracker = TrajectoryTracker()
        trajectory_tracker.finalize(task_id, config)
        current_task_id = task_id
        
        return EnvResponse(
            state=state.to_dict(),
            action_space_size=env.get_action_space()
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


@app.get("/state", response_model=EnvResponse)
def get_state():
    if env is None or env.get_state() is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    
    return EnvResponse(
        state=env.get_state().to_dict(),
        action_space_size=env.get_action_space()
    )


@app.post("/step", response_model=StepResponse)
def step_action(request: StepRequest):
    global env, trajectory_tracker
    
    if env is None or env.get_state() is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    
    try:
        action = request.action
        state, reward, done, info = env.step(action)
        
        if trajectory_tracker is not None:
            trajectory_tracker.record_step(
                state=state.to_dict(),
                action=action,
                reward=reward,
                info=info
            )
        
        return StepResponse(
            state=state.to_dict(),
            reward=reward,
            done=done,
            info=info,
            action_space_size=env.get_action_space()
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step failed: {str(e)}")


@app.get("/tasks")
def list_tasks():
    tasks = get_all_tasks()
    return {
        "tasks": [
            {
                "task_id": t.task_id,
                "name": t.name,
                "description": t.description,
                "difficulty": t.difficulty
            }
            for t in tasks
        ]
    }


@app.get("/grade")
def grade():
    global trajectory_tracker, current_task_id
    
    if trajectory_tracker is None or current_task_id is None:
        raise HTTPException(status_code=400, detail="No trajectory to grade. Complete episodes first.")
    
    trajectory = trajectory_tracker.to_trajectory_dict()
    result = grade_task(trajectory, current_task_id)
    
    return {
        "score": result.score,
        "completion_score": result.completion_score,
        "time_score": result.time_score,
        "fuel_score": result.fuel_score
    }


@app.get("/trajectory")
def get_trajectory():
    global trajectory_tracker
    
    if trajectory_tracker is None:
        raise HTTPException(status_code=400, detail="No trajectory recorded.")
    
    return trajectory_tracker.to_trajectory_dict()


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
