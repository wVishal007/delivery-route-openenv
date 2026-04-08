"""
Inference Script - Smart Delivery Route Optimization Environment
==============================================================
MANDATORY:
- API_BASE_URL: https://router.huggingface.co/v1
- MODEL_NAME: Qwen/Qwen2.5-72B-Instruct
- HF_TOKEN: Your Hugging Face token

STDOUT FORMAT (STRICT):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import os
import asyncio
import httpx
from typing import Optional, List, Dict, Any

from openai import OpenAI


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "")

TASKS = ["easy", "medium", "hard"]
MAX_STEPS = 20
ENV_NAME = "delivery_route_optimization"


class DeliveryEnv:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=60.0)
        self.current_state: Optional[Dict[str, Any]] = None
        self.task_id: Optional[str] = None

    def reset(self, task_id: str) -> Dict[str, Any]:
        response = self.client.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id}
        )
        response.raise_for_status()
        data = response.json()
        self.current_state = data["state"]
        self.task_id = task_id
        return self.current_state

    def step(self, action: int) -> Dict[str, Any]:
        response = self.client.post(
            f"{self.base_url}/step",
            json={"action": action}
        )
        response.raise_for_status()
        data = response.json()
        self.current_state = data["state"]
        return {
            "state": data["state"],
            "reward": data["reward"],
            "done": data["done"],
            "error": None
        }

    def close(self):
        self.client.close()


class HeuristicAgent:
    """Fallback agent using nearest delivery heuristic with traffic consideration."""

    def choose_action(
        self,
        current_location: Dict[str, float],
        remaining_deliveries: List[Dict[str, Any]]
    ) -> int:
        if not remaining_deliveries:
            return 0

        min_score = float('inf')
        best_idx = 0

        for i, loc in enumerate(remaining_deliveries):
            dx = loc.get('x', 0) - current_location.get('x', 0)
            dy = loc.get('y', 0) - current_location.get('y', 0)
            distance = (dx * dx + dy * dy) ** 0.5
            traffic = loc.get('traffic', 1.0)
            score = distance * traffic

            if score < min_score:
                min_score = score
                best_idx = i

        return best_idx


class LLMAgent:
    """LLM agent with fallback to heuristic on failure."""

    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.fallback = HeuristicAgent()
        self.use_llm = bool(api_key and base_url)

    def get_action(
        self,
        current_location: Dict[str, float],
        remaining_deliveries: List[Dict[str, Any]],
        time_elapsed: float,
        fuel_used: float
    ) -> int:
        if not self.use_llm:
            return self.fallback.choose_action(current_location, remaining_deliveries)

        prompt = self._build_prompt(
            current_location, remaining_deliveries, time_elapsed, fuel_used
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a delivery route optimization AI. Return ONLY a single integer - the index of the next delivery location to visit. No explanation."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0.0,
                timeout=30.0
            )
            content = response.choices[0].message.content.strip()
            action = self._parse_action(content)
            return max(0, min(action, len(remaining_deliveries) - 1))
        except Exception:
            return self.fallback.choose_action(current_location, remaining_deliveries)

    def _build_prompt(
        self,
        current_location: Dict[str, float],
        remaining_deliveries: List[Dict[str, Any]],
        time_elapsed: float,
        fuel_used: float
    ) -> str:
        lines = [
            f"Current location: x={current_location.get('x', 0):.2f}, y={current_location.get('y', 0):.2f}",
            f"Time elapsed: {time_elapsed:.2f}",
            f"Fuel used: {fuel_used:.2f}",
            f"Remaining deliveries ({len(remaining_deliveries)}):"
        ]
        for i, loc in enumerate(remaining_deliveries):
            lines.append(f"  {i}: x={loc.get('x', 0):.2f}, y={loc.get('y', 0):.2f}, traffic={loc.get('traffic', 1.0):.2f}")
        lines.append("Return ONLY the integer index (0 to " + str(len(remaining_deliveries) - 1) + ").")
        return "\n".join(lines)

    def _parse_action(self, content: str) -> int:
        content = content.strip()
        for char in content:
            if char.isdigit():
                num_str = char
                for j in range(1, len(content)):
                    if content[j].isdigit():
                        num_str += content[j]
                    else:
                        break
                try:
                    return int(num_str)
                except ValueError:
                    pass
        return 0


def log_start(task_name: str, benchmark: str, model_name: str):
    print(f"[START] task={task_name} env={benchmark} model={model_name}", flush=True)


def log_step(step_num: int, action: int, reward: float, done: bool, error: Optional[str]):
    error_str = "null" if error is None else str(error).replace(",", ";")
    print(f"[STEP] step={step_num} action={action} reward={reward:.2f} done={str(done).lower()} error={error_str}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


async def run_task(env: DeliveryEnv, agent: LLMAgent, task_id: str) -> Dict[str, Any]:
    log_start(task_id, ENV_NAME, MODEL_NAME)

    rewards: List[float] = []
    step_num = 0

    try:
        state = env.reset(task_id)
    except Exception as e:
        log_end(False, 0, 0.0, [])
        return {"success": False, "steps": 0, "score": 0.0, "rewards": [], "error": str(e)}

    for step_num in range(1, MAX_STEPS + 1):
        try:
            remaining = state.get("remaining_deliveries", [])
            if not remaining:
                break

            current_loc = state.get("current_location", {"x": 0, "y": 0})
            time_elapsed = state.get("time_elapsed", 0.0)
            fuel_used = state.get("fuel_used", 0.0)

            action = agent.get_action(current_loc, remaining, time_elapsed, fuel_used)

            result = env.step(action)
            reward = result["reward"]
            done = result["done"]
            last_error = result.get("error")

            rewards.append(reward)
            log_step(step_num, action, reward, done, last_error)

            if done:
                break

            state = result["state"]

        except Exception as e:
            rewards.append(0.0)
            log_step(step_num, 0, 0.0, False, str(e))
            break

    total_reward = sum(rewards) if rewards else 0.0
    score = total_reward / len(rewards) if rewards else 0.0
    score = max(0.0, min(1.0, score))
    success = score >= 0.6

    log_end(success, step_num, score, rewards)

    return {
        "success": success,
        "steps": step_num,
        "score": score,
        "rewards": rewards
    }


async def main():
    agent = LLMAgent(
        api_key=HF_TOKEN,
        base_url=API_BASE_URL,
        model=MODEL_NAME
    )
    env = DeliveryEnv(base_url="http://localhost:8000")

    try:
        for task_id in TASKS:
            await run_task(env, agent, task_id)
    except Exception:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    asyncio.run(main())
