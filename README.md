# Smart Delivery Route Optimization Environment

A production-ready OpenEnv-compatible reinforcement learning environment for optimizing delivery routes in a logistics system. The agent learns to minimize total time, fuel consumption, and traffic delays while completing all deliveries.

## Why This is Real-World

This environment models **last-mile delivery optimization**, a critical challenge in modern logistics:

- **Scalable**: Handles dynamic constraints (traffic, fuel, time windows)
- **Real Constraints**: Models actual delivery challenges with realistic parameters
- **Production-Ready**: Designed for real logistics systems deployment
- **Evaluated Objectively**: Strict scoring system with automated evaluation

## Project Overview

This environment simulates realistic logistics challenges including:

- **Route Planning**: Navigate across multiple delivery points efficiently
- **Traffic Management**: Dynamic traffic conditions affecting travel times
- **Resource Constraints**: Fuel consumption limits and time windows
- **Multi-Difficulty Tasks**: From simple campus deliveries to complex metro distribution

## Environment Details

### State Space

The observation space includes:

| Component              | Type                  | Description                                    |
| ---------------------- | --------------------- | ---------------------------------------------- |
| `current_location`     | (x, y)                | Current vehicle position coordinates           |
| `remaining_deliveries` | List[(x, y, traffic)] | Undelivered locations with traffic multipliers |
| `time_elapsed`         | float                 | Total time spent since departure               |
| `fuel_used`            | float                 | Total fuel consumed                            |
| `progress`             | float [0,1]           | Fraction of deliveries completed               |
| `action_space_size`    | int                   | Number of available actions                    |

### Action Space

Discrete action space where action `i` means "travel to the i-th remaining delivery location".

The action space size is dynamic and equals the number of remaining undelivered locations.

### Reward Function

The reward is bounded between 0.0 and 1.0 for each step, designed to balance:

| Component        | Value          | Description                              |
| ---------------- | -------------- | ---------------------------------------- |
| Delivery Reward  | +0.15          | Base reward for completing each delivery |
| Efficiency Bonus | [-0.05, +0.10] | Based on travel time vs expected         |
| Fuel Penalty     | [0, -0.10]     | Proportional to fuel consumption         |
| Traffic Penalty  | [0, -0.05]     | For high traffic conditions              |
| Completion Bonus | [0, +0.45]     | When all deliveries complete             |

### Tasks

#### Easy: Campus Delivery

- 3 delivery locations on a small 50x50 map
- No traffic delays
- 200 time units, 100 fuel units limit

#### Medium: City Logistics

- 6 delivery locations on a 100x100 city grid
- Moderate static traffic congestion
- 300 time units, 150 fuel units limit

#### Hard: Metro Distribution

- 12 delivery locations across 200x200 zones
- Dynamic traffic that changes during route
- 400 time units, 180 fuel units limit

## Installation

### Local Development

```bash
# Clone the repository
git clone https://github.com/your-org/delivery-route-env.git
cd delivery-route-env

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Docker

```bash
# Build the image
docker build -t delivery-route-env .

# Run the container
docker run -p 8000:8000 delivery-route-env
```

## Running the API

### Start the Server

```bash
# Using uvicorn directly
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Or using Python module
python -m app.main
```

### API Endpoints

| Endpoint      | Method | Description                   |
| ------------- | ------ | ----------------------------- |
| `/`           | GET    | Health check and service info |
| `/health`     | GET    | Detailed health status        |
| `/reset`      | POST   | Reset environment with task   |
| `/state`      | GET    | Get current environment state |
| `/step`       | POST   | Execute an action             |
| `/tasks`      | GET    | List all available tasks      |
| `/tasks/{id}` | GET    | Get task details              |
| `/grade`      | GET    | Grade current trajectory      |
| `/trajectory` | GET    | Get trajectory data           |

### Example Usage

```bash
# Reset environment for easy task (POST)
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy", "seed": 42}'

# Reset environment for easy task (GET)
curl "http://localhost:8000/reset?task_id=easy&seed=42"

# Take a step
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": 0}'

# Get current state
curl http://localhost:8000/state

# Get final grade
curl http://localhost:8000/grade
```

## Running Inference

The inference script runs an LLM-powered agent on all tasks:

```bash
# Set environment variables
export API_BASE_URL=http://localhost:8000
export MODEL_NAME=gpt-4o
export HF_TOKEN=your_token_here

# Run inference
python inference.py
```

### Logging Format

The inference script logs in STRICT format for automated evaluation:

```
[START] task=easy env=delivery_route_optimization model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=0 reward=0.14 done=false error=null
[STEP] step=2 action=1 reward=0.13 done=false error=null
[STEP] step=3 action=0 reward=0.18 done=true error=null
[END] success=true steps=3 score=0.85 rewards=0.14,0.13,0.18
```

Format specifications (STRICT - no deviations):

- `[START]` - One per task: `task=<name> env=<benchmark> model=<model>`
- `[STEP]` - One per action: `step=<n> action=<idx> reward=<0.00> done=<true|false> error=<msg|null>`
- `[END]` - One per task: `success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>`

## OpenEnv Compliance

This environment follows OpenEnv specification strictly:

| Method    | Description                            |
| --------- | -------------------------------------- |
| `reset()` | Reset environment to initial state     |
| `step()`  | Execute an action and return new state |
| `state()` | Get current environment state          |

The `openenv.yaml` file validates:

- API endpoints (`/reset`, `/step`, `/state`)
- State space definition
- Action space definition
- Task configurations
- Reward function bounds [0, 1]
- Performance constraints

## Grading System

The grader evaluates performance based on three metrics:

| Metric          | Weight (Easy) | Weight (Medium) | Weight (Hard) |
| --------------- | ------------- | --------------- | ------------- |
| Completion      | 50%           | 40%             | 35%           |
| Time Efficiency | 30%           | 35%             | 40%           |
| Fuel Efficiency | 20%           | 25%             | 25%           |

Additional penalty of 0.15 for constraint violations in hard mode.

## Project Structure

```
delivery-route-env/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI application
│   ├── env.py           # Environment logic
│   ├── tasks.py         # Task definitions
│   └── grader.py        # Grading system
├── openenv.yaml         # OpenEnv specification
├── inference.py         # LLM inference script
├── gradio_app.py        # Gradio interface
├── Dockerfile
├── requirements.txt
├── hf_space.json        # Hugging Face Spaces config
└── README.md
```

## Deployment

### Hugging Face Spaces

The project is ready for Hugging Face Spaces deployment:

1. Create a new Space at huggingface.co/spaces
2. Select "Docker" as the SDK
3. Upload all project files
4. The app will automatically start

### Docker Compose (Production)

```yaml
version: "3.8"
services:
  delivery-env:
    build: .
    ports:
      - "8000:8000"
    environment:
      - API_BASE_URL=http://localhost:8000
    resources:
      cpus: "2"
      memory: 8G
```

## Performance Constraints

- **Total Runtime**: < 20 minutes for full evaluation
- **Hardware**: 2 vCPU, 8GB RAM
- **Max Steps per Episode**: 40

## API Reference

### POST /reset

Reset the environment to initial state.

**Request Body:**

```json
{
  "task_id": "easy", // Task identifier (easy, medium, hard)
  "seed": 42 // Optional random seed
}
```

**Response:**

```json
{
  "state": { ... },
  "observation": { ... },
  "action_space_size": 3
}
```

### POST /step

Execute an action in the environment.

**Request Body:**

```json
{
  "action": 0
}
```

**Response:**

```json
{
  "state": { ... },
  "observation": { ... },
  "reward": 0.142,
  "done": false,
  "info": {
    "distance_traveled": 12.5,
    "time_taken": 14.2,
    "fuel_consumed": 1.25,
    "deliveries_completed": 1,
    "deliveries_remaining": 2
  },
  "action_space_size": 2
}
```

## License

MIT License

## Contributing

Contributions are welcome! Please read the contribution guidelines and submit pull requests.
