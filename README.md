# dobot-mujoco-env

A compact MuJoCo environment for a Dobot manipulator and a small training example.

What this repository contains
- `dobot_mujoco/` — the package with the MuJoCo scene and Gym-compatible environment implementations (look in `dobot_mujoco/env`).
- `train.py` — a minimal example that creates `DobotCubeStack-v0`, trains a `CrossQ` agent (from `sb3_contrib`), and saves the trained model.

How it works (brief)
- MuJoCo runs the physics simulation using the XML scene in `dobot_mujoco/env/assets`.
- The environment exposes a continuous action space (5 actions: 4 joint controls + suction pump) and a flat observation vector (joint states, end-effector pose and cube states).
- `train.py` uses Gymnasium to create the env, Stable Baselines3 (sb3_contrib) for the agent, and saves models under `models/`.

## Requirements

- Python 3.10 or higher (see `pyproject.toml`).
- MuJoCo (Python bindings), PyTorch, Stable-Baselines3, sb3-contrib, numpy, and related dependencies listed in `requirements.txt` or `pyproject.toml`.

## Quickstart — create a virtual environment

Run these commands in the project root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools
```

Option A — install with the pinned/declarative tool used by this project

This repository is configured to be used with `uv` (see `pyproject.toml` for `tool.uv` settings). To sync/install the environment using that configuration run:

```bash
# sync the project environment; choose the extra you want: cpu, cu124
uv sync --extra cpu
# Or with a CUDA compatible device
uv sync --extra cu124
```

Option B — install with pip

```bash
pip install -r requirements.txt
```

## Using the training script (`train.py`)

`train.py` is a minimal example showing how to create the environment and train a CrossQ agent.

Basic usage:

```bash
python train.py --n-timesteps 200000 --tensorboard-log logs/exp1 --progress-bar
```

Command-line options available (default shown in script):
- `--n-timesteps`: number of training timesteps (default 200000)
- `--log-interval`: how often to log (in episodes)
- `--tensorboard-log`: directory for tensorboard logs
- `--progress-bar`: show a progress bar during training

When the script starts it prints detected PyTorch devices (GPUs) and then creates the `DobotCubeStack-v0` environment and trains a `CrossQ` model from `sb3_contrib`.

Trained models are saved to `models/crossq_dobotcubestack` (see the end of `train.py`). Create the `models` directory if you plan to save runs.

## Project structure (important files)

- `dobot_mujoco/` — package containing the environment and assets
  - `dobot_mujoco/env/` — MuJoCo env implementations and XML assets
- `train.py` — small training example that uses Gymnasium + SB3
- `pyproject.toml` — packaging + `uv` configuration and optional extras (cpu/cu128/rocm)
- `requirements.txt` — pip-compatible list of dependencies

## License

See `LICENSE` in the repository root.
