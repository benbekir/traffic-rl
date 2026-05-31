# 🚦 Adaptive Traffic Light Optimization using Deep Reinforcement Learning and SUMO

Deep Reinforcement Learning for adaptive traffic signal control in a realistic urban network. This project trains and evaluates **DQN** and **PPO** agents on a real-world road network from **Jönköping, Sweden**, simulated in [SUMO (Simulation of Urban Mobility)](https://eclipse.dev/sumo/), and compares them against a fixed-time baseline controller.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Baseline](#baseline)
  - [Analysis & Visualization](#analysis--visualization)
- [Architecture](#architecture)
  - [Environment](#environment)
  - [Multi-Agent Wrapper](#multi-agent-wrapper)
  - [Gridlock Monitor](#gridlock-monitor)
  - [Curriculum Training](#curriculum-training)
- [Hyperparameters](#hyperparameters)
- [Pre-trained Models](#pre-trained-models)
- [Report](#report)
- [License](#license)

---

## Overview

Urban traffic congestion is one of the most impactful challenges in modern cities. Traditional fixed-time traffic signals fail to adapt to dynamic traffic conditions, leading to unnecessary delays and congestion.

This project addresses the problem by formulating traffic signal control as a **multi-agent reinforcement learning** task:

- Each signalized intersection is controlled by an independent RL agent.
- Agents observe local traffic state (queue lengths, densities, phases) and learn to select signal phases that minimize total queue length.
- A **curriculum learning** strategy progressively increases traffic demand during training to improve robustness.
- A **gridlock detection monitor** terminates episodes early when the network becomes deadlocked, applying a large negative penalty to discourage gridlock-inducing policies.

---

## Project Structure

```
traffic-rl/
├── src/
│   ├── train_dqn.py              # Train a DQN agent
│   ├── train_ppo.py              # Train a PPO agent
│   ├── run_dqn.py                # Evaluate a trained DQN agent
│   ├── run_ppo.py                # Evaluate a trained PPO agent
│   ├── run_baseline.py           # Run fixed-time baseline controller
│   ├── envs/
│   │   ├── factory.py            # Environment factory (SUMO config)
│   │   ├── multi_agent_env.py    # Multi-agent → VecEnv adapter
│   │   └── gridlock_monitor.py   # Gridlock detection wrapper
│   ├── experiments/
│   │   └── runner.py             # Training & evaluation orchestration
│   └── analysis/
│       └── visualize.py          # Multi-run comparison dashboard
├── simulations/
│   ├── networks/
│   │   ├── jkpg/                 # Jönköping road network files
│   │   └── custom/               # Custom test network
│   └── recordings/               # Simulation output (trip info CSVs)
├── report/
│   ├── DRL Traffic Light Control.tex   # LaTeX research report
│   ├── DRL Traffic Light Control.pdf   # Compiled report
│   └── resources/                      # Figures used in the report
├── dqn_agent.zip                 # Pre-trained DQN model
├── ppo_agent.zip                 # Pre-trained PPO model (50 episodes)
├── ppo_agent_100_eps.zip         # Pre-trained PPO model (100 episodes)
├── requirements.txt
└── README.md
```

---

## Prerequisites

- **Python** 3.9+
- **SUMO** (Simulation of Urban Mobility) — must be installed and the `SUMO_HOME` environment variable must be set.
  - Download from the [official SUMO website](https://eclipse.dev/sumo/).
  - Verify installation:
    ```bash
    sumo --version
    echo $SUMO_HOME
    ```

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/benbekir/traffic-rl.git
   cd traffic-rl
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

> **Note:** All scripts should be run from the `src/` directory so that relative paths to simulation files resolve correctly.

```bash
cd src
```

### Training

Train a **DQN** agent with curriculum learning:

```bash
python train_dqn.py
```

Train a **PPO** agent with curriculum learning:

```bash
python train_ppo.py
```

Training uses a 3-stage curriculum (75% → 87.5% → 100% traffic scale) with 50 episodes per stage. Checkpoints are saved after each curriculum stage and a final model is saved upon completion.

### Evaluation

Run a trained **DQN** agent for 10 evaluation episodes:

```bash
python run_dqn.py
```

Run a trained **PPO** agent for 10 evaluation episodes:

```bash
python run_ppo.py
```

Evaluation runs produce trip-info CSV files in `simulations/recordings/` for downstream analysis.

### Baseline

Run a **fixed-time baseline** controller (10-second phase cycles, 10 iterations):

```bash
python run_baseline.py
```

### Analysis & Visualization

Generate a multi-run comparison dashboard across all models:

```bash
python analysis/visualize.py
```

This produces a 4-panel figure comparing:
- **Completed trips over time** (with 95% confidence intervals)
- **Time loss distribution** (kernel density estimation)
- **Waiting time vs. stop count** (per-model scatter with error bars)
- **Average speed vs. route length** (regression trends)

The output is saved to `simulations/recordings/multi_run_model_comparison.png`.

---

## Architecture

### Environment

The simulation environment is built on top of the [`sumo-rl`](https://github.com/LucasAlegre/sumo-rl) library using its PettingZoo parallel API. Key configuration (defined in `src/envs/factory.py`):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `delta_time` | 5 s | Simulation step interval for agent actions |
| `num_seconds` | 8,400 s | Total simulation duration (2 hours 20 min) |
| `reward_fn` | `"queue"` | Negative sum of queue lengths at controlled intersections |
| Network | Jönköping, Sweden | Real-world urban road network |

### Multi-Agent Wrapper

The `MultiAgentEnv` class (`src/envs/multi_agent_env.py`) adapts the PettingZoo multi-agent environment into a Stable Baselines3-compatible `VecEnv`:

- **Observation padding:** Heterogeneous observation spaces are padded to a uniform size so all agents share a single policy network.
- **Action masking:** Actions exceeding an agent's action space are wrapped via modulo to remain valid.
- **Shared policy:** All intersection agents use a single `MlpPolicy`, enabling parameter sharing across the network.

### Gridlock Monitor

The `GridlockMonitor` class (`src/envs/gridlock_monitor.py`) wraps the multi-agent environment to detect network-wide gridlocks:

- Monitors the total number of halting vehicles across all edges at each step.
- If halting vehicles exceed a threshold (default: 800), the episode is terminated early with a large negative reward (−100,000) to strongly penalize gridlock-inducing policies.

### Curriculum Training

Training follows a **curriculum learning** strategy (`src/experiments/runner.py`):

1. **Stage 1:** 75% traffic demand scale — agents learn basic signal coordination
2. **Stage 2:** 87.5% traffic demand — intermediate complexity
3. **Stage 3:** 100% traffic demand — full real-world conditions

Each stage trains for 50 episodes. Model weights carry over between stages for continuous improvement.

---

## Hyperparameters

### DQN

| Parameter | Value |
|-----------|-------|
| Learning Rate | 1 × 10⁻³ |
| Replay Buffer Size | 500,000 |
| Batch Size | 128 |
| Exploration Fraction | 0.1 |
| Initial ε | 1.0 |
| Final ε | 0.05 |
| Policy | MlpPolicy |

### PPO

| Parameter | Value |
|-----------|-------|
| Learning Rate | 1 × 10⁻⁴ |
| Rollout Steps (`n_steps`) | 512 |
| Batch Size | 256 |
| Epochs per Update | 10 |
| Entropy Coefficient | 0.2 |
| Clip Range | 0.3 |
| Policy | MlpPolicy |

---

## Pre-trained Models

The repository includes pre-trained model checkpoints:

| File | Description |
|------|-------------|
| `dqn_agent.zip` | DQN agent trained with curriculum learning |
| `ppo_agent.zip` | PPO agent trained for 50 episodes per curriculum stage |
| `ppo_agent_100_eps.zip` | PPO agent trained for 100 episodes per curriculum stage |

Load and evaluate any model:

```python
from stable_baselines3 import DQN  # or PPO
from experiments.runner import run_loaded_model

run_loaded_model(DQN, checkpoint_path="dqn_agent.zip", recording_name="my_eval.xml")
```

---

## Report

A detailed research report is available in the `report/` directory:

- **[DRL Traffic Light Control.pdf](report/DRL%20Traffic%20Light%20Control.pdf)** — Full IEEE-formatted paper covering the methodology, experimental setup, results, and analysis.

---

## License

This project is for academic and research purposes.
