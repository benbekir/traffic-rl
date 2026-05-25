from stable_baselines3 import DQN
from experiments.runner import train_model

train_model(
    DQN,
    checkpoint_path="dqn_agent",
    model_kwargs={
        "learning_rate": 1e-3,
        "buffer_size": 100000,
        "batch_size": 128,
        "exploration_fraction": 0.1,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
    },
)