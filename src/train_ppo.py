from stable_baselines3 import PPO
from experiments.runner import train_model

train_model(
    PPO,
    checkpoint_path="ppo_agent",
    episodes_per_curriculum=30,
    model_kwargs={
        "n_steps": 512,
        "batch_size": 256,
        "n_epochs": 10,
        "learning_rate": 1e-4,
        "ent_coef": 0.2,
        "clip_range": 0.3
    },
)