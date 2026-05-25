from stable_baselines3 import PPO
from experiments.runner import train_model

train_model(
    PPO,
    checkpoint_path="ppo_agent",
    total_training_episodes=100, # final run should be about 300
    model_kwargs={
        "n_steps": 512,
        "batch_size": 256,
        "n_epochs": 10,
        "learning_rate": 3e-4,
        "ent_coef": 0.05,
    },
)