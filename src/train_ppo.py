from stable_baselines3 import PPO
from experiments.runner import train_model
from envs.factory import DEFAULT_DELTA_TIME, DEFAULT_NUM_SECONDS

updates_per_episode = 10
steps_per_episode = DEFAULT_NUM_SECONDS // DEFAULT_DELTA_TIME
n_steps = steps_per_episode // updates_per_episode

train_model(
    PPO,
    checkpoint_path="ppo_agent",
    total_training_episodes=25,
    model_kwargs={
        "n_steps": n_steps,
        "batch_size": 128,
        "learning_rate": 3e-4,
        "ent_coef": 0.01,
    },
)