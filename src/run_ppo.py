from stable_baselines3 import PPO
from experiments.runner import test_model

test_model(PPO, checkpoint_path="ppo_agent.zip", recording_name="jkpg_ppo.xml")