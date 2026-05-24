from stable_baselines3 import PPO
from experiments.runner import run_loaded_model

run_loaded_model(PPO, checkpoint_path="ppo_agent.zip", recording_name="jkpg_ppo.xml")