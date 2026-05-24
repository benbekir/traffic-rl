from stable_baselines3 import DQN
from experiments.runner import run_loaded_model

run_loaded_model(DQN, checkpoint_path="dqn_agent.zip", recording_name="jkpg_dqn.xml")
