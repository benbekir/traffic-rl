from stable_baselines3 import DQN
from experiments.runner import test_model

test_model(DQN, checkpoint_path="dqn_agent.zip", recording_name="jkpg_dqn.xml")
