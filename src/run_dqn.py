from stable_baselines3 import DQN
from utils.config import make_env

def run_dqn_agent() -> None:
    """Runs a full simulation episode using a deep Q-network agent.
    """

    env = make_env(recording_name="jkpg_dqn.xml")
    env.allow_reset = False

    model = DQN.load("dqn_agent.zip", env=env)

    obs = env.reset()

    try:
        while True:
            actions, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(actions)
            
            if any(dones):
                break
                
    except Exception as e:
        print(f"An error occurred during evaluation execution: {e}")
    finally:
        env.close()
        
if __name__ == "__main__":
    run_dqn_agent()
