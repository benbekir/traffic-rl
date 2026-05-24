from stable_baselines3 import DQN
from utils.config import make_env, DEFAULT_DELTA_TIME, DEFAULT_NUM_SECONDS

def train_dqn() -> None:
    env = make_env()
    
    steps_per_episode = DEFAULT_NUM_SECONDS // DEFAULT_DELTA_TIME
    
    total_agents = env.num_envs # represents the 23 traffic lights
    total_training_steps = steps_per_episode * total_agents * 10  # train for 10 episodes

    model = DQN(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=1e-3,
        buffer_size=100000,
        batch_size=128,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05
    )

    model.learn(total_timesteps=total_training_steps, progress_bar=True)
    model.save("dqn_agent")
    env.close()

if __name__ == "__main__":
    train_dqn()
