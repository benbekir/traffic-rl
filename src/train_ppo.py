from stable_baselines3 import PPO
from utils.config import make_env, DEFAULT_DELTA_TIME, DEFAULT_NUM_SECONDS

def train_ppo() -> None:
    env = make_env()
    
    steps_per_episode = DEFAULT_NUM_SECONDS // DEFAULT_DELTA_TIME
    n_steps = steps_per_episode // 10 # 10 updates per episode
    
    total_agents = env.num_envs # represents the 23 traffic lights
    total_training_steps = steps_per_episode * total_agents * 10  # train for 10 episodes

    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        n_steps=n_steps,
        batch_size=128,
        learning_rate=3e-4,
        ent_coef=0.01
    )

    model.learn(total_timesteps=total_training_steps, progress_bar=True)
    model.save("ppo_agent")
    env.close()

if __name__ == "__main__":
    train_ppo()