from pathlib import Path
from stable_baselines3 import PPO
from sumo_rl import SumoEnvironment

NETWORK = "jkpg"
NETWORK_DIR = Path("simulations/networks/jkpg")
RECORDINGS_DIR = Path("simulations/recordings")

# Set MODE to "baseline" for fixed-time control or "train" for PPO.
MODE = "baseline"


def make_env(recording_name: str, delta_time: int = 10) -> SumoEnvironment:
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    return SumoEnvironment(
        net_file=str(NETWORK_DIR / f"{NETWORK}.net.xml"),
        route_file=str(NETWORK_DIR / f"{NETWORK}.rou.xml"),
        single_agent=True,
        use_gui=False,  # GUI can be unstable on macOS setups.
        sumo_warnings=True,
        delta_time=delta_time,
        additional_sumo_cmd=f"--tripinfo-output {RECORDINGS_DIR / recording_name}",
    )


def run_fixed_time_baseline(total_sim_seconds: int = 3600) -> None:
    env = make_env(recording_name=f"{NETWORK}_baseline.xml", delta_time=10)
    obs, info = env.reset()
    n_actions = env.action_space.n
    n_steps = total_sim_seconds // 10

    for t in range(n_steps):
        # Cycle through all phases every decision step (10 seconds).
        action = int(t % n_actions)
        step_result = env.step(action)

        # Support both Gym (4-tuple) and Gymnasium (5-tuple) APIs.
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
        else:
            obs, reward, done, info = step_result
            terminated, truncated = bool(done), False

        if terminated or truncated:
            obs, info = env.reset()

    env.close()


def train_ppo(total_timesteps: int = 1000) -> None:
    env = make_env(recording_name=f"{NETWORK}_ppo.xml", delta_time=10)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    model.save("ppo_agent")
    env.close()


if __name__ == "__main__":
    if MODE == "baseline":
        run_fixed_time_baseline(total_sim_seconds=3600)
    elif MODE == "train":
        train_ppo(total_timesteps=1000)
    else:
        raise ValueError("MODE must be either 'baseline' or 'train'.")