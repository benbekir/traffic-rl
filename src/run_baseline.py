import numpy as np
from utils.config import make_env, DEFAULT_DELTA_TIME

def run_fixed_time_baseline(phase_duration_seconds: int) -> None:
    """Runs a full simulation episode using a simple fixed-time policy
    where traffic phases cycle at a set interval.
    """

    recording_name = f"jkpg_baseline_{phase_duration_seconds}s.xml"
    env = make_env(recording_name=recording_name)
    env.allow_reset = False

    steps_per_phase_change = max(1, phase_duration_seconds // DEFAULT_DELTA_TIME)
    
    _ = env.reset()
    total_envs = env.num_envs
    steps = 0
    
    # phase tracker for all agents
    current_actions = np.zeros(total_envs, dtype=int)
    
    try:
        while True:
            if steps % steps_per_phase_change == 0 and steps > 0:
                # switch to next phase for all traffic lights
                current_actions = (current_actions + 1) % env.action_space.n

            obs, rewards, dones, infos = env.step(current_actions)
            steps += 1
            
            if any(dones):
                break
                
    except Exception as e:
        print(f"An error occurred during simulation execution: {e}")
    finally:
        env.close()
        
if __name__ == "__main__":
    run_fixed_time_baseline(phase_duration_seconds=10)