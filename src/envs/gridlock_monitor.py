import numpy as np
from envs.multi_agent_env import MultiAgentEnv

class GridlockMonitor(MultiAgentEnv):
    """
    Custom VecEnv wrapper that overrides step_wait to detect network gridlocks.
    If the number of halting vehicles breaches the maximum limit, it forces 
    the episode to end early and triggers an automatic environment reset.
    """
    def __init__(self, raw_env, max_halting_vehicles):
        super().__init__(raw_env)
        self.max_halting_vehicles = max_halting_vehicles

    def step_wait(self):
        obs, rews, dones, infos = super().step_wait()
        
        try:
            sumo_instance = self.env.unwrapped.sumo
            
            total_halting = sum(
                sumo_instance.edge.getLastStepHaltingNumber(edge_id)
                for edge_id in sumo_instance.edge.getIDList()
            )
            
            if super().allow_reset and total_halting >= self.max_halting_vehicles:
                dones = np.ones(self.num_envs, dtype=bool)
                rews = np.full(self.num_envs, -100000.0, dtype=np.float32)
                obs = self.reset()
                    
        except Exception as e:
            pass
            
        return obs, rews, dones, infos