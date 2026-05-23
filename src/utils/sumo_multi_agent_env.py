import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

class SumoMultiAgentEnv(VecEnv):
    def __init__(self, raw_env, allow_reset: bool = True):
        self.env = raw_env
        self.agent_ids = list(self.env.possible_agents)
        num_envs = len(self.agent_ids)
        self.allow_reset = allow_reset

        # max observation size for uniform env
        self.max_obs_size = max(
            self.env.observation_space(agent).shape[0] for agent in self.agent_ids
        )

        # max action size for uniform env
        self.agent_action_spaces = {
            agent: self.env.action_space(agent) for agent in self.agent_ids
        }
        self.max_action_size = max(space.n for space in self.agent_action_spaces.values())
        
        obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_obs_size,), dtype=np.float32)
        action_space = gym.spaces.Discrete(self.max_action_size)

        super().__init__(num_envs, obs_space, action_space)

    def _pad_obs(self, obs_dict):
        """Pads observations to provide uniform shape for all traffic lights."""
        obs_list = []
        for agent in self.agent_ids:
            raw = np.array(obs_dict[agent], dtype=np.float32)
            padded = np.zeros(self.max_obs_size, dtype=np.float32)
            padded[: len(raw)] = raw
            obs_list.append(padded)
        return np.array(obs_list)

    def reset(self):
        obs_dict, _ = self.env.reset()
        return self._pad_obs(obs_dict)

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        action_dict = {}
        for agent_id, act in zip(self.agent_ids, self._actions):
            allowed_actions = self.agent_action_spaces[agent_id].n
            action_dict[agent_id] = act.item() % allowed_actions # TODO: check if act.item() > allowed_actions for any case

        obs_d, rew_d, term_d, trunc_d, info_d = self.env.step(action_dict)

        episode_is_done = all(term_d.values()) or all(trunc_d.values()) or not obs_d
        if episode_is_done:
            rews = np.array([rew_d.get(a, 0.0) for a in self.agent_ids], dtype=np.float32)
            dones = np.ones(self.num_envs, dtype=bool)
            infos = [info_d.get(a, {}) for a in self.agent_ids]
            if self.allow_reset:
                obs = self.reset()
            else:
                obs = np.zeros((self.num_envs, self.max_obs_size), dtype=np.float32)
        else:
            rews = np.array([rew_d.get(a, 0.0) for a in self.agent_ids], dtype=np.float32)
            dones = np.array([term_d.get(a, False) or trunc_d.get(a, False) for a in self.agent_ids], dtype=bool)
            infos = [info_d.get(a, {}) for a in self.agent_ids]
            obs = self._pad_obs(obs_d)

        return obs, rews, dones, infos

    def close(self):
        self.env.close()

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name)] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [None] * self.num_envs

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs