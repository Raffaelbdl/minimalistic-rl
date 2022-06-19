from copy import deepcopy

import gym
import numpy as np


class VecEnv(gym.Env):
    def __init__(self, env, num_envs) -> None:
        self.num_envs = num_envs
        self.envs = [deepcopy(env) for _ in range(num_envs)]

    def reset(self, seed=None):
        return np.array([env.reset() for env in self.envs], dtype=np.float32)

    def step(self, action):
        actions = action
        if len(actions) != self.num_envs:
            raise ValueError("actions length should be the same as num_envs")
        states, rewards, dones, infos = tuple(
            zip(*[env.step(action) for (env, action) in zip(self.envs, actions)])
        )
        return (
            np.array(states, dtype=self.observation_type),
            np.array(rewards, dtype=np.float32),
            np.array(dones),
            infos,
        )

    @property
    def action_space(self):
        return self.envs[0].action_space

    @property
    def observation_space(self):
        return self.envs[0].observation_space

    @property
    def observation_type(self):
        dummy_observation = self.envs[0].reset()
        if isinstance(dummy_observation, np.ndarray):
            return dummy_observation.dtype
        else:
            return type(dummy_observation)

    def __getitem__(self, env_number):
        return self.envs[env_number]
