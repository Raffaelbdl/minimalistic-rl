from copy import deepcopy

import gym
import envpool
import numpy as np


class VecEnv:
    def __init__(
        self,
        environment: gym.Env,
        num_envs: int,
        use_envpool: bool = False,
        env_type: str = None,
    ) -> None:
        self.num_envs = num_envs
        self.use_envpool = use_envpool

        if use_envpool:
            if not isinstance(environment, str):
                raise ValueError(
                    "When using envpool, environment should be the task id"
                )
            if env_type is None:
                raise ValueError("When using envpool, an env type should be provided")
            self.envs = envpool.make(environment, env_type, num_envs=num_envs)
        else:
            if isinstance(environment, str):
                self.envs = [gym.make(environment) for _ in range(num_envs)]
            else:
                self.envs = [deepcopy(environment) for _ in range(num_envs)]

    def reset(self, seed=None):
        if self.use_envpool:
            return self.envs.reset()
        else:
            return np.array([env.reset() for env in self.envs], dtype=np.float32)

    def step(self, actions):
        if len(actions) != self.num_envs:
            raise ValueError("actions length should be the same as num_envs")

        if self.use_envpool:
            return self.envs.step(actions)

        else:
            next_observations, rewards, dones, infos = [], [], [], []
            for i in range(self.num_envs):
                next_observation, reward, done, info = self.envs[i].step(actions[i])
                if done:
                    next_observation = self.envs[i].reset()
                next_observations.append(next_observation)
                rewards.append(reward)
                dones.append(done)
                infos.append(info)

            return (
                np.array(next_observations, dtype=self.observation_type),
                np.array(rewards, dtype=np.float32),
                np.array(dones),
                infos,
            )

    @property
    def action_space(self):
        if self.use_envpool:
            return self.envs.action_space
        else:
            return self.envs[0].action_space

    @property
    def observation_space(self):
        if self.use_envpool:
            return self.envs.observation_space
        else:
            return self.envs[0].observation_space

    @property
    def observation_type(self):
        dummy_observation = self.observation_space.sample()
        if isinstance(dummy_observation, np.ndarray):
            return dummy_observation.dtype
        else:
            return type(dummy_observation)

    def __getitem__(self, env_number):
        return self.envs[env_number]
