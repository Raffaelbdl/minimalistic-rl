from abc import abstractmethod
import dataclasses
from typing import Optional, Union

import gym
import jax

from minimalistic_rl.buffer import Buffer
from minimalistic_rl.types import Action, State
from minimalistic_rl.wrapper import VecEnv


@dataclasses.dataclass
class BaseConfig:
    """Basic configuration for RL agents

    Attributes:
        seed (int)

        num_env_steps (int): number of steps in the environment
    """

    seed: int = 0
    num_env_steps: int = int(1e7)


@dataclasses.dataclass
class OnPolicyConfig(BaseConfig):
    """Basic configuration for on policy RL agents

    Attributes:
        seed (int)

        num_env_steps (int): number of steps in the environment
    """

    num_buffer_steps: int = 128


class Base:
    """Base class for RL agents"""

    def __init__(
        self,
        agent_name: str,
        policy_type: str,
        environment: Union[gym.Env, VecEnv],
        agent_config: Optional[BaseConfig] = None,
    ) -> None:

        self.agent_name = agent_name
        self.policy_type = policy_type

        if not isinstance(environment, VecEnv):
            environment = VecEnv(environment, 1)

        self.envs = environment
        self.num_envs = environment.num_envs
        self.num_env_steps = agent_config.num_env_steps
        self.key = jax.random.PRNGKey(agent_config.seed)

        self.buffer: Buffer = None

    @abstractmethod
    def select_action(
        self,
        rng: jax.random.PRNGKey,
        s: State,
    ) -> Action:
        """Perform an action in the environment"""
        raise NotImplementedError("Act method must be implemented")

    @abstractmethod
    def improve(self, logs: dict) -> dict:
        """Perform a training step"""
        raise NotImplementedError("Improve method must be implemented")


class OnPolicy(Base):
    """Base class for on policy agents"""

    def __init__(
        self,
        agent_name: str,
        environment: Union[gym.Env, VecEnv],
        on_policy_config: Optional[OnPolicyConfig] = None,
    ) -> None:
        on_policy_config = (
            on_policy_config if on_policy_config is not None else OnPolicyConfig()
        )
        super().__init__(agent_name, "on", environment, on_policy_config)
        self.agent_config = on_policy_config
