from abc import abstractmethod
from typing import Tuple, Union

import chex

from minimalistic_rl.buffer import Buffer

ArrayNumpy = chex.ArrayNumpy
PRNGKey = chex.PRNGKey
Scalar = chex.Scalar


class Base:
    """Base class for RL algorithms"""

    policy: str = "none"  # "off", "on"

    def __init__(self, config: dict, rng: PRNGKey, **kwargs) -> None:

        self.config = config

        self.rng = rng

        capacity = config["capacity"]
        self.buffer = Buffer(capacity)

    @abstractmethod
    def act(
        self,
        rng: PRNGKey,
        s: ArrayNumpy,
    ) -> Tuple[Scalar, Union[Scalar, None]]:
        """Performs an action in the environment"""
        raise NotImplementedError("Act method must be implemented")

    @abstractmethod
    def improve(self) -> None:
        """Performs a single training step"""
        raise NotImplementedError("Improve method must be implemented")
