from abc import abstractmethod
from typing import Tuple, Union

import chex
import jax.random as jrng

from minimalistic_rl.buffer import Buffer

ArrayNumpy = chex.ArrayNumpy
PRNGKey = chex.PRNGKey
Scalar = chex.Scalar


class Base:
    """Base class for RL algorithms"""

    def __init__(self, config: dict, rng: PRNGKey) -> None:

        self.config = config
        self.policy = config["policy"]
        self.algo = config["algo"]

        self.rng, rng1 = jrng.split(rng, 2)

        capacity = config["T"] if self.policy == "on" else config["capacity"]
        self.buffer = Buffer(capacity, seed=int(rng1[0]))

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
