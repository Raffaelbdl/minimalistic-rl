from collections import deque
from dataclasses import field
import random
from typing import Optional

import chex
import jax
import numpy as np

Array = chex.Array
ArrayNumpy = chex.ArrayNumpy
Numeric = chex.Numeric


@chex.dataclass
class TransitionBatch:
    S: Array
    A: Array
    R: Array
    Done: Array
    S_next: Array
    Logp: Array = field(default=0.0)
    Adv: Array = field(default=0.0)
    Return: Array = field(default=0.0)


def from_singles(
    s, a, r, done, s_next, logp=None, adv=None, ret=None
) -> TransitionBatch:
    logp = logp if logp is not None else np.zeros_like(r)
    adv = adv if adv is not None else np.zeros_like(r)
    ret = ret if ret is not None else np.zeros_like(r)

    cls_kwargs = {}
    for k, x in zip(
        TransitionBatch.__annotations__.keys(), (s, a, r, done, s_next, logp, adv, ret)
    ):
        cls_kwargs[k] = x
    return TransitionBatch(**cls_kwargs)


class Buffer:
    def __init__(self, capacity, seed):
        self.capacity = capacity
        random.seed(seed)
        self.clear()

    def clear(self):
        self.storage = deque(maxlen=self.capacity)

    def add(self, transition: TransitionBatch):
        self.storage.extend([transition])

    def sample(self, batch_size: int) -> TransitionBatch:
        transitions = random.sample(self.storage, batch_size)
        return jax.tree_util.tree_map(lambda *leaves: np.stack(leaves), *transitions)

    def sample_all(self) -> TransitionBatch:
        return jax.tree_util.tree_map(lambda *leaves: np.stack(leaves), *self.storage)

    def __str__(self):
        return str(self.storage)

    def __len__(self):
        return len(self.storage)
