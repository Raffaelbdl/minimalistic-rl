
from collections import deque, namedtuple
import random
from typing import Optional

import chex
import jax
import numpy as np

ArrayNumpy = chex.ArrayNumpy
Numeric = chex.Numeric

transition_namedtuple = namedtuple("TransitionBatch", 
                                    ["S", "A", "R", "Done", "S_next", "logP"])


class TransitionBatch():

    def __init__(self, S, A, R, Done, S_next, logP=0):

        self.S = S
        self.A = A
        self.R = R
        self.Done = Done
        self.S_next = S_next
        self.logP = logP
    
    def _to_named_tuple(self):
        return transition_namedtuple(self.S, self.A, self.R, self.Done, self.S_next, self.logP)
    
    @classmethod
    def _from_singles(cls, s, a, r, done, s_next, logp=Optional[None]):
        logp = logp if logp else 0
        
        cls_args = []
        for x in (s, a, r, done, s_next, logp):
            cls_args.append(as_batch(x))
        return cls(*cls_args)

    
class Buffer():

    def __init__(self, capacity):
        self.capacity = capacity
        self.clear()
    
    def clear(self):
        self.storage = deque(maxlen=self.capacity)
    
    def add(self, transition: TransitionBatch):
        self.storage.extend([transition._to_named_tuple()])
    
    def sample(self, batch_size):
        transitions = random.sample(self.storage, batch_size)
        return jax.tree_util.tree_map(lambda *leaves: np.stack(leaves), *transitions)
    
    def sample_all(self):
        return jax.tree_util.tree_map(lambda *leaves: np.stack(leaves), *self.storage)
    
    def __str__(self):
        return str(self.storage)

    def __len__(self):
        return len(self.storage)


def as_batch(x: Numeric) -> ArrayNumpy:
    if not isinstance(x, ArrayNumpy):
        x = np.array(x)
    if not len(x.shape) >= 1:
        x = np.expand_dims(x, axis=0)
    return x