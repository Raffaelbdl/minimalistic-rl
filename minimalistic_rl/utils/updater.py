import functools

import jax
from jax import numpy as jnp
import optax


@functools.partial(jax.jit, static_argnums=(0))
def apply_updates(optimizer, params, opt_state, grads):
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state
