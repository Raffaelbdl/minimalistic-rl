import functools

import chex
import haiku as hk
import jax
import optax

Grads = chex.ArrayTree


@functools.partial(jax.jit, static_argnums=(0))
def apply_updates(
    optimizer: optax.GradientTransformation,
    params: hk.Params,
    opt_state: optax.OptState,
    grads: Grads,
):
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state
