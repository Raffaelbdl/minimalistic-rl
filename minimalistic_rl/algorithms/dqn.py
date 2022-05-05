"""Simple DQN implementation in JAX"""
import functools
from typing import Callable, Tuple

import chex
import gym
import haiku as hk
import jax
from jax import numpy as jnp, random as jrng
import numpy as np
import optax

from minimalistic_rl.algorithms import Base
from minimalistic_rl.buffer import to_tuple
from minimalistic_rl.utils.updater import apply_updates

Array = chex.Array
ArrayNumpy = chex.ArrayNumpy
PRNGKey = chex.PRNGKey
Scalar = chex.Scalar


class DQN(Base):
    """Most basic DQN variant"""

    policy = "off"

    def __init__(
        self,
        config: dict,
        rng: PRNGKey,
        env: gym.Env,
        critic_transformed: hk.Transformed,
    ) -> None:
        super().__init__(config=config, rng=rng)
        self.rng, rng1 = jrng.split(self.rng, 2)

        dummy_s = env.observation_space.sample()
        dummy_S = jnp.expand_dims(dummy_s, axis=0)

        self.critic_transformed = critic_transformed
        self.params = self.critic_transformed.init(rng1, dummy_S)

        learning_rate = self.config["learning_rate"]
        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(self.params)

        self.critic_apply = jax.jit(self.critic_transformed.apply)

    def act(self, rng: PRNGKey, s: ArrayNumpy) -> Tuple[int, None]:
        """Performs an action in the environment"""
        rng1, rng2 = jrng.split(rng, 2)

        epsilon = self.config["epsilon"]
        params = self.params

        S = jnp.expand_dims(s, axis=0)
        Q = self.critic_apply(params, S)

        a_greedy = jnp.argmax(Q, axis=-1)
        a_random = jrng.choice(key=rng1, a=jnp.arange(Q.shape[-1]))

        if jrng.uniform(rng2) > epsilon:
            a = a_greedy
        else:
            a = a_random

        return int(a), None

    def improve(self):
        """Performs n_train_steps training loops"""
        batch_size = self.config["batch_size"]

        Transition = self.buffer.sample(batch_size=batch_size)
        S, A, R, Done, S_next, _ = to_tuple(Transition)

        n_batch = len(S) // batch_size

        gamma = self.config["gamma"]
        Target = compute_Target(self.params, self.critic_apply, gamma, R, Done, S_next)

        n_train_steps = self.config["n_train_steps"]
        for _ in range(n_train_steps):
            for i in range(n_batch):
                idx = np.array(range(i * batch_size, (i + 1) * batch_size))
                _S = S[idx]
                _A = A[idx]

                loss, grads = jax.value_and_grad(critic_loss)(
                    self.params, self.critic_apply, Target, _S, _A
                )
                self.params, self.opt_state = apply_updates(
                    self.optimizer, self.params, self.opt_state, grads
                )


@functools.partial(jax.jit, static_argnums=(1, 2))
def compute_Target(
    params: hk.Params,
    critic_apply: Callable,
    gamma: float,
    R: Array,
    Done: Array,
    S_next: Array,
) -> Array:
    """Computes the 1-step boostrapped value, DQN style"""

    Q_next = critic_apply(params, S_next)
    nDone = jnp.where(Done, 0.0, 1.0)

    Target = R
    Target += gamma * nDone * jnp.max(Q_next, axis=-1)[..., None]

    return Target


@functools.partial(jax.jit, static_argnums=(1))
def critic_loss(
    params: hk.Params,
    critic_apply: Callable,
    Target: Array,
    S: Array,
    A: Array,
) -> Scalar:
    """Computes the critic loss, DQN style"""

    Q = critic_apply(params, S)
    Q_a = jnp.take_along_axis(Q, A, axis=-1)

    TD_error = jnp.square(Q_a - Target)

    return jnp.mean(TD_error)
