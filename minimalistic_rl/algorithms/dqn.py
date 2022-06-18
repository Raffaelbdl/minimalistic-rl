"""Simple DQN implementation in JAX"""
import functools
import os
from typing import Callable, Optional, Tuple

import chex
import gym
import haiku as hk
import jax
from jax import numpy as jnp, random as jrng
import numpy as np
import optax
import rlax

from minimalistic_rl.algorithms import Base
from minimalistic_rl.updater import apply_updates
from minimalistic_rl.algorithms.configs import DQN_CONFIG

Array = chex.Array
ArrayNumpy = chex.ArrayNumpy
PRNGKey = chex.PRNGKey
Scalar = chex.Scalar


class DQN(Base):
    """Most basic DQN variant"""

    policy = "off"
    algo = "dqn"

    def __init__(
        self,
        rng: PRNGKey,
        env: gym.Env,
        critic_transformed: hk.Transformed,
        config: Optional[dict] = None,
    ) -> None:
        _config = DQN_CONFIG
        if config is not None:
            _config.update(config)
        super().__init__(config=_config, rng=rng)
        self.rng, rng1 = jrng.split(self.rng, 2)

        dummy_s = env.reset()
        dummy_S = jax.tree_map(lambda x: jnp.expand_dims(x, axis=0), dummy_s)

        self.critic_transformed = critic_transformed
        self.params = self.critic_transformed.init(rng1, dummy_S, True)

        learning_rate = self.config["learning_rate"]
        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(self.params)

        self.critic_apply = self.critic_transformed.apply

        self._discount = self.config["discount"]
        self._epsilon = self.config["epsilon"]

    def act(self, rng: PRNGKey, s: ArrayNumpy) -> Tuple[int, None]:
        """Performs an action in the environment"""

        rng1, rng2 = jrng.split(rng, 2)

        S = jax.tree_map(lambda x: jnp.expand_dims(x, axis=0), s)
        q = jax.jit(self.critic_apply)(self.params, rng1, S)[0]

        a = rlax.epsilon_greedy(self._epsilon).sample(rng2, q)

        return int(a), None

    def improve(self):
        """Performs n_train_steps training loops"""

        self.rng, rng1, rng2 = jrng.split(self.rng, 3)

        batch_size = self.config["batch_size"]

        Transition = self.buffer.sample(batch_size=batch_size)

        n_batch = len(Transition.R) // batch_size

        n_train_steps = self.config["n_train_steps"]
        for _ in range(n_train_steps):
            for i in range(n_batch):

                idx = np.array(range(i * batch_size, (i + 1) * batch_size))
                _Transition = jax.tree_map(
                    lambda leaf: jax.tree_map(lambda x: x[idx], leaf), Transition
                )

                loss, grads = jax.value_and_grad(critic_loss)(
                    self.params, rng1, self.critic_apply, self._discount, _Transition
                )
                self.params, self.opt_state = apply_updates(
                    self.optimizer, self.params, self.opt_state, grads
                )


@functools.partial(jax.jit, static_argnums=(2))
def critic_loss(
    params: hk.Params,
    rng: PRNGKey,
    critic_apply: Callable,
    discount: float,
    Transition,
) -> Scalar:
    """Computes the critic loss, DQN style"""

    S, A, R, Done, S_next, _ = Transition.to_tuple()

    Q = critic_apply(params, rng, S, True)
    Q_next = critic_apply(params, rng, S_next, True)
    Discount = discount * jnp.where(Done, 0.0, 1.0)

    TD_error = jax.vmap(
        lambda q_tm1, a_tm1, r_t, discount_t, q_t: rlax.q_learning(
            q_tm1, a_tm1, r_t, discount_t, q_t, stop_target_gradients=True
        )
    )(Q, A, R, Discount, Q_next)

    return jnp.mean(jnp.square(TD_error))
