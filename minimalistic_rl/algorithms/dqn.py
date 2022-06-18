"""Simple DQN implementation in JAX"""
import functools
from typing import Callable, Optional, Tuple

import chex
import gym
import haiku as hk
import jax
import jax.numpy as jnp
import jax.random as jrng
import optax
import rlax

from minimalistic_rl.algorithms import Base
from minimalistic_rl.buffer import TransitionBatch
from minimalistic_rl.updater import apply_updates

Array = chex.Array
ArrayNumpy = chex.ArrayNumpy
PRNGKey = chex.PRNGKey
Scalar = chex.Scalar


def make_DQN_config(user_config: dict):
    user_config = user_config if user_config is not None else {}
    config = {
        "algo": user_config.pop("algo", "dqn"),
        "policy": user_config.pop("policy", "off"),
        "discount": user_config.pop("discount", 0.99),
        "epsilon": user_config.pop("epsilon", 0.1),
        "capacity": user_config.pop("capacity", int(1e6)),
        "batch_size": user_config.pop("batch_size", 128),
        "learning_rate": user_config.pop("learning_rate", 2.5e-4),
        "improve_cycle": user_config.pop("improve_cycle", 1),
        "n_train_steps": user_config.pop("n_train_steps", 1),
        "n_steps": user_config.pop("n_steps", int(1e6)),
        "episode_cycle_len": user_config.pop("episode_cycle_len", 10),
        "verbose": user_config.pop("verbose", 2),
    }
    return config


class DQN(Base):
    """Simple DQN"""

    def __init__(
        self,
        rng: PRNGKey,
        env: gym.Env,
        critic_transformed: hk.Transformed,
        config: Optional[dict] = None,
    ) -> None:
        config = make_DQN_config(config)
        super().__init__(config=config, rng=rng)

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

        self.rng, rng1 = jrng.split(self.rng, 2)

        n_train_steps = self.config["n_train_steps"]
        for _ in range(n_train_steps):

            Transition = self.buffer.sample(self.config["batch_size"])

            loss, grads = jax.value_and_grad(critic_loss)(
                self.params, rng1, self.critic_apply, self._discount, Transition
            )
            self.params, self.opt_state = apply_updates(
                self.optimizer, self.params, self.opt_state, grads
            )


@functools.partial(jax.jit, static_argnums=(2))
def critic_loss(
    params: hk.Params,
    rng: PRNGKey,
    critic_apply: Callable,
    discount: Scalar,
    Transition: TransitionBatch,
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
