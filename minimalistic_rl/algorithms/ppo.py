"""Simple PPo implementation in JAX"""
from collections import deque
import functools
from typing import Callable, Optional, Tuple

import chex
import gym
import haiku as hk
import jax
import jax.nn as nn
import jax.numpy as jnp
import jax.random as jrng
import optax
import numpy as np
import rlax

from minimalistic_rl.algorithms import Base
from minimalistic_rl.buffer import TransitionBatch
from minimalistic_rl.updater import apply_updates

Array = chex.Array
ArrayNumpy = chex.ArrayNumpy
PRNGKey = chex.PRNGKey
Scalar = chex.Scalar


def make_PPO_config(user_config: dict):
    user_config = user_config if user_config is not None else {}
    config = {
        "algo": user_config.pop("algo", "ppo"),
        "policy": user_config.pop("policy", "on"),
        "discount": user_config.pop("discount", 0.99),
        "epsilon": user_config.pop("epsilon", 0.1),
        "lambda": user_config.pop("lambda", 0.95),
        "critic_coef": user_config.pop("critic_coef", 0.5),
        "entropy_coef": user_config.pop("entropy_coef", 0.01),
        "T": user_config.pop("T", 512),
        "batch_size": user_config.pop("batch_size", 128),
        "learning_rate": user_config.pop("learning_rate", 2.5e-4),
        "improve_cycle": user_config.pop("improve_cycle", 1),
        "n_train_steps": user_config.pop("n_train_steps", 1),
        "n_steps": user_config.pop("n_steps", int(1e6)),
        "episode_cycle_len": user_config.pop("episode_cycle_len", 10),
        "verbose": user_config.pop("verbose", 2),
    }
    return config


class PPO(Base):
    """Simple PPO"""

    def __init__(
        self,
        rng: PRNGKey,
        env: gym.Env,
        actor_transformed: hk.Transformed,
        critic_transformed: hk.Transformed,
        config: Optional[dict] = None,
    ) -> None:
        config = make_PPO_config(config)
        super().__init__(config=config, rng=rng)

        self.rng, rng1, rng2 = jrng.split(rng, 3)

        dummy_s = env.reset()
        dummy_S = jax.tree_map(lambda x: jnp.expand_dims(x, axis=0), dummy_s)

        self.actor_params = actor_transformed.init(rng1, dummy_S)
        self.critic_params = critic_transformed.init(rng2, dummy_S)
        self.params = hk.data_structures.merge(self.actor_params, self.critic_params)

        learning_rate = self.config["learning_rate"]
        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(self.params)

        self.actor_apply = actor_transformed.apply
        self.critic_apply = critic_transformed.apply

        self._discount = self.config["discount"]
        self._epsilon = self.config["epsilon"]
        self._lambda = self.config["lambda"]

        self._critic_coef = self.config["critic_coef"]
        self._entropy_coef = self.config["entropy_coef"]

    def act(self, rng: PRNGKey, s: ArrayNumpy) -> Tuple[int, float]:
        """Performs an action in the environment"""

        rng1, rng2 = jrng.split(rng, 2)

        S = jax.tree_map(lambda x: jnp.expand_dims(x, axis=0), s)
        logits = jax.jit(self.actor_apply)(self.params, rng1, S)[0]
        probs = nn.softmax(logits, axis=-1)

        a = rlax.categorical_sample(rng2, probs)

        return int(a), jnp.log(probs[a])

    def improve(self, logs: dict):
        """Performs n_train_steps training loops"""

        self.rng, rng1 = jrng.split(self.rng, 2)

        Transition = self.buffer.sample_all()

        n_batch = self.config["T"] // self.config["batch_size"]
        idx = jnp.arange(self.config["T"])

        n_train_steps = self.config["n_train_steps"]
        mean_loss = deque()
        mean_actor_loss = deque()
        mean_critic_loss = deque()
        mean_entropy = deque()
        for _ in range(n_train_steps):
            for i in range(n_batch):

                rng1, _rng1 = jrng.split(rng1, 2)

                _idx = idx[
                    i * self.config["batch_size"] : (i + 1) * self.config["batch_size"]
                ]
                _Transition = jax.tree_map(
                    lambda leaf: jax.tree_map(lambda x: x[_idx], leaf), Transition
                )
                (loss, (actor_loss, critic_loss, entropy)), grads = jax.value_and_grad(
                    ppo_loss, has_aux=True
                )(
                    self.params,
                    _rng1,
                    self.actor_apply,
                    self.critic_apply,
                    self._discount,
                    self._epsilon,
                    self._lambda,
                    self._critic_coef,
                    self._entropy_coef,
                    _Transition,
                )
                self.params, self.opt_state = apply_updates(
                    self.optimizer, self.params, self.opt_state, grads
                )
                mean_loss.append(np.array(loss))
                mean_actor_loss.append(np.array(actor_loss))
                mean_critic_loss.append(np.array(critic_loss))
                mean_entropy.append(np.array(entropy))

        logs.update(
            {
                "total_loss": sum(mean_loss) / len(mean_loss),
                "actor_loss": sum(mean_actor_loss) / len(mean_actor_loss),
                "critic_loss": sum(mean_critic_loss) / len(mean_critic_loss),
                "entropy": sum(mean_entropy) / len(mean_entropy),
                "params": self.params,
            }
        )

        self.buffer.clear()

        return logs


@functools.partial(jax.jit, static_argnums=(2, 3))
def ppo_loss(
    params: hk.Params,
    rng: PRNGKey,
    actor_apply: Callable,
    critic_apply: Callable,
    discount: Scalar,
    epsilon: Scalar,
    lambda_: Scalar,
    critic_coef: Scalar,
    entropy_coef: Scalar,
    Transition: TransitionBatch,
) -> Scalar:
    """Computes the loss, PPO style"""

    S, A, R, Done, S_next, Logp = Transition.to_tuple()
    Discount = discount * jnp.where(Done, 0.0, 1.0)

    new_Logits = actor_apply(params, rng, S)  # (128, 2)
    V = critic_apply(params, rng, S)[..., 0]  # (128, )
    V_last = critic_apply(params, rng, S_next[-1:])[..., 0]

    new_Probs = jnp.take_along_axis(
        nn.softmax(new_Logits, axis=-1), A[..., jnp.newaxis], axis=-1
    )[..., 0]
    new_Logp = jnp.log(new_Probs + 1e-6)

    Ratio = jnp.exp(new_Logp - Logp)
    Adv = rlax.truncated_generalized_advantage_estimation(
        R, Discount, lambda_, jnp.concatenate([V, V_last], axis=0), True
    )
    actor_loss = rlax.clipped_surrogate_pg_loss(Ratio, Adv, epsilon)

    TD_error = rlax.td_lambda(V, R, Discount, V, lambda_)
    critic_loss = jnp.mean(jnp.square(TD_error))

    entropy_loss = rlax.entropy_loss(new_Logits, jnp.ones_like(new_Probs))

    return (
        actor_loss + critic_coef * critic_loss + entropy_coef * entropy_loss,
        (
            actor_loss,
            critic_loss,
            -entropy_loss,
        ),
    )
