"""Simple PPo implementation in JAX"""
from collections import deque
import functools
from typing import Callable, List, Optional, Tuple, Union

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
from minimalistic_rl.buffer import TransitionBatch, from_singles
from minimalistic_rl.updater import apply_updates
from minimalistic_rl.wrapper import VecEnv

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
        env: Union[gym.Env, VecEnv],
        actor_transformed: hk.Transformed,
        critic_transformed: hk.Transformed,
        config: Optional[dict] = None,
    ) -> None:
        config = make_PPO_config(config)
        super().__init__(config=config, rng=rng, env=env)

        self.rng, rng1, rng2 = jrng.split(rng, 3)

        dummy_S = self.env.reset()  # TODO needs a wrapper if custom observations
        # dummy_S = jax.tree_map(lambda x: jnp.expand_dims(x, axis=0), dummy_s)

        self.actor_params = actor_transformed.init(rng1, dummy_S)
        self.critic_params = critic_transformed.init(rng2, dummy_S)
        self.params = hk.data_structures.merge(self.actor_params, self.critic_params)

        learning_rate = self.config["learning_rate"]
        num_updates = (
            self.config["n_steps"]
            // self.config["batch_size"]
            * self.config["n_train_steps"]
        )
        learning_rate_schedule = optax.linear_schedule(
            learning_rate, 0.0, num_updates, 0
        )
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adam(learning_rate_schedule),
        )

        self.opt_state = self.optimizer.init(self.params)

        self.actor_apply = actor_transformed.apply
        self.critic_apply = critic_transformed.apply

        self._discount = self.config["discount"]
        self._epsilon = self.config["epsilon"]
        self._lambda = self.config["lambda"]

        self._critic_coef = self.config["critic_coef"]
        self._entropy_coef = self.config["entropy_coef"]

    def act(
        self, rng: PRNGKey, s: ArrayNumpy
    ) -> Tuple[int, float]:  # TODO needs a wrapper if custom observations
        """Performs an action in the environment"""
        S = s
        rng1, rng2 = jrng.split(rng, 2)
        # S = jax.tree_map(lambda x: jnp.expand_dims(x, axis=0), s)
        Logits = jax.jit(self.actor_apply)(self.params, rng1, S)
        Probs = nn.softmax(Logits, axis=-1)

        A = rlax.categorical_sample(rng2, Probs)
        Probs_A = jnp.take_along_axis(Probs, A[..., jnp.newaxis], axis=-1)
        return np.array(A, dtype=np.int32), jnp.log(Probs_A)[..., 0]

    def improve(self, logs: dict):
        """Performs n_train_steps training loops"""

        self.rng, rng1 = jrng.split(self.rng, 2)

        Transition = self.buffer.sample_all()
        S, A, R, Done, S_next, Logp, Adv, Return = prepare_data(
            self.params,
            rng1,
            self.critic_apply,
            self._discount,
            self._lambda,
            Transition,
        )
        Transition = from_singles(S, A, R, Done, S_next, Logp, Adv, Return)

        idx = jnp.arange(len(S))
        n_batch = max(len(idx) // self.config["batch_size"], 1)

        n_train_steps = self.config["n_train_steps"]
        mean_loss = deque()
        mean_actor_loss = deque()
        mean_critic_loss = deque()
        mean_entropy = deque()
        mean_approx_kl = deque()
        for _ in range(n_train_steps):
            rng1 = jrng.split(rng1)[0]
            idx = jrng.permutation(rng1, idx, independent=True)

            for i in range(n_batch):

                rng1, _rng1 = jrng.split(rng1, 2)

                _idx = idx[
                    i * self.config["batch_size"] : (i + 1) * self.config["batch_size"]
                ]
                _Transition = jax.tree_map(lambda leaf: leaf[_idx], Transition)

                (
                    loss,
                    (
                        actor_loss,
                        critic_loss,
                        entropy,
                        approx_kl,
                    ),
                ), grads = jax.value_and_grad(ppo_loss, has_aux=True)(
                    self.params,
                    _rng1,
                    self.actor_apply,
                    self.critic_apply,
                    self._epsilon,
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
                mean_approx_kl.append(np.array(approx_kl))

        logs.update(
            {
                "total_loss": sum(mean_loss) / len(mean_loss),
                "params": self.params,
                "actor_loss": sum(mean_actor_loss) / len(mean_actor_loss),
                "critic_loss": sum(mean_critic_loss) / len(mean_critic_loss),
                "entropy": sum(mean_entropy) / len(mean_entropy),
                "approx_kl": sum(mean_approx_kl) / len(mean_approx_kl),
            }
        )

        self.buffer.clear()

        return logs


@functools.partial(jax.jit, static_argnums=(2, 3, 8))
def ppo_loss(
    params: hk.Params,
    rng: PRNGKey,
    actor_apply: Callable,
    critic_apply: Callable,
    epsilon: Scalar,
    critic_coef: Scalar,
    entropy_coef: Scalar,
    Transition: TransitionBatch,
    normalize: bool = True,
) -> Scalar:
    """Computes the loss, PPO style"""
    S, A, _, _, _, Logp, Adv, Return = Transition.to_tuple()

    if normalize:
        Adv = (Adv - jnp.mean(Adv)) / (jnp.std(Adv) + 1e-8)

    new_Logits = actor_apply(params, rng, S)
    V = critic_apply(params, rng, S)[..., 0]

    new_Probs = jnp.take_along_axis(
        nn.softmax(new_Logits, axis=-1), A[..., jnp.newaxis], axis=-1
    )[..., 0]
    new_Logp = jnp.log(new_Probs + 1e-6)

    logRatio = new_Logp - Logp
    Ratio = jnp.exp(logRatio)

    approx_kl = jax.lax.stop_gradient(jnp.mean((Ratio - 1) - logRatio))

    actor_loss = rlax.clipped_surrogate_pg_loss(Ratio, Adv, epsilon)

    critic_loss = jnp.mean(jnp.square(Return - V))

    entropy_loss = rlax.entropy_loss(new_Logits, jnp.ones_like(new_Probs))

    return actor_loss + critic_coef * critic_loss + entropy_coef * entropy_loss, (
        actor_loss,
        critic_loss,
        -entropy_loss,
        approx_kl,
    )


@functools.partial(jax.jit, static_argnums=(2))
def prepare_data(
    params: hk.Params,
    rng: PRNGKey,
    critic_apply: Callable,
    discount: float,
    lambda_: float,
    Transition: TransitionBatch,
):

    S, A, R, Done, S_next, Logp, _, _ = Transition.to_tuple()
    S = jnp.swapaxes(S, 1, 0)
    A = jnp.swapaxes(A, 1, 0)
    R = jnp.swapaxes(R, 1, 0)
    Done = jnp.swapaxes(Done, 1, 0)
    S_next = jnp.swapaxes(S_next, 1, 0)
    Logp = jnp.swapaxes(Logp, 1, 0)
    Discount = discount * jnp.where(Done, 0.0, 1.0)

    V_ = jax.vmap(lambda s: critic_apply(params, rng, s))(S)[..., 0]
    V_last_ = jax.vmap(lambda s: critic_apply(params, rng, s))(S_next[:, -1:])[..., 0]

    def get_gae(V, V_last, R, Discount):
        Adv = rlax.truncated_generalized_advantage_estimation(
            R, Discount, lambda_, jnp.concatenate([V, V_last], axis=0), True
        )
        return Adv

    Adv = jax.vmap(get_gae)(V_, V_last_, R, Discount)

    def get_return(V, R, Discount):
        Lambda_return = rlax.lambda_returns(R, Discount, V, lambda_, True)
        return Lambda_return

    Return = jax.vmap(get_return)(V_, R, Discount)

    S = jnp.reshape(S, (-1,) + S.shape[2:])
    A = jnp.reshape(A, (-1,) + A.shape[2:]).astype(jnp.int32)
    R = jnp.reshape(R, (-1,) + R.shape[2:])
    Done = jnp.reshape(Done, (-1,) + Done.shape[2:])
    S_next = jnp.reshape(S_next, (-1,) + S_next.shape[2:])
    Logp = jnp.reshape(Logp, (-1,) + Logp.shape[2:])
    Adv = jnp.reshape(Adv, (-1,) + Adv.shape[2:])
    Return = jnp.reshape(Return, (-1,) + Return.shape[2:])

    return S, A, R, Done, S_next, Logp, Adv, Return
