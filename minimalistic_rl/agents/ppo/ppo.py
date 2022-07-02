"""All-In-One PP0 implementation in JAX"""
from collections import deque
import dataclasses
import functools
from typing import Callable, List, Optional, Tuple, Union, Mapping, NamedTuple

import dm_env
import gym
import haiku as hk
import jax
import jax.nn as nn
from jax import numpy as jnp
import jax.random as jrandom
import optax
import numpy as np
import rlax
from minimalistic_rl.agents.ppo.ppo_nets import make_mlp_nets

from minimalistic_rl.types import Action, State
from minimalistic_rl.agents.base import Base
from minimalistic_rl.buffer import TransitionBatch, from_singles, Buffer
from minimalistic_rl.updater import apply_updates
from minimalistic_rl.wrapper import VecEnv


@dataclasses.dataclass
class PPOConfig:
    """Configuration for PPO

    Attributes:
        seed (int)

        num_env_steps (int): number of steps in the environment
        num_buffer_steps (int): number of steps in buffer
        num_epochs (int): number of loops over the buffer
        num_minibatches (int): number of gradient steps per epoch

        ppo_clip_epsilon (float)
        critic_coef (float)
        entropy_coef (float)
        discount (float)
        gae_lambda (float)

        learning_rate (float)
        adam_epsilon (float)
        learning_rate_annealing (bool)
        max_gradient_norm (float)
    """

    seed: int = 0

    num_env_steps: int = int(1e7)
    num_buffer_steps: int = 512
    num_epochs: int = 3
    num_minibatches: int = 4

    ppo_clip_epsilon: float = 0.1
    critic_coef: float = 0.5
    entropy_coef: float = 0.01
    discount: float = 0.99
    gae_lambda: float = 0.95

    learning_rate: float = 2.5e-4
    adam_epsilon: float = 1e-5
    learning_rate_annealing: bool = True
    max_gradient_norm: float = 0.5


class PPO(Base):
    def __init__(
        self,
        environment: gym.Env,
        actor_transformed: hk.Transformed,
        critic_transformed: hk.Transformed,
        ppo_config: Optional[PPOConfig] = None,
    ) -> None:
        super().__init__("ppo", "on", environment)
        self.ppo_config = ppo_config if ppo_config is not None else PPOConfig()
        self.key = jax.random.PRNGKey(self.ppo_config.seed)

        self.init_buffer()
        self.init_networks(actor_transformed, critic_transformed)
        self.init_optimizer()

        self.actor_apply = actor_transformed.apply
        self.critic_apply = critic_transformed.apply

        self._ppo_clip_epsilon = self.ppo_config.ppo_clip_epsilon
        self._critic_coef = self.ppo_config.critic_coef
        self._entropy_coef = self.ppo_config.entropy_coef
        self._discount = self.ppo_config.discount
        self._gae_lambda = self.ppo_config.gae_lambda

        self.batch_size = (
            self.ppo_config.num_buffer_steps
            * self.num_envs
            // self.ppo_config.num_minibatches
        )

    def init_buffer(self):
        self.buffer = Buffer(
            capacity=self.ppo_config.num_buffer_steps, seed=self.ppo_config.seed
        )

    def init_networks(
        self, actor_transformed: hk.Transformed, critic_transformed: hk.Transformed
    ):
        self.actor_transformed = actor_transformed
        self.critic_transformed = critic_transformed

        key1, key2 = jax.random.split(self.key, 2)
        dummy_s = self.environment.observation_space.sample()
        dummy_S = jax.tree_map(lambda x: jnp.expand_dims(x, axis=0), dummy_s)
        self.actor_params = self.actor_transformed.init(key1, dummy_S)
        self.critic_params = self.critic_transformed.init(key2, dummy_S)
        self.params = hk.data_structures.merge(self.actor_params, self.critic_params)

    def init_optimizer(self):
        learning_rate = self.ppo_config.learning_rate
        if self.ppo_config.learning_rate_annealing:
            num_updates = (
                self.ppo_config.num_env_steps
                // self.ppo_config.num_buffer_steps
                // self.num_envs
                * self.ppo_config.num_epochs
                * self.ppo_config.num_minibatches
            )
            print(num_updates)
            learning_rate = optax.linear_schedule(learning_rate, 0.0, num_updates, 0)
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(self.ppo_config.max_gradient_norm),
            optax.adam(learning_rate, eps=self.ppo_config.adam_epsilon),
        )
        self.opt_state = self.optimizer.init(self.params)

    def init_metrics(self):
        self.total_loss = deque()
        self.actor_loss = deque()
        self.critic_loss = deque()
        self.entropy = deque()
        self.approx_kl = deque()

    def get_prepared_transition(self, key) -> TransitionBatch:
        Transition = self.buffer.sample_all()
        S, A, R, Done, S_next, Logp, Adv, Return = prepare_data(
            self.params,
            key,
            self.critic_apply,
            self._discount,
            self._gae_lambda,
            Transition,
        )
        Transition = from_singles(S, A, R, Done, S_next, Logp, Adv, Return)
        return Transition

    def select_action(
        self, key: jax.random.PRNGKey, s: State
    ) -> Action:  # TODO needs a wrapper if custom observations
        """Perform an action in the environment"""

        key1, key2 = jax.random.split(key, 2)
        if isinstance(self.environment, VecEnv):
            S = s
        else:
            S = jax.tree_map(lambda x: jnp.expand_dims(x, axis=0), s)

        Logits = jax.jit(self.actor_apply)(self.params, key1, S)
        Probs = nn.softmax(Logits, axis=-1)

        A = rlax.categorical_sample(key2, Probs)
        Probs_A = jnp.take_along_axis(Probs, A[..., jnp.newaxis], axis=-1)

        return np.array(A, dtype=np.int32), jnp.log(Probs_A)[..., 0]

    def improve(self, logs: dict):
        """Perform num_epochs training loops"""

        self.key, key1, key2, key3 = jax.random.split(self.key, 4)
        self.init_metrics()
        Transition = self.get_prepared_transition(key1)
        idx = jnp.arange(len(Transition.S))

        for epoch in range(self.ppo_config.num_epochs):
            key2 = jax.random.split(key2)[0]
            idx = jax.random.permutation(key2, idx, independent=True)

            for i in range(self.ppo_config.num_minibatches):
                logs["n_updates"] += 1
                key3 = jax.random.split(key3)[0]
                _idx = idx[i * self.batch_size : (i + 1) * self.batch_size]
                _Transition = jax.tree_map(lambda leaf: leaf[_idx], Transition)

                (loss, others), grads = jax.value_and_grad(ppo_loss, has_aux=True)(
                    self.params,
                    key3,
                    self.actor_apply,
                    self.critic_apply,
                    self._ppo_clip_epsilon,
                    self._critic_coef,
                    self._entropy_coef,
                    _Transition,
                )
                self.params, self.opt_state = apply_updates(
                    self.optimizer, self.params, self.opt_state, grads
                )
                self.total_loss.append(np.array(loss))
                self.actor_loss.append(np.array(others["actor_loss"]))
                self.critic_loss.append(np.array(others["critic_loss"]))
                self.entropy.append(np.array(others["entropy"]))
                self.approx_kl.append(np.array(others["approx_kl"]))

        logs.update(
            {
                "params": self.params,
                "total_loss": np.mean(self.total_loss),
                "actor_loss": np.mean(self.actor_loss),
                "critic_loss": np.mean(self.critic_loss),
                "entropy": np.mean(self.entropy),
                "approx_kl": np.mean(self.approx_kl),
            }
        )

        self.buffer.clear()

        return logs


@functools.partial(jax.jit, static_argnums=(2, 3, 8))
def ppo_loss(
    params: hk.Params,
    rng: jrandom.PRNGKey,
    actor_apply: Callable,
    critic_apply: Callable,
    epsilon,
    critic_coef,
    entropy_coef,
    Transition: TransitionBatch,
    normalize: bool = True,
):
    """Compute the PPO loss"""
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

    return actor_loss + critic_coef * critic_loss + entropy_coef * entropy_loss, {
        "actor_loss": actor_loss,
        "critic_loss": critic_loss,
        "entropy": -entropy_loss,
        "approx_kl": approx_kl,
    }


@functools.partial(jax.jit, static_argnums=(2))
def prepare_data(
    params: hk.Params,
    rng: jrandom.PRNGKey,
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
