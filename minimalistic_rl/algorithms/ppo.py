"""Simple PPO implementation in JAX"""
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
from minimalistic_rl.updater import apply_updates

Array = chex.Array
ArrayNumpy = chex.ArrayNumpy
PRNGKey = chex.PRNGKey
Scalar = chex.Scalar


class PPO(Base):
    """Most basic PPO variant"""

    policy = "on"
    algo = "ppo"

    def __init__(
        self,
        config: dict,
        rng: PRNGKey,
        env: gym.Env,
        actor_transformed: hk.Transformed,
        critic_transformed: hk.Transformed,
    ) -> None:
        super().__init__(config=config, rng=rng)
        self.rng, rng1, rng2 = jrng.split(self.rng, 3)

        dummy_s = env.reset()
        dummy_S = jax.tree_map(lambda x: jnp.expand_dims(x, axis=0), dummy_s)
        self.num_actions = env.action_space.n

        self.actor_transformed = actor_transformed
        actor_params = self.actor_transformed.init(rng1, dummy_S, True)
        self.critic_transformed = critic_transformed
        critic_params = self.critic_transformed.init(rng2, dummy_S, True)
        self.params = hk.data_structures.merge(actor_params, critic_params)

        learning_rate = self.config["learning_rate"]
        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(self.params)

        self.actor_apply = self.actor_transformed.apply
        self.critic_apply = self.critic_transformed.apply

    def act(self, rng: PRNGKey, s: ArrayNumpy) -> Tuple[int, Scalar]:
        """Performs an action in the environment"""

        rng1, rng2 = jrng.split(rng, 2)

        params = self.params

        S = jax.tree_map(lambda x: jnp.expand_dims(x, axis=0), s)
        logit = jnp.squeeze(jax.jit(self.actor_apply)(params, rng1, S), axis=0)
        prob = jax.nn.softmax(logit)

        a = jrng.choice(rng2, jnp.arange(0, self.num_actions), p=prob)
        logp = jnp.log(prob[a])

        return int(a), logp

    def improve(self):
        """Performs n_train_steps training loops"""

        self.rng, rng1, rng2 = jrng.split(self.rng, 3)

        batch_size = self.config["batch_size"]

        Transition = self.buffer.sample_all()
        S, A, R, Done, S_next, Logp = Transition.to_tuple()

        n_batch = len(R) // batch_size

        gamma = self.config["gamma"]
        Discount_R = compute_Discount_R(gamma, R, Done)
        Adv = compute_Adv(self.params, rng1, self.critic_apply, S, Discount_R)

        n_train_steps = self.config["n_train_steps"]
        for _ in range(n_train_steps):
            for i in range(n_batch):

                rng2, _rng2 = jrng.split(rng2, 2)

                idx = np.array(range(i * batch_size, (i + 1) * batch_size))
                _S = jax.tree_map(lambda x: x[idx], S)
                _A = A[idx]
                _Logp = Logp[idx]
                _Adv = Adv[idx]
                _Discount_R = Discount_R[idx]

                epsilon = self.config["epsilon"]
                loss, grads = jax.value_and_grad(ppo_loss)(
                    self.params,
                    _rng2,
                    self.actor_apply,
                    self.critic_apply,
                    epsilon,
                    _S,
                    _A,
                    _Logp,
                    _Adv,
                    _Discount_R,
                )
                self.params, self.opt_state = apply_updates(
                    self.optimizer, self.params, self.opt_state, grads
                )

        self.buffer.clear()


@functools.partial(jax.jit, static_argnums=(0))
def compute_Discount_R(gamma: float, R: Array, Done: Array) -> Array:
    """Computes the non boostrapped value"""

    def propagate_r(r_t1, rnd_t):
        r_t = (
            rnd_t[0] + gamma * rnd_t[1] * r_t1
        )  # backpropagates reward if env not done
        return r_t, r_t

    nDone = jnp.where(Done, 0.0, 1.0)
    RnD = jnp.flip(jnp.stack([R, nDone], axis=1), axis=0)
    _, Discount_R = jax.lax.scan(f=propagate_r, init=R[-1], xs=RnD)
    Discount_R = jnp.flip(Discount_R, axis=0)

    return Discount_R


@functools.partial(jax.jit, static_argnums=(2))
def compute_Adv(
    params: hk.Params, rng: PRNGKey, critic_apply: Callable, S: Array, Discount_R: Array
) -> Array:
    """Computes the advantage, td error"""

    return Discount_R - critic_apply(params, rng, S)


def actor_loss(
    params: hk.Params,
    rng: PRNGKey,
    actor_apply: Callable,
    epsilon: float,
    S: Array,
    A: Array,
    Logp: Array,
    Adv: Array,
) -> Scalar:

    new_Logit = actor_apply(params, rng, S, True)
    new_Prob = jax.nn.softmax(new_Logit)
    new_Prob_a = jnp.take_along_axis(new_Prob, A, axis=-1)
    new_Logp = jnp.log(new_Prob_a)

    Ratio = jnp.exp(new_Logp - Logp)

    T1 = Ratio * Adv
    T2 = jnp.clip(Ratio, 1 - epsilon, 1 + epsilon) * Adv

    T = jnp.minimum(T1, T2)

    return -jnp.mean(T)


def critic_loss(
    params: hk.Params, rng: PRNGKey, critic_apply: Callable, S: Array, Discount_R: Array
) -> Scalar:

    V = critic_apply(params, rng, S, True)
    Loss = jnp.square(V - Discount_R)

    return jnp.mean(Loss)


@functools.partial(jax.jit, static_argnums=(2, 3))
def ppo_loss(
    params: hk.Params,
    rng: PRNGKey,
    actor_apply: Callable,
    critic_apply: Callable,
    epsilon: float,
    S: Array,
    A: Array,
    Logp: Array,
    Adv: Array,
    Discount_R: Array,
) -> Scalar:

    rng1, rng2 = jrng.split(rng, 2)

    a_loss = actor_loss(params, rng1, actor_apply, epsilon, S, A, Logp, Adv)

    c_loss = critic_loss(params, rng2, critic_apply, S, Discount_R)

    return a_loss + c_loss
