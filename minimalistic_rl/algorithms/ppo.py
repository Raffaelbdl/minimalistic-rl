"""Simple PPO implementation in JAX"""
import functools
from ssl import OP_NO_TLSv1
from typing import Callable, Tuple

import chex
import gym
import haiku as hk
import jax
from jax import numpy as jnp, random as jrng
import numpy as np
import optax

from minimalistic_rl import Buffer
from minimalistic_rl.algorithms import Base
from minimalistic_rl.utils.updater import apply_updates

Array = chex.Array
ArrayNumpy = chex.ArrayNumpy 
PRNGKey = chex.PRNGKey
Scalar = chex.Scalar


class PPO(Base):
    """Most basic PPO variant"""
    policy = "on"

    def __init__(
        self, 
        config: dict,
        rng: PRNGKey,
        env: gym.Env,
        actor_transformed: hk.Transformed,
        critic_transformed: hk.Transformed
    ) -> None:
        super().__init__(
            config=config,
            rng=rng
        )
        self.rng, rng1, rng2 = jrng.split(self.rng, 3)

        dummy_s = env.observation_space.sample()
        dummy_S = jnp.expand_dims(dummy_s, axis=0)
        self.num_actions = env.action_space.n
        
        self.actor_transformed = actor_transformed
        actor_params = self.actor_transformed.init(
            rng1,
            dummy_S
        )

        self.critic_transformed = critic_transformed
        critic_params = self.critic_transformed.init(
            rng2,
            dummy_S
        )

        self.params = hk.data_structures.merge(
            actor_params, 
            critic_params
        )

        learning_rate = self.config["learning_rate"]
        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(self.params)

        self.actor_apply = jax.jit(self.actor_transformed.apply)
        self.critic_apply = jax.jit(self.critic_transformed.apply)
    
    def act(
        self,
        rng: PRNGKey,
        s: ArrayNumpy
    ) -> Tuple[Scalar, Scalar]: 
        """Performs an action in the environment

        Args:
            rng (PRNGKey)
            s (ArrayNumpy) [s]
        
        Returns:
            a (Scalar): action to perform in the env
            logp (Scalar): logp corresponding to action taken
        """
        params = self.params
        S = jnp.expand_dims(s, axis=0)

        logit = jnp.squeeze(self.actor_apply(params, S), axis=0)
        prob = jax.nn.softmax(logit)

        a = jrng.choice(
            rng,
            jnp.arange(0, self.num_actions),
            p=prob
        )
        logp = jnp.log(prob[a])

        return int(a), logp
    
    def improve(self):
        """Performs a training loop"""
        batch_size = self.config["batch_size"]
        S, A, R, Done, S_next, Logp = self.buffer.sample_all()

        n_batch = len(S) // batch_size

        gamma = self.config["gamma"]
        Target = compute_Target(
            self.params,
            self.critic_apply,
            gamma,
            R,
            Done, 
            S_next
        )
        Adv = compute_Adv(
            self.params,
            self.critic_apply,
            S
        )

        n_train_steps = self.config["n_train_steps"]
        for _ in range(n_train_steps):
            for i in range(n_batch):
                idx = np.array(
                    range(
                        i * batch_size,
                        (i + 1) * batch_size
                    )
                )
                _S = S[idx]
                _A = A[idx]
                _Logp = Logp[idx]
                _Adv = Adv[idx]
                _Target = Target[idx]
            
                epsilon = self.config["epsilon"]
                loss, grads = jax.value_and_grad(
                    ppo_loss
                )(
                    self.params,
                    self.actor_apply,
                    self.critic_apply,
                    epsilon,
                    _S,
                    _A, 
                    _Logp,
                    _Adv,
                    _Target   
                )
                self.params, self.opt_state = apply_updates(
                self.optimizer,
                self.params,
                self.opt_state,
                grads 
            )
        
        self.buffer.clear()

@functools.partial(jax.jit, static_argnums=(1, 2))
def compute_Target(
    params: hk.Params,
    critic_apply: Callable,
    gamma: float,
    R: Array,
    Done: Array, 
    S_next: Array
) -> Array:
    """Computes the 1-step boostrapped value"""

    V_next = critic_apply(params, S_next)
    nDone = jnp.where(Done, 0., 1.)

    Target = R
    Target += gamma * nDone * V_next

    return Target

@functools.partial(jax.jit, static_argnums=(1))
def compute_Adv(
    params: hk.Params,
    critic_apply: Callable,
    S: Array
) -> Array:
    """Computes the advantage, value approximation"""

    return critic_apply(params, S)

def actor_loss(
    params: hk.Params,
    actor_apply: Callable,
    epsilon: float,
    S: Array,
    A: Array,
    Logp: Array,
    Adv: Array,
) -> Scalar:

    new_Logit = actor_apply(params, S)
    new_Prob = jax.nn.softmax(new_Logit)
    new_Prob_a = jnp.take_along_axis(
        new_Prob,
        A,
        axis=-1
    )
    new_Logp = jnp.log(new_Prob_a)

    Ratio = jnp.exp(new_Logp - Logp)

    T1 = Ratio * Adv
    T2 = jnp.clip(
        Ratio,
        1 - epsilon,
        1 + epsilon
    ) * Adv
    
    T = jnp.minimum(T1, T2)

    return - jnp.mean(T)

def critic_loss(
    params: hk.Params,
    critic_apply: Callable,
    S: Array,
    Target: Array
) -> Scalar:

    V = critic_apply(params, S)
    Loss = jnp.square(V - Target)

    return jnp.mean(Loss)

def ppo_loss(
    params: hk.Params,
    actor_apply: Callable,
    critic_apply: Callable,
    epsilon: float,
    S: Array,
    A: Array, 
    Logp: Array,
    Adv: Array,
    Target: Array
) -> Scalar:

    a_loss = actor_loss(
        params,
        actor_apply,
        epsilon,
        S,
        A,
        Logp,
        Adv
    )
    
    c_loss = critic_loss(
        params, 
        critic_apply,
        S,
        Target
    )
    
    return a_loss + c_loss
        
        