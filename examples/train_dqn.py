import haiku as hk
import gym
import jax
from jax import numpy as jnp, random as jrng

from minimalistic_rl import algorithms as algo
from minimalistic_rl.trainer import train


env = gym.make("CartPole-v1")


@hk.transform
def critic_fn(S, is_training: bool = False):
    h = hk.Linear(64)(S)
    h = jax.nn.relu(h)
    h = hk.Linear(64)(h)
    h = jax.nn.relu(h)
    return hk.Linear(env.action_space.n, w_init=jnp.zeros)(h)


agent = algo.DQN(rng=jrng.PRNGKey(0), env=env, critic_transformed=critic_fn)


train(jrng.PRNGKey(1), agent, env)
