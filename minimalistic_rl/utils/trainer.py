"""Simple training loop"""
import functools

import chex
import gym
from jax import random as jrng

from minimalistic_rl import algorithms as algo
from minimalistic_rl.buffer import from_singles


Array = chex.Array
ArrayNumpy = chex.ArrayNumpy
PRNGKey = chex.PRNGKey
Scalar = chex.Scalar


def train(
    config: dict,
    rng: PRNGKey,
    agent: algo.Base,
    env: gym.Env,
):

    n_steps = config["n_steps"]
    if agent.policy == "off":
        improve_condition = functools.partial(
            off_policy_improve_condition,
            improve_cycle=config["improve_cycle"],
            batch_size=config["batch_size"],
        )
    elif agent.policy == "on":
        improve_condition = functools.partial(
            on_policy_improve_condition, T=config["T"]
        )

    ep_count = 0
    ep_r = 0.0

    s = env.reset()
    for step in range(n_steps):
        rng, rng1 = jrng.split(rng, 2)

        a, logp = agent.act(rng=rng1, s=s)
        s_next, r, done, _ = env.step(action=a)

        transition = from_singles(s, a, r, done, s_next, logp)
        agent.buffer.add(transition=transition)

        if improve_condition(step=step, agent=agent):
            agent.improve()

        if done:
            ep_count += 1
            print(f"{ep_count} -> {ep_r}")
            s = env.reset()
            ep_r = 0.0
        else:
            ep_r += r
            s = s_next


def off_policy_improve_condition(
    step: int, agent: algo.Base, improve_cycle: int, batch_size: int
) -> bool:
    return step % improve_cycle == 0 and len(agent.buffer) > batch_size


def on_policy_improve_condition(step: int, agent: algo.Base, T: int) -> bool:
    return len(agent.buffer) >= T
