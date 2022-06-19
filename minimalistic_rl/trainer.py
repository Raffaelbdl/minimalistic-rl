"""Simple training loop"""
import functools
from typing import List, Optional

import chex
import gym
import jax.random as jrng

from minimalistic_rl import algorithms as algo
from minimalistic_rl.buffer import from_singles
from minimalistic_rl.callbacks import Callback, Logger


Array = chex.Array
ArrayNumpy = chex.ArrayNumpy
PRNGKey = chex.PRNGKey
Scalar = chex.Scalar


def train(
    rng: PRNGKey,
    agent: algo.Base,
    env: gym.Env,
    callbacks: Optional[List[Callback]] = None,
    render: bool = False,
):
    config = agent.config
    logs = init_logs(agent)
    callbacks = init_callbacks(config, callbacks)

    for c in callbacks:
        c.at_train_start(logs)

    n_steps = logs["n_steps"]
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

    s = env.reset(seed=int(rng[0]))
    for step in range(1, n_steps + 1):
        if render:
            env.render()

        logs["step_count"] = step
        for c in callbacks:
            c.at_step_start(logs)

        rng, rng1 = jrng.split(rng, 2)

        a, logp = agent.act(rng=rng1, s=s)
        s_next, r, done, _ = env.step(action=a)
        logs["step_reward"] = r

        transition = from_singles(s, a, r, done, s_next, logp)
        agent.buffer.add(transition=transition)

        if improve_condition(step=step, agent=agent):
            logs = agent.improve(logs)

        if done:
            logs["ep_count"] += 1

            for c in callbacks:
                c.at_episode_end(logs)

            s = env.reset(seed=int(rng[0]))
            logs["ep_reward"] = 0.0
        else:
            logs["ep_reward"] += r
            s = s_next

        for c in callbacks:
            c.at_step_end(logs)

    for c in callbacks:
        c.at_train_end(logs)


def off_policy_improve_condition(
    step: int, agent: algo.Base, improve_cycle: int, batch_size: int
) -> bool:
    return step % improve_cycle == 0 and len(agent.buffer) > batch_size


def on_policy_improve_condition(step: int, agent: algo.Base, T: int) -> bool:
    return len(agent.buffer) >= T


def init_logs(agent: algo.Base) -> dict:
    logs = {
        "algo": agent.algo,
        "policy": agent.policy,
        "ep_count": 0,
        "ep_reward": 0.0,
        "total_loss": 0.0,
    }
    if agent.algo == "ppo":
        logs.update(
            {
                "actor_loss": 0.0,
                "critic_loss": 0.0,
                "entropy": 0.0,
            }
        )
    logs.update(agent.config)

    return logs


def init_callbacks(
    config: dict, callbacks: Optional[List[Callback]] = None
) -> List[Callback]:
    callbacks = callbacks if callbacks else []
    callbacks.append(Logger(config))

    return callbacks
