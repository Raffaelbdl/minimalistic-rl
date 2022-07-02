"""Simple training loop"""
import dataclasses
import functools
from gc import callbacks
from typing import List, Optional, Union

import jax
import numpy as np
from tqdm import tqdm

from minimalistic_rl.agents import base
from minimalistic_rl.buffer import from_singles
from minimalistic_rl.callbacks import Callback, Logger, callback
from minimalistic_rl.wrapper import VecEnv


@dataclasses.dataclass
class TrainingConfig:
    """Configuration for training

    Attributes:
        seed (int)
        verbose (int)
        episode_cycle_length (int)
        render (bool)
    """

    seed: int = 0
    verbose: int = 0
    episode_cycle_length: int = 10
    render: bool = False


def _train(
    agent: base.Base,
    callbacks: Optional[List[Callback]] = None,
    training_config: TrainingConfig = Optional[None],
):

    training_config = (
        training_config if training_config is not None else TrainingConfig()
    )
    training_config.num_envs = agent.num_envs
    training_config.num_env_steps = agent.num_env_steps

    logs = init_logs(agent, env.num_envs)
    callbacks = init_callbacks(training_config, callbacks)

    if not isinstance(env, VecEnv):
        env = VecEnv(env, 1)

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
    logs["max_steps"] = (n_steps + 1) // env.num_envs
    for step in range(1, (n_steps + 1) // env.num_envs):

        if render:
            env[0].render()

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

        s = np.zeros(
            (env.num_envs,) + env.observation_space.shape, dtype=env.observation_type
        )
        for i, d in enumerate(done):
            if d:
                logs["ep_count"] += 1
                logs["last_ended"] = i
                logs["last_ep_reward"] = logs["ep_reward"][i]
                for c in callbacks:
                    c.at_episode_end(logs)

                rng = jrng.split(rng)[0]
                s[i] = env[i].reset(seed=int(rng[0]))
                logs["ep_reward"][i] = 0.0
            else:
                logs["ep_reward"][i] += r[i]
                s[i] = s_next[i]

        for c in callbacks:
            c.at_step_end(logs)

    for c in callbacks:
        c.at_train_end(logs)

    return logs


def train(
    agent: base.Base,
    training_config: TrainingConfig = None,
    callbacks: List[Callback] = None,
):
    training_config = (
        training_config if training_config is not None else TrainingConfig()
    )
    key = jax.random.PRNGKey(training_config.seed)
    envs = agent.envs

    training_config.num_env_steps = agent.num_env_steps
    training_config.num_envs = agent.num_envs
    num_steps = agent.num_env_steps // agent.num_envs
    training_config.num_steps = num_steps

    callbacks = init_callbacks(training_config, callbacks)
    logs = init_logs(agent)

    observations = envs.reset()
    for step in range(num_steps):
        global_step = int(step * agent.num_envs)
        logs["step_count"] = global_step

        key = jax.random.split(key)[0]

        actions, logps = agent.select_action(key, observations)
        next_observations, rewards, dones, infos = envs.step(actions)
        logs["step_reward"] = rewards
        logs["episodic_reward"] += rewards

        agent.buffer.add(
            from_singles(
                observations, actions, rewards, dones, next_observations, logps
            )
        )

        if agent.policy_type == "on":
            if on_policy_improve_condition(step, agent):
                logs = agent.improve(logs)
        elif agent.policy_type == "off":
            if off_policy_improve_condition(step, agent, 1, 1):
                raise NotImplementedError()

        for i, d in enumerate(dones):
            if d:
                for c in callbacks:
                    logs["episode_count"] += 1
                    logs["last_ended"] = i
                    c.at_episode_end(logs)
                logs["episodic_reward"][i] = 0.0

        observations = next_observations

        for c in callbacks:
            c.at_step_end(logs)

    return logs


def off_policy_improve_condition(
    step: int, agent: base.Base, improve_cycle: int, batch_size: int
) -> bool:
    raise NotImplementedError()
    # return step % improve_cycle == 0 and len(agent.buffer) > batch_size


def on_policy_improve_condition(step: int, agent: base.OnPolicy) -> bool:
    return len(agent.buffer) >= agent.agent_config.num_buffer_steps


def init_logs(agent: base.Base) -> dict:
    logs = {
        "agent_name": agent.agent_name,
        "policy_type": agent.policy_type,
        "global_step": 0,
        "episode_count": 0,
        "last_ended": 0,
        "step_count": 0,
        "step_reward": np.zeros((agent.num_envs,), dtype=np.float32),
        "episodic_reward": np.zeros((agent.num_envs,), dtype=np.float32),
        "num_updates": 0,
    }

    if agent.agent_name == "ppo":
        logs.update(
            {
                "total_loss": 0.0,
                "actor_loss": 0.0,
                "critic_loss": 0.0,
                "entropy": 0.0,
                "approx_kl": 0.0,
            }
        )

    return logs


def init_callbacks(
    callbacks_config: dict, callbacks: Optional[List[Callback]] = None
) -> List[Callback]:
    callbacks = callbacks if callbacks else []
    callbacks.append(Logger(callbacks_config))

    return callbacks
