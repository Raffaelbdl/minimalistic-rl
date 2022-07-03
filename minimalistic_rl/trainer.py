"""Simple training loop"""
from collections import deque
import dataclasses
import functools
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
                    logs["episodic_return"] = logs["episodic_reward"][i]
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
        "last_episode_return": 0,
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
