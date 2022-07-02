from collections import deque
from dataclasses import dataclass
import logging

import numpy as np
import tqdm

from minimalistic_rl.callbacks.callback import Callback

DEFAULT_FMT = "%(asctime)s | %(levelname)s | %(message)s"
STEP_FMT = "%(asctime)s | STEP    : %(message)s"
EPISODE_FMT = "%(asctime)s | EPISODE : %(message)s"


class Logger(Callback):
    """Logs information during training

    Verbose levels :
    - 0 : no logs
    - 1 : logs at each episode cycle
    - 2 : logs at each episode
    - 3 : logs at each step

    When working with vectorized envs, only the first one will
    be used for logging.
    """

    def __init__(self, callback_config: dataclass):

        self.logger = init_logger("minimalistic-rl/logger", DEFAULT_FMT, logging.INFO)
        self.verbose = callback_config.verbose

        self.num_env_steps = callback_config.num_env_steps
        self.num_steps = callback_config.num_steps
        self.num_envs = callback_config.num_envs

        if self.verbose == 0:
            self.step_bar = tqdm.tqdm(
                desc="Training ... ", total=(self.num_env_steps + 1)
            )

        if self.verbose == 1:
            self.rewards = deque(maxlen=callback_config.episode_cycle_length)

    def at_train_start(self, logs: dict):

        fmtter = logging.Formatter(DEFAULT_FMT)
        self.logger.handlers[0].setFormatter(fmtter)

        self.logger.info("Training is starting !")

        agent_name = logs["agent_name"]
        num_env_steps = log_handle_large_int(self.num_env_steps)
        self.logger.info(
            f"Algo is {agent_name}, training for {num_env_steps} env steps"
        )

    def at_train_end(self, logs: dict):

        fmtter = logging.Formatter(DEFAULT_FMT)
        self.logger.handlers[0].setFormatter(fmtter)

        self.logger.info("Training has finished !")

    def at_episode_end(self, logs: dict):

        fmtter = logging.Formatter(EPISODE_FMT)
        self.logger.handlers[0].setFormatter(fmtter)
        last_ended = logs["last_ended"] if "last_ended" in logs.keys() else 0
        ep_count = (
            logs["episode_count"][last_ended]
            if isinstance(logs["episode_count"], (list, np.ndarray))
            else logs["episode_count"]
        )
        ep_reward = (
            logs["episodic_reward"][last_ended]
            if isinstance(logs["episodic_reward"], (list, np.ndarray))
            else logs["episodic_reward"]
        )
        total_loss = (
            logs["total_loss"][last_ended]
            if isinstance(logs["total_loss"], (list, np.ndarray))
            else logs["total_loss"]
        )

        if self.verbose == 1:
            self.rewards.append(ep_reward)
            if ep_count % self.rewards.maxlen == 0:
                mean_reward = sum(self.rewards) / len(self.rewards)
                msg = f"{ep_count} |"
                msg += f" mean_reward : {mean_reward:.1f} |"
                msg += f" loss : {total_loss:.2f} |"
                self.logger.info(msg)

        elif self.verbose == 2:
            msg = f"{ep_count} |"
            msg += f" reward : {ep_reward:.1f} |"
            msg += f" loss : {total_loss:.2f} |"
            self.logger.info(msg)

        fmtter = logging.Formatter(DEFAULT_FMT)
        self.logger.handlers[0].setFormatter(fmtter)

    def at_step_end(self, logs: dict):

        fmtter = logging.Formatter(STEP_FMT)
        self.logger.handlers[0].setFormatter(fmtter)

        if self.verbose == 0:
            self.step_bar.update(self.num_envs)

        if self.verbose == 3:

            step_count = (
                logs["step_count"][0]
                if isinstance(logs["step_count"], (list, np.ndarray))
                else logs["step_count"]
            )
            step_reward = (
                logs["step_reward"][0]
                if isinstance(logs["step_reward"], (list, np.ndarray))
                else logs["step_reward"]
            )
            self.logger.info(f"{step_count} | reward : {step_reward:.1f}")

        fmtter = logging.Formatter(DEFAULT_FMT)
        self.logger.handlers[0].setFormatter(fmtter)


def log_handle_large_int(x: int) -> str:
    if x >= 1e9:
        right_part = x / 1e9
        return f"{right_part:.1f}B"
    elif 1e9 > x >= 1e6:
        right_part = x / 1e6
        return f"{right_part:.1f}M"
    elif 1e6 > x >= 1e3:
        right_part = x / 1e3
        return f"{right_part:.1f}K"
    else:
        return str(x)


def init_logger(name: str, default_fmt: str, log_level: int) -> logging.Logger:

    logger = logging.Logger(name)
    logger.setLevel(log_level)

    formatter = logging.Formatter(default_fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)

    return logger
