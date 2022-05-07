from collections import deque
import logging

from minimalistic_rl.callbacks.callback import Callback

DEFAULT_FMT = "%(asctime)s | %(levelname)s | %(message)s"
STEP_FMT = "%(asctime)s | STEP    | %(message)s"
EPISODE_FMT = "%(asctime)s | EPISODE | %(message)s"


class Logger(Callback):
    """Logs information during training

    Verbose levels :
    - 0 : no logs
    - 1 : logs at each episode cycle
    - 2 : logs at each episode
    - 3 : logs at each step
    """

    def __init__(self, config: dict):

        self.logger = init_logger("minimalistic-rl/logger", DEFAULT_FMT, logging.INFO)
        self.verbose = config["verbose"]

        if self.verbose == 1:
            self.rewards = deque(maxlen=config["episode_cycle_len"])

    def at_train_start(self, logs: dict):

        fmtter = logging.Formatter(DEFAULT_FMT)
        self.logger.handlers[0].setFormatter(fmtter)

        self.logger.info("Training is starting !")

        algo = logs["algo"]
        n_steps = log_handle_large_int(logs["n_steps"])
        self.logger.info(f"Algo is {algo}, training for {n_steps} steps")

    def at_train_end(self, logs: dict):

        fmtter = logging.Formatter(DEFAULT_FMT)
        self.logger.handlers[0].setFormatter(fmtter)

        self.logger.info("Training has finished !")

    def at_episode_end(self, logs: dict):

        fmtter = logging.Formatter(EPISODE_FMT)
        self.logger.handlers[0].setFormatter(fmtter)

        if self.verbose == 1:

            ep_count = logs["ep_count"]
            ep_reward = logs["ep_reward"]
            self.rewards.append(ep_reward)
            if ep_count % self.rewards.maxlen == 0:
                mean_reward = sum(self.rewards) / len(self.rewards)
                self.logger.info(
                    f"At episode {ep_count} > mean_reward is {mean_reward:.1f}"
                )

        elif self.verbose >= 2:

            ep_count = logs["ep_count"]
            ep_reward = logs["ep_reward"]
            self.logger.info(f"At episode {ep_count} > reward is {ep_reward:.1f}")

        fmtter = logging.Formatter(DEFAULT_FMT)
        self.logger.handlers[0].setFormatter(fmtter)

    def at_step_end(self, logs: dict):

        fmtter = logging.Formatter(STEP_FMT)
        self.logger.handlers[0].setFormatter(fmtter)

        if self.verbose == 3:

            step_count = logs["step_count"]
            step_reward = logs["step_reward"]
            self.logger.info(f"At step {step_count} > reward id {step_reward:.1f}")

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
