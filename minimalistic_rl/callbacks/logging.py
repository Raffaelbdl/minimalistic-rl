import logging

from minimalistic_rl.callbacks.callback import Callback

DEFAULT_FMT = "%(asctime)s | %(levelname)s | %(message)s"
STEP_FMT = "%(asctime)s | STEP | %(message)s"
EPISODE_FMT = "%(asctime)s | EPISODE | %(message)s"


class Logger(Callback):
    def __init__(self, verbose: int):

        self.logger = init_logger("minimalistic-rl/logger", DEFAULT_FMT, logging.INFO)
        self.verbose = verbose

    def at_train_start(self, logs: dict):
        self.logger.info("Training is starting !")

        algo = logs["algo"]
        n_steps = log_handle_large_int(logs["n_steps"])
        self.logger.info(f"Algo is {algo}, training for {n_steps} steps")

    def at_train_end(self, logs: dict):
        self.logger.info("Training has finished !")

    def at_episode_end(self, logs: dict):
        fmtter = logging.Formatter(EPISODE_FMT)
        self.logger.handlers[0].setFormatter(fmtter)

        ep_count = logs["ep_count"]
        ep_reward = logs["ep_reward"]
        self.logger.info(f"Episode is {ep_count}, reward is {ep_reward}")

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
