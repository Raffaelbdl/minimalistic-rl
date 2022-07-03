from collections import deque
from typing import List

import wandb


class Callback:
    def at_train_start(self, logs: dict):
        pass

    def at_train_end(self, logs: dict):
        pass

    def at_episode_start(self, logs: dict):
        pass

    def at_episode_end(self, logs: dict):
        pass

    def at_step_start(self, logs: dict):
        pass

    def at_step_end(self, logs: dict):
        pass


class WandbCallback(Callback):
    def __init__(self, log_keys: List[str] = None):
        self.log_keys = (
            log_keys
            if log_keys is not None
            else ["step_count", "episodic_return", "total_loss"]
        )

    def at_episode_end(self, logs: dict):
        wandb_log = {}
        for key in self.log_keys:
            if "mean" in key and isinstance(logs[key], deque):
                wandb_log[key] = sum(logs[key]) / len(logs[key])
            else:
                wandb_log[key] = logs[key]
        wandb.log(wandb_log)
