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
