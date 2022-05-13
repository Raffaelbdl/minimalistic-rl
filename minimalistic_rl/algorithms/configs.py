"""Stores the python configs for all algorithms"""

DQN_CONFIG = {
    "algo": "dqn",
    "policy": "off",
    "epsilon": 0.1,
    "gamma": 0.99,
    "capacity": int(1e6),
    "batch_size": 128,
    "learning_rate": 0.001,
    "improve_cycle": 1,
    "n_train_steps": 1,
    "n_steps": int(1e6),
    "episode_cycle_len": 10,
    "verbose": 2,
}

PPO_CONFIG = {
    "algo": "ppo",
    "policy": "on",
    "epsilon": 0.1,
    "gamma": 0.99,
    "T": 512,
    "batch_size": 128,
    "learning_rate": 0.001,
    "improve_cycle": 1,
    "n_train_steps": 1,
    "n_steps": int(1e6),
    "episode_cycle_len": 10,
    "verbose": 2,
}
