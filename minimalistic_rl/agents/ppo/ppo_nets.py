from typing import Tuple

import haiku as hk
import jax
from jax import numpy as jnp
import numpy as np

from minimalistic_rl.wrapper import VecEnv


def make_mlp_nets(
    environment: VecEnv, hidden_layers: Tuple[int] = (64, 64)
) -> Tuple[hk.Transformed, hk.Transformed]:
    """Make a simple MLP module with PPO weights and biases initialization"""

    class MLP(hk.Module):
        def __call__(self, inputs, is_training: bool = False):
            x = hk.Flatten()(inputs)
            for i, h in enumerate(hidden_layers):
                x = hk.Linear(
                    h,
                    w_init=hk.initializers.Orthogonal(np.sqrt(2)),
                    b_init=hk.initializers.Constant(0.0),
                    name=f"linear_{i}",
                )(x)
                x = jax.nn.relu(x)
            return x

    @hk.transform
    def actor_fn(S, is_training: bool = False):

        x = MLP("actor_mlp")(S, is_training)
        x = hk.Linear(
            environment.action_space.n,
            w_init=hk.initializers.Orthogonal(0.01),
            b_init=hk.initializers.Constant(0.0),
            name="actor_output",
        )(x)

        return x

    @hk.transform
    def critic_fn(S, is_training: bool = False):

        x = MLP("critic_mlp")(S, is_training)
        x = hk.Linear(
            1,
            w_init=hk.initializers.Orthogonal(1),
            b_init=hk.initializers.Constant(0.0),
            name="critic_output",
        )(x)

        return x

    return actor_fn, critic_fn


def make_atari_nets(environment: VecEnv) -> Tuple[hk.Transformed, hk.Transformed]:
    """Make a simple network with shared nature CNN"""

    class NatureCNN(hk.Module):
        def __call__(self, inputs, is_training: bool = False):
            inputs = jnp.transpose(inputs, (0, 2, 3, 1)).astype(jnp.float32)
            return hk.Sequential(
                [
                    hk.Conv2D(
                        32,
                        8,
                        4,
                        w_init=hk.initializers.Orthogonal(jnp.sqrt(2)),
                        b_init=hk.initializers.Constant(0.0),
                    ),
                    jax.nn.relu,
                    hk.Conv2D(
                        64,
                        4,
                        2,
                        w_init=hk.initializers.Orthogonal(jnp.sqrt(2)),
                        b_init=hk.initializers.Constant(0.0),
                    ),
                    jax.nn.relu,
                    hk.Conv2D(
                        64,
                        3,
                        1,
                        w_init=hk.initializers.Orthogonal(jnp.sqrt(2)),
                        b_init=hk.initializers.Constant(0.0),
                    ),
                    jax.nn.relu,
                    hk.Flatten(),
                    hk.Linear(
                        512,
                        w_init=hk.initializers.Orthogonal(jnp.sqrt(2)),
                        b_init=hk.initializers.Constant(0.0),
                    ),
                    jax.nn.relu,
                ],
            )(inputs)

    @hk.transform
    def actor_fn(S, is_training: bool = False):

        x = NatureCNN("nature_cnn")(S, is_training)
        x = hk.Linear(
            environment.action_space.n,
            w_init=hk.initializers.Orthogonal(0.01),
            b_init=hk.initializers.Constant(0.0),
            name="actor_output",
        )(x)

        return x

    @hk.transform
    def critic_fn(S, is_training: bool = False):

        x = NatureCNN("nature_cnn")(S, is_training)
        x = hk.Linear(
            1,
            w_init=hk.initializers.Orthogonal(1),
            b_init=hk.initializers.Constant(0.0),
            name="critic_output",
        )(x)

        return x

    return actor_fn, critic_fn
