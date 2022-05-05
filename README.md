# Minimalistic RL
RL algorithms made simple in JAX

![status](https://img.shields.io/badge/status-work%20in%20progress-red)

The goal of this project is to make simple, efficient, user-friendly RL algorithms in JAX using HAIKU

## Implemented algorithms
- [x] DQN
- [x] PPO


## Handle `chex.dataclass` as observation

In some cases, custom observations may be necessary. Those can be automatically handled given a certain syntax.
The hk.Transformed given to the agent should be designed to also handle the custom observation.

```python
import chex
from gym.core import ObservationWrapper

@chex.dataclass
class CustomObservation:
    x: chex.Array
    y: chex.Array

class CustomWrapper(ObservationWrapper):
    def observation(self, observation):
        x = make_x_observation(observation)
        y = make_y_observation(observation)
        custom_observation = CustomObservation(
            x = x,
            y = y
        )
        return custom_observation
    
env = CustomWrapper(env)
```