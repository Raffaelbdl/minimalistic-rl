# From [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)

## PPO implementation details implemented

- [x] Vectorized architecture
- [x] Orthogonal Initialization of Weights and Constant Initialization of biases
- [x] The Adam Optimizer’s Epsilon Parameter
- [x] Adam Learning Rate Annealing
- [x] Generalized Advantage Estimation
- [x] Mini-batch Updates
- [x] Normalization of Advantages
- [x] Clipped surrogate objective
- [] Value Function Loss Clipping
- [x] Overall Loss and Entropy Bonus
- [x] Global Gradient Clipping
- [] Debug variables
- [x] Shared and separate MLP networks for policy and value functions
- [x] The Use of NoopResetEnv
- [x] The Use of MaxAndSkipEnv
- [x] The Use of EpisodicLifeEnv
- [] The Use of FireResetEnv
- [x] The Use of WarpFrame
- [x] The Use of ClipRewardEnv
- [x] The Use of FrameStack
- [x] Shared Nature-CNN network for the policy and value functions
- [x] Scaling the Images to Range [0, 1]
- [] 