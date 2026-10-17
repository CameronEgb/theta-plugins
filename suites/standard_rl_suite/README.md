# Standard RL Suite

Meta-package providing a complete, turnkey reinforcement learning stack for Theta-IDE.

## Included Components
- **Models**:
  - `dnn`: Multi-Layer Perceptron policy and Q-network.
  - `nature_cnn`: Deep convolutional policy for visual/Atari tasks.
  - `dueling_resnet`: Scaled residual network with dueling Q-decomposition.
- **Algorithms**:
  - `online_baselines`: PyTorch Lightning PPO algorithm.
  - `offline_baselines`: PyTorch Lightning CQL and IQL algorithms.
- **Environments**:
  - `gym_classic_control`: Standard Gymnasium classic control environments (CartPole, MountainCar, Pendulum, Acrobot).
