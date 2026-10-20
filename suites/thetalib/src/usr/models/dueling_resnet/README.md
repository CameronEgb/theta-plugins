# Dueling ResNet Component

Scaled Dueling Residual Network (Dueling ResNet) policy and Q-network architecture for Theta-IDE.

## Features
- Scaled deep representations with LayerNorm, GELU activations, and Residual skip connections.
- Dueling Q-Value decomposition: `Q(s, a) = V(s) + (A(s, a) - mean_a'(A(s, a')))`.
- Actor-critic policy heads compatible with hybrid BlendRL and RL algorithms (PPO, CQL, IQL).
