# Nature CNN Component

Canonical 3-stage convolutional neural network architecture (Mnih et al., 2015) for visual reinforcement learning and pixel-based state inputs (e.g. Atari 2600).

## Features
- 3 convolutional layers: `8x8 (stride 4) -> 4x4 (stride 2) -> 3x3 (stride 1)`
- 512-dim dense latent representation
- Integrated Actor-Critic and Q-value policy heads
- Automatic scaling and normalization for `[0, 255]` pixel inputs
