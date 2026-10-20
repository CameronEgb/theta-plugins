# Deep Neural Network (DNN / MLP) Component

Standard Multi-Layer Perceptron (DNN) component plugin for Theta-IDE.
Wraps standard PyTorch Linear, LayerNorm, and Activation layers with optional Dueling Q-value heads.

## Features
- Fully configurable hidden layer dimensions
- LayerNorm and Dropout options
- Compatible with online RL (PPO) and offline RL (CQL, IQL)
- Implements `ForwardModelProtocol`
