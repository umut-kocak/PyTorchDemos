# Fully Connected Neural Network with Precision Configurations

This script showcases a PyTorch implementation of a fully connected neural network, allowing users to experiment with different precision configurations. Key features include:

1. **Model Construction**: Dynamically constructs a sequential neural network with configurable input/output sizes, number of layers, and activation functions.
2. **Training Configurations**: Supports training with default precision, mixed precision using `torch.autocast`, and mixed precision with gradient scaling.
3. **Performance Monitoring**: Includes timers to measure the impact of precision settings on training performance.

This project is designed for understanding the benefits of mixed precision training and provides a flexible framework for experimenting with precision optimization.
