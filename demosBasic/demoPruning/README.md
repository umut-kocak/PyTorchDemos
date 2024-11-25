# PyTorch Model Pruning Demo

This script showcases multiple pruning techniques applied to a PyTorch implementation of the LeNet model, focusing on improving sparsity and optimizing model performance.

## Features

1. **Layer-Specific Pruning**
   - Applies unstructured pruning to individual layers, such as randomly removing weights or pruning by L1 norm.
   - Demonstrates structured pruning by channel removal using L2 norm.

2. **Global Pruning**
   - Performs global pruning across all layers of the model based on L1 norm.
   - Calculates and logs the global sparsity achieved.

3. **Custom Pruning**
   - Implements a custom pruning method to prune every other entry in a specified tensor.
   - Showcases flexibility in defining and applying user-defined pruning techniques.

4. **Iterative Sparsity Analysis**
   - Logs changes in sparsity and parameter structure at each pruning stage.
   - Allows tracking of sparsity percentages across layers and globally.

5. **Support for Additional Layers**
   - Pruning applied across convolutional and fully connected layers of the LeNet architecture.

## Key Use Cases
- Reducing model size by sparsifying weights while maintaining accuracy.
- Experimenting with custom pruning strategies for research and optimization.
- Analyzing the impact of different pruning methods on neural network structure.
