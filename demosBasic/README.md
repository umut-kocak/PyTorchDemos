# demosBasic

The `demosBasic` folder contains a collection of fundamental PyTorch demos, each focused on a specific concept or feature to enhance understanding and practical implementation of key PyTorch functionalities.

## Subfolders Overview

- **demoAMP**: Demonstrates Automatic Mixed Precision (AMP) to optimize model performance by reducing memory usage and accelerating training through mixed-precision training.

- **demoCompileAOT**: Explores Ahead-of-Time (AOT) compilation techniques in PyTorch to improve model performance by compiling parts of the model graph prior to execution.

- **demoDynamicQuantization**: Shows the application of Dynamic Quantization as a post-training technique, allowing the activation parameters to be quantized dynamically for improved inference efficiency on compatible hardware.

- **demoExport**: Provides an example of exporting PyTorch models using PyTorch's native export functionality, enabling easy model saving and loading for deployment.

- **demoImageTransforms**: Covers common image transformation techniques, demonstrating how to preprocess image data for training and inference in computer vision models.

- **demoLogging**: Illustrates best practices for logging in PyTorch, including tracking metrics, losses, and other key information during training to improve model monitoring and debugging.

- **demoONNX**: Examines the options for exporting PyTorch models to ONNX, enabling model compatibility with ONNX Runtime and other ONNX-supported frameworks.

- **demoProfiler**: Introduces PyTorch Profiler, a tool for performance analysis, helping identify bottlenecks and optimize model performance during training and inference.

- **demoPruning**: Demonstrates model pruning techniques to reduce the size of neural networks by removing redundant parameters, resulting in a more efficient model without significantly affecting performance.

- **demoStaticQuantization**: Details Static Quantization as a post-training technique, where model weights and activations are quantized to optimize inference performance on compatible hardware.

Each subfolder contains a standalone Python demo, presenting the core concept in a clear and practical manner to support learning and experimentation with PyTorch.
