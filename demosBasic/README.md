# demosBasic

The `demosBasic` folder contains a collection of fundamental PyTorch demos, each focused on a specific concept or feature to enhance understanding and practical implementation of key PyTorch functionalities.

## Demos Overview

### 1. demoAMP
Demonstrates Automatic Mixed Precision (AMP) to optimize model performance by reducing memory usage and accelerating training through mixed-precision training.

### 2. demoCompileAOT
Explores Ahead-of-Time (AOT) compilation techniques in PyTorch to improve model performance by compiling parts of the model graph prior to execution.
Highlights PyTorch's model export and optimization features.

### 3. demoDynamicQuantization
Compares a floating-point LSTM model with its dynamically quantized counterpart, evaluating size, latency, and accuracy to showcase PyTorch's quantization capabilities.

### 4. demoExport
Provides an example of exporting PyTorch models using PyTorch's native export functionality, enabling easy model saving and loading for deployment.
Showcases basic model export, dynamic shape handling, and custom operator creation for optimized deployment.

### 5. demoImageTransforms
Applies geometric, photometric, and augmentation transformations on images using PyTorch, showcasing their visual effects and preprocessing pipelines.

### 6. demoLogging
Illustrates best practices for logging in PyTorch, including function compilation and analysis through logging configurations like tracing, graph generation, and fusion decisions.

### 7. demoONNX
Showcases a PyTorch-based super-resolution pipeline, including ONNX export, inference time comparison, and super-resolved image generation and saving.

### 8. demoProfiler
Profiles a ResNet18 model on CPU and GPU, measuring inference time, memory usage, and tracing long-running tasks with customizable schedules, providing trace logs for optimization insights.

### 9. demoPruning
Applies various pruning techniques to a LeNet model, including structured, unstructured, global, and custom methods, showcasing sparsity optimization and flexibility in model size reduction.

### 10. demoStaticQuantization
Performs static quantization of a PyTorch model, comparing FP32 and INT8 formats in terms of size, latency, and accuracy, highlighting the benefits of quantization for deployment.

Each subfolder contains a standalone Python demo, presenting the core concept in a clear and practical manner to support learning and experimentation with PyTorch.
