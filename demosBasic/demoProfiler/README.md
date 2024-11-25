# PyTorch Model Profiling Demo

This script demonstrates profiling the performance of a PyTorch ResNet18 model on both CPU and GPU. It includes detailed analysis of inference time and memory usage, as well as tracing long-running tasks using a custom profiling schedule.

## Features

1. **CPU Profiling**
   - Measures inference time and memory usage for a ResNet18 model on the CPU.
   - Provides detailed statistics about the computation.

2. **GPU Profiling**
   - Measures inference time and memory usage for the model on the GPU (if available).
   - Highlights CUDA-specific profiling details.

3. **Custom Task Tracing**
   - Profiles long-running tasks with a customizable profiling schedule.
   - Generates detailed trace logs for each step of the execution.

4. **Output Trace Files**
   - Saves detailed trace logs to user-specified filenames for further analysis.

## Key Use Cases
- Analyzing model performance for optimization.
- Understanding resource usage (time and memory) across CPU and GPU.
- Profiling and tracing long-running or complex computation tasks.
