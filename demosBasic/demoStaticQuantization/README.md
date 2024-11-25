# PyTorch Static Quantization Demo

This script showcases the static quantization process for a PyTorch model, comparing its performance in FP32 and INT8 formats.

## Features

1. **Model Quantization**
   - Implements static quantization using PyTorch’s QuantStub and DeQuantStub.
   - Prepares the model for calibration and converts it to INT8 format.

2. **Model Size Comparison**
   - Compares the file size of the FP32 and INT8 versions of the model.
   - Quantized model demonstrates significant size reduction.

3. **Latency Measurement**
   - Measures and compares the latency of FP32 and INT8 models during inference.

4. **Accuracy Evaluation**
   - Compares the outputs of the FP32 and INT8 models in terms of mean absolute values and differences.
   - Quantifies the trade-off in accuracy due to quantization.

5. **Detailed Logging**
   - Logs the transformations and configurations applied to the model at each stage.

## Key Insights
- Quantized models are smaller and faster but may experience minor accuracy trade-offs.
- Static quantization can be an effective optimization strategy for deployment on edge devices.
