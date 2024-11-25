# LSTM Dynamic Quantization and Performance Comparison

This script demonstrates the dynamic quantization of an LSTM model using PyTorch. It includes:

1. Creation of a simple LSTM model.
2. Application of dynamic quantization to reduce model size and improve inference performance.
3. Comparison of floating-point (FP32) and quantized (INT8) models in terms of:
   - Model size.
   - Inference latency.
   - Accuracy, measured as the mean absolute difference in outputs.

The demonstration highlights the effectiveness of quantization for reducing resource usage while maintaining acceptable accuracy.
