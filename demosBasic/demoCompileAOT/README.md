# PyTorch ResNet18 Model Export and Inference Timing Demo

This script showcases how to export a PyTorch ResNet18 model using AOTInductor and measure the inference performance. It includes the following key features:

1. Loads a pretrained ResNet18 model from `torchvision`.
2. Exports the model as a shared object file using dynamic shapes with AOTInductor.
3. Measures and compares the inference times of the exported model with the results of `torch.compile` optimization.

The demonstration is designed to highlight the capabilities of PyTorch's advanced model export and optimization pipelines.
