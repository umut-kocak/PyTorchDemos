# Super-Resolution with PyTorch and ONNX

This script demonstrates the process of using a pre-trained super-resolution model with PyTorch and ONNX for upscaling images. It includes exporting the model to ONNX format, running inference, and saving super-resolved images.

## Features

1. **Pre-trained Model Loading**
   - Loads a super-resolution model pre-trained on image data.

2. **ONNX Export**
   - Converts the PyTorch model to the ONNX format for compatibility with ONNX runtime.

3. **Inference Comparison**
   - Runs inference using both PyTorch and ONNX.
   - Measures and compares inference times between the two frameworks.
   - Ensures consistency in results by validating outputs.

4. **Image Super-Resolution**
   - Processes an input image and generates a super-resolved version using ONNX runtime.
   - Combines the super-resolved luminance channel with original chroma channels for the final image.

5. **Image Saving**
   - Saves both the super-resolved image and a resized version of the original image for comparison.
