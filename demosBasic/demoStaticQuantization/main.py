from common.utils.arg_parser import get_args as get_common_args

import torch
import torch.nn as nn
import os
import timeit
from pathlib import Path

class M(torch.nn.Module):
    """
    A sample model with layers that can be statically quantized.
    Includes QuantStub and DeQuantStub to handle conversions.
    """
    def __init__(self):
        super().__init__()
        # QuantStub converts tensors from floating point to quantized
        self.quant = torch.ao.quantization.QuantStub()
        self.conv = nn.Conv2d(1, 1, 1)
        self.relu = nn.ReLU()
        # DeQuantStub converts tensors from quantized to floating point
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        # Convert tensors to quantized form
        x = self.quant(x)
        x = self.conv(x)
        x = self.relu(x)
        # Convert tensors back to floating point form
        x = self.dequant(x)
        return x

def get_args():
    """
    Retrieves and parses command-line arguments.

    Returns:
        Namespace: Parsed command-line arguments.
    """
    parser, _ = get_common_args()
    return parser.parse_args()

def print_size_of_model(model: nn.Module, label: str = "") -> int:
    """
    Calculates and prints the size of the model in kilobytes.

    Args:
        model (nn.Module): The model whose size is to be calculated.
        label (str): A label to identify the model in the output.

    Returns:
        int: Size of the model in KB.
    """
    temp_path = Path("temp.p")
    torch.save(model.state_dict(), temp_path)
    size = temp_path.stat().st_size
    print(f"Model: {label} \t Size (KB): {size / 1e3:.2f}")
    temp_path.unlink()
    return size

def compare_latency(model: nn.Module, input_data: torch.Tensor, label: str = "") -> float:
    """
    Evaluates and prints the latency of the model.

    Args:
        model (nn.Module): The model to test.
        input_data (torch.Tensor): Sample input for the model.
        label (str): Label to identify the model type.

    Returns:
        float: Execution time for a single forward pass.
    """
    print(f"Latency for {label} model:")
    latency = timeit.timeit(lambda: model(input_data), number=1)
    print(f"Time: {latency:.6f} seconds")
    return latency

def compare_accuracy(fp32_out: torch.Tensor, int8_out: torch.Tensor) -> None:
    """
    Compares the mean absolute values and difference between FP32 and INT8 model outputs.

    Args:
        fp32_out (torch.Tensor): Output tensor from the FP32 model.
        int8_out (torch.Tensor): Output tensor from the INT8 model.
    """
    mag_fp32 = torch.mean(torch.abs(fp32_out)).item()
    mag_int8 = torch.mean(torch.abs(int8_out)).item()
    diff = torch.mean(torch.abs(fp32_out - int8_out)).item()

    print(f"Mean absolute value of FP32 model output: {mag_fp32:.5f}")
    print(f"Mean absolute value of INT8 model output: {mag_int8:.5f}")
    print(f"Difference mean absolute value: {diff:.5f} ({(diff / mag_fp32 * 100):.2f}% difference)")

def main():
    """
    Main function to execute model quantization and comparisons.
    """
    args = get_args()
    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    torch.set_default_device(device)

    # Initialize and prepare model for quantization
    model_fp32 = M().to(device)
    model_fp32.eval()

    # Set quantization configuration and fuse modules
    model_fp32.qconfig = torch.ao.quantization.get_default_qconfig('x86')
    model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32, [['conv', 'relu']])

    # Prepare model for calibration
    model_fp32_prepared = torch.ao.quantization.prepare(model_fp32_fused)

    # Calibrate with sample input
    input_fp32 = torch.randn(4, 1, 4, 4, device=device)
    model_fp32_prepared(input_fp32)

    # Convert to quantized model
    model_int8 = torch.ao.quantization.convert(model_fp32_prepared)

    # Display model structures
    print("\nModel in floating point (FP32):")
    print(model_fp32)
    print("\nModel quantized to INT8:")
    print(model_int8)

    # Model size comparison
    fp32_size = print_size_of_model(model_fp32, "FP32")
    int8_size = print_size_of_model(model_int8, "INT8")
    print(f"Quantized model is {fp32_size / int8_size:.2f} times smaller than FP32 model.")

    # Latency comparison
    compare_latency(model_fp32, input_fp32, "FP32")
    compare_latency(model_int8, input_fp32, "INT8")

    # Accuracy comparison
    output_fp32 = model_fp32(input_fp32)
    output_int8 = model_int8(input_fp32)
    compare_accuracy(output_fp32, output_int8)

if __name__ == '__main__':
    main()
