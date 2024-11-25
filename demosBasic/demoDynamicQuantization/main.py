"""
This script demonstrates the creation and dynamic quantization of an LSTM model, 
and compares floating-point and quantized versions in terms of size, latency, and accuracy.
"""
from common.utils.arg_parser import get_args as get_common_args

import os
import timeit
import torch
import torch.nn as nn
import torch.quantization

# Constants
TEMP_FILE_PATH = "temp.p"

class LSTMForDemonstration(nn.Module):
    """
    A simple LSTM model for demonstration purposes, wrapping around `nn.LSTM`.

    Args:
        in_dim (int): Input dimension for the LSTM.
        out_dim (int): Output dimension for the LSTM.
        depth (int): Number of LSTM layers.
    """
    def __init__(self, in_dim, out_dim, depth):
        super(LSTMForDemonstration, self).__init__()
        self.lstm = nn.LSTM(in_dim, out_dim, depth)

    def forward(self, inputs, hidden):
        """
        Forward pass of the LSTM model.

        Args:
            inputs (torch.Tensor): Input tensor.
            hidden (tuple): Initial hidden and cell states.

        Returns:
            tuple: Output and hidden states.
        """
        out, hidden = self.lstm(inputs, hidden)
        return out, hidden

def get_args():
    """
    Parses command-line arguments for LSTM model configuration.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser, config = get_common_args(True)

    parser.add_argument('--model-dimension', type=int, default=8,
                        help='Dimension of LSTM (default: 8)')
    parser.add_argument('--number-of-layers', type=int, default=1,
                        help='Number of LSTM layers (default: 1)')
    parser.add_argument('--sequence-length', type=int, default=20,
                        help='Length of the input sequence (default: 20)')

    args = parser.parse_args()
    args.config = config
    return args

def initialize_hidden_state(num_layers, batch_size, model_dim):
    """
    Initializes the hidden and cell states for the LSTM model.

    Args:
        num_layers (int): Number of LSTM layers.
        batch_size (int): Batch size.
        model_dim (int): Model dimension.

    Returns:
        tuple: Initialized hidden and cell states.
    """
    return (
        torch.randn(num_layers, batch_size, model_dim),
        torch.randn(num_layers, batch_size, model_dim)
    )

def print_size_of_model(model, label=""):
    """
    Prints and returns the size of a model in KB.

    Args:
        model (torch.nn.Module): Model whose size is to be calculated.
        label (str): Label to identify the model in the printout.

    Returns:
        float: Size of the model in KB.
    """
    torch.save(model.state_dict(), TEMP_FILE_PATH)
    size = os.path.getsize(TEMP_FILE_PATH) / 1e3  # Convert bytes to KB
    print(f"Model: {label} \t Size (KB): {size:.2f}")
    os.remove(TEMP_FILE_PATH)
    return size

def compare_latency(model, inputs, hidden, label=""):
    """
    Measures and prints the latency of a model.

    Args:
        model (torch.nn.Module): Model to measure.
        inputs (torch.Tensor): Input data.
        hidden (tuple): Initial hidden state.
        label (str): Label for identifying the model type.
    """
    latency = timeit.timeit(lambda: model(inputs, hidden), number=1)
    print(f"{label} Latency: {latency:.4f} seconds")

def compare_accuracy(float_model, quantized_model, inputs, hidden):
    """
    Compares accuracy between the floating-point and quantized models by
    measuring the mean absolute difference in outputs.

    Args:
        float_model (torch.nn.Module): Floating-point model.
        quantized_model (torch.nn.Module): Quantized model.
        inputs (torch.Tensor): Input data.
        hidden (tuple): Initial hidden state.
    """
    with torch.no_grad():
        out_fp, _ = float_model(inputs, hidden)
        out_q, _ = quantized_model(inputs, hidden)

        mean_fp = torch.mean(torch.abs(out_fp)).item()
        mean_q = torch.mean(torch.abs(out_q)).item()
        mean_diff = torch.mean(torch.abs(out_fp - out_q)).item()

        print(f"Mean absolute output (FP32): {mean_fp:.5f}")
        print(f"Mean absolute output (INT8): {mean_q:.5f}")
        print(f"Mean absolute difference: {mean_diff:.5f} "
              f"({mean_diff / mean_fp * 100:.2f}%)")

def main():
    """
    Main function to set up the LSTM model, quantize it, and compare
    model size, latency, and accuracy between floating-point and quantized versions.
    """
    args = get_args()
    torch.manual_seed(args.seed)

    device = torch.device("cpu")  # Dynamic quantization only supported on CPU
    torch.set_default_device(device)

    # Generate random input and hidden state
    inputs = torch.randn(args.sequence_length, args.config.batch_size, args.model_dimension)
    hidden = initialize_hidden_state(args.number_of_layers, args.config.batch_size, args.model_dimension)

    # Create the floating-point LSTM model
    float_lstm = LSTMForDemonstration(args.model_dimension, args.model_dimension, args.number_of_layers)

    # Perform dynamic quantization
    try:
        quantized_lstm = torch.quantization.quantize_dynamic(
            float_lstm, {nn.LSTM, nn.Linear}, dtype=torch.qint8
        )
    except Exception as e:
        print(f"Quantization failed: {e}")
        return

    # Display model structures
    print('Floating-point LSTM:')
    print(float_lstm)
    print('\nQuantized LSTM:')
    print(quantized_lstm)

    # Compare model sizes
    size_fp32 = print_size_of_model(float_lstm, "FP32")
    size_int8 = print_size_of_model(quantized_lstm, "INT8")
    print(f"Quantized model is {size_fp32 / size_int8:.2f} times smaller")

    # Compare latency
    compare_latency(float_lstm, inputs, hidden, "Floating-point FP32")
    compare_latency(quantized_lstm, inputs, hidden, "Quantized INT8")

    # Compare accuracy
    compare_accuracy(float_lstm, quantized_lstm, inputs, hidden)

if __name__ == '__main__':
    main()
