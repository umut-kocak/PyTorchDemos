import logging
import torch

def check_device_compatibility():
    """
    Checks if the CUDA device supports `torch.compile`. This function currently requires
    a CUDA device with compute capability >= 7.0.
    
    Returns:
        bool: True if the device is compatible; otherwise, False.
    """
    if torch.cuda.is_available() and torch.cuda.get_device_capability() >= (7, 0):
        return True
    print("Skipping because torch.compile is not supported on this device.")
    return False

def separator(name):
    """
    Prints a visual separator for each section in the output and resets the dynamo state.
    
    Args:
        name (str): A label for the separator.
    """
    print(f"==================={name}=========================")
    torch._dynamo.reset()

def main():
    """
    Main function to compile and analyze a simple function using `torch.compile`.
    
    This script performs the following:
    1. Checks if the CUDA device is compatible.
    2. Defines a compiled function `fn` to be executed on the device.
    3. Runs `fn` multiple times with different logging configurations (tracing, graph, 
       fusion decisions, and output code) to understand the output of `torch.compile`.
    """
    # Check if CUDA device supports `torch.compile`
    if not check_device_compatibility():
        return

    # Define a simple function to be compiled by torch.compile
    @torch.compile()
    def fn(x, y):
        """
        Adds two tensors and returns the result incremented by 2.
        
        Args:
            x (torch.Tensor): First input tensor.
            y (torch.Tensor): Second input tensor.
        
        Returns:
            torch.Tensor: Resulting tensor after the computation.
        """
        z = x + y
        return z + 2

    # Prepare input tensors
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = (torch.ones(2, 2, device=device), torch.zeros(2, 2, device=device))

    # Set logging configurations for different stages of compilation and output
    separator("Dynamo Tracing")
    # Set log level to DEBUG to see dynamo tracing
    # Equivalent to TORCH_LOGS="+dynamo"
    torch._logging.set_logs(dynamo=logging.DEBUG)
    fn(*inputs)

    separator("Traced Graph")
    # Enable logging for the traced graph
    # Equivalent to TORCH_LOGS="graph"
    torch._logging.set_logs(graph=True)
    fn(*inputs)

    separator("Fusion Decisions")
    # Enable logging for fusion decisions
    # Equivalent to TORCH_LOGS="fusion"
    torch._logging.set_logs(fusion=True)
    fn(*inputs)

    separator("Output Code")
    # Enable logging to view the generated output code by inductor
    # Equivalent to TORCH_LOGS="output_code"
    torch._logging.set_logs(output_code=True)
    fn(*inputs)

    # Reset logging level to its original state (optional)
    separator("Reset Logging")
    torch._logging.set_logs(dynamo=False, graph=False, fusion=False, output_code=False)

if __name__ == '__main__':
    main()
