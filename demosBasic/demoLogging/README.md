# PyTorch Compile Demo

This script demonstrates the usage of PyTorch's `torch.compile` feature for compiling and analyzing a simple function. It provides insights into how `torch.compile` works through various logging configurations.

## Features

1. **Function Compilation**
   - Defines and compiles a simple function using `torch.compile`.
   - Executes the compiled function with sample input tensors.

2. **Analysis via Logging**
   - Dynamically configures logging to showcase:
     1. **Dynamo Tracing**: Logs the function's tracing details.
     2. **Traced Graph**: Logs the intermediate traced graph.
     3. **Fusion Decisions**: Logs decisions regarding operator fusion.
     4. **Output Code**: Displays the generated output code by the inductor backend.

3. **Logging Management**
   - Demonstrates toggling logging configurations to analyze different stages of compilation.

This script serves as a practical demonstration of PyTorch's `torch.compile` capabilities and its associated analysis tools.
