"""
This script demonstrates the use of `torch.export` for exporting PyTorch models with examples
that include basic model export, handling dynamic shapes, and creating custom operators.
"""
import torch
from torch.export import Dim, export

from common.utils.arg_parser import get_common_args


class MyModule(torch.nn.Module):
    """
    Example neural network module for demonstration purposes.

    A simple model with a linear layer followed by a ReLU activation function.
    """

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(100, 10)

    def forward(self, x, y):
        """ Forward pass of the model.
        """
        return torch.nn.functional.relu(self.lin(x + y), inplace=True)


def export_model_examples(verbose=False):
    """
    Demonstrates exporting a PyTorch model to an ExportedProgram.

    Args:
        verbose (bool): If True, prints detailed information about the exported model.
    """
    model = MyModule()
    exported_model = export(model, (torch.randn(8, 100), torch.randn(8, 100)))

    if verbose:
        print("Exported Model Details:")
        print(f"Type: {type(exported_model)}")
        print(f"Graph Module: {exported_model.graph_module}")
        print(f"Graph Signature: {exported_model.graph_signature}")


def dynamic_shapes_examples():
    """
    Demonstrates handling dynamic shapes in exported models using torch.export.
    """
    inp1 = torch.randn(10, 10, 2)

    class DynamicShapesExample1(torch.nn.Module):
        """ Example for the DynamicShapes.
        """

        def forward(self, x):
            """ Forward pass of the model.
            """
            x = x[:, 2:]
            return torch.relu(x)

    inp1_dim1 = Dim("inp1_dim1", min=4, max=18)
    dynamic_shapes1 = {
        "x": {1: inp1_dim1},
    }
    _ = export(
        DynamicShapesExample1(), (inp1,), dynamic_shapes=dynamic_shapes1)
    print("Dynamic Shapes Example 1 exported successfully.")

    # Demonstrating dimension constraints
    inp2 = torch.randn(4, 8)
    inp3 = torch.randn(8, 2)

    class DynamicShapesExample2(torch.nn.Module):
        """ Example for the DynamicShapes.
        """

        def forward(self, x, y):
            """ Forward pass of the model.
            """
            return x @ y

    inp2_dim0 = Dim("inp2_dim0")
    inner_dim = Dim("inner_dim")
    inp3_dim1 = Dim("inp3_dim1")
    dynamic_shapes2 = {
        "x": {0: inp2_dim0, 1: inner_dim},
        "y": {0: inner_dim, 1: inp3_dim1},
    }
    _ = export(
        DynamicShapesExample2(), (inp2, inp3), dynamic_shapes=dynamic_shapes2)
    print("Dynamic Shapes Example 2 exported successfully.")


def custom_op_example():
    """
    Demonstrates creating and exporting a custom operator with torch.export.
    """
    @torch.library.custom_op("my_custom_library::custom_op", mutates_args={})
    def custom_op(tensor_input: torch.Tensor) -> torch.Tensor:
        print("Custom_op called!")
        return torch.relu(tensor_input)

    @custom_op.register_fake
    def custom_op_meta(x):
        return torch.empty_like(x)

    class CustomOpExample(torch.nn.Module):
        """ Class for the custom operator.
        """

        def forward(self, x):
            """ Forward pass of the model.
            """
            x = torch.sin(x)
            x = torch.ops.my_custom_library.custom_op(x)
            x = torch.cos(x)
            return x

    _ = export(CustomOpExample(), (torch.randn(3, 3),))
    print("Custom Op Example exported successfully.")


def main():
    """
    Entry point of the script.
    Demonstrates various functionalities of torch.export with examples.
    """
    args = get_common_args().parse_args()
    torch.manual_seed(args.seed)

    print("Exporting basic model example...")
    export_model_examples(verbose=args.verbose)

    print("\nExporting dynamic shapes examples...")
    dynamic_shapes_examples()

    print("\nExporting custom operator example...")
    custom_op_example()


if __name__ == "__main__":
    main()
