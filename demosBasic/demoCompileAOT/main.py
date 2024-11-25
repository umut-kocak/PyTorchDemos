"""
This script demonstrates exporting a PyTorch ResNet18 model using AOTInductor, 
then compares the inference times of the exported model with torch.compile optimization.
"""
from common.utils.arg_parser import get_args as get_common_args
import common.utils.helper as helper
import os
import torch
from torchvision.models import ResNet18_Weights, resnet18

# Default model export filename
DEFAULT_MODEL_FILENAME = "resnet18_pt.so"

def get_args():
    """
    Parses command-line arguments, including the output file name for the exported model.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser, _ = get_common_args()
    parser.add_argument('--output-model-file-name', default=DEFAULT_MODEL_FILENAME,
                        help='File name for the exported model (default: resnet18_pt.so)')
    return parser.parse_args()

def export_model(model, example_inputs, output_model_file):
    """
    Exports the given model to a shared object using AOTInductor with specified example inputs.

    Args:
        model (torch.nn.Module): Model to be exported.
        example_inputs (tuple): Example inputs to define model structure.
        output_model_file (str): Path to save the shared object model file.

    Returns:
        str: Path to the exported model file.
    """
    batch_dim = torch.export.Dim("batch", min=2, max=32)  # Dynamic batch dimension
    aot_compile_options = {"aot_inductor.output_path": output_model_file}

    exported_program = torch.export.export(
        model,
        example_inputs,
        dynamic_shapes={"x": {0: batch_dim}},  # Dynamic shape specification
    )
    
    torch._inductor.aot_compile(
        exported_program.module(),
        example_inputs,
        options=aot_compile_options
    )

    return output_model_file

def timed_inference(model, example_input, label):
    """
    Measures and prints the time taken for model inference.

    Args:
        model (torch.nn.Module): Model to evaluate.
        example_input (torch.Tensor): Example input for the model.
        label (str): Label for the timing output.
    """
    torch._dynamo.reset()
    with torch.inference_mode():
        _, time_taken = helper.timed_function_call(lambda: model(example_input), True)
        print(f"Time taken for first inference with {label} is {time_taken:.2f} ms")

def main():
    """
    Main function to parse arguments, export the model, and compare inference times
    between AOTInductor and `torch.compile` optimization.
    """
    args = get_args()
    torch.manual_seed(args.seed)
    device = "cpu"

    # Load and prepare the ResNet18 model with pretrained weights
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.eval()
    model.to(device)

    # Define example inputs and export the model
    example_inputs = (torch.randn(2, 3, 224, 224, device=device),)
    output_model_path = os.path.join(os.getcwd(), args.output_model_file_name)

    # Export the model with AOTInductor
    print("Exporting model...")
    export_model(model, example_inputs, output_model_path)

    # Load the exported model for inference
    model = torch._export.aot_load(output_model_path, device=device)
    example_input_single = torch.randn(1, 3, 224, 224, device=device)

    # Measure time for first inference with AOTInductor
    timed_inference(model, example_input_single, label="AOTInductor")

    # Measure time for first inference with torch.compile
    model = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
    model.eval()
    model = torch.compile(model)
    timed_inference(model, example_input_single, label="torch.compile")

if __name__ == '__main__':
    main()


