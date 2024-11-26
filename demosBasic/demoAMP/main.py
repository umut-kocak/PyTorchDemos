"""
This script demonstrates training a fully connected neural network with configurable
precision settings, including mixed precision with gradient scaling.
"""
import torch

from common.utils import helper, train
from common.utils.arg_parser import get_common_args

# Default values as constants
DEFAULT_INPUT_SIZE = 4096
DEFAULT_OUTPUT_SIZE = 4096
DEFAULT_NUM_LAYERS = 3
DEFAULT_NUM_BATCHES = 50


def make_model(in_size, out_size, num_layers, activation_fn=torch.nn.ReLU):
    """
    Constructs a sequential neural network model with specified input and output sizes,
    number of layers, and activation function.

    Args:
        in_size (int): Size of input layer and hidden layers.
        out_size (int): Size of the output layer.
        num_layers (int): Number of layers in the model.
        activation_fn (torch.nn.Module): Activation function for hidden layers. Default is ReLU.

    Returns:
        torch.nn.Sequential: Constructed model.
    """
    layers = []
    for _ in range(num_layers - 1):
        layers.append(torch.nn.Linear(in_size, in_size))
        layers.append(activation_fn())
    layers.append(torch.nn.Linear(in_size, out_size))
    return torch.nn.Sequential(*layers).cuda()


def get_args():
    """
    Parses command-line arguments for model and training configurations.

    Returns:
        argparse.Namespace: Parsed arguments with added configuration attributes.
    """
    parser = get_common_args()

    parser.add_argument('--input-size', type=int, default=DEFAULT_INPUT_SIZE,
                        help='Input size to the model (default: 4096)')
    parser.add_argument('--output-size', type=int, default=DEFAULT_OUTPUT_SIZE,
                        help='Output size from the model (default: 4096)')
    parser.add_argument('--number-of-layers', type=int, default=DEFAULT_NUM_LAYERS,
                        help='Number of layers in the model (default: 3)')
    parser.add_argument('--number-of-batches', type=int, default=DEFAULT_NUM_BATCHES,
                        help='Number of batches of the data (default: 50)')

    args = parser.parse_args()
    args.config = helper.load_config_file(args.config_path)
    return args


def train_with_precision(args, model, input_data, target_data, optimizer,
                         criterion, device, use_amp, scaler=None, description=""):
    """
    Helper function to train the model with a specified precision setting and timer.

    Args:
        args: Command-line arguments.
        model: Neural network model.
        input_data: List of input tensors.
        target_data: List of target tensors.
        optimizer: Optimizer for model training.
        criterion: Loss function.
        device: Computation device.
        use_amp: Boolean to use automatic mixed precision.
        scaler: Gradient scaler for mixed precision, if needed.
        description: Description for timing output.
    """
    helper.start_timer()
    for epoch in range(1, args.config.epochs + 1):
        train.train_single_epoch(
            args,
            model,
            (input_data,
             target_data),
            optimizer,
            criterion,
            device,
            epoch,
            use_amp,
            scaler,
            None,
            True)
    helper.end_timer_and_print(description)


def main():
    """
    Main function to initialize configurations, create model, data, and start training loops.
    """
    args = get_args()
    torch.manual_seed(args.seed)
    device = helper.select_default_device(args)
    torch.set_default_device(device)

    # Create model, data, and criterion/optimizer
    input_data = [
        torch.randn(
            args.config.batch_size,
            args.input_size) for _ in range(
            args.number_of_batches)]
    target_data = [
        torch.randn(
            args.config.batch_size,
            args.output_size) for _ in range(
            args.number_of_batches)]
    model = make_model(
        args.input_size,
        args.output_size,
        args.number_of_layers)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.config.learning_rate)

    # Training with default precision
    train_with_precision(
        args,
        model,
        input_data,
        target_data,
        optimizer,
        criterion,
        device,
        use_amp=False,
        description="Default precision:")

    # Training with torch.autocast
    train_with_precision(
        args,
        model,
        input_data,
        target_data,
        optimizer,
        criterion,
        device,
        use_amp=True,
        description="With autograd:")

    # Training with torch.autocast and gradient scaler for mixed precision
    scaler = torch.amp.GradScaler(enabled=True)
    train_with_precision(
        args,
        model,
        input_data,
        target_data,
        optimizer,
        criterion,
        device,
        use_amp=True,
        scaler=scaler,
        description="With autograd and scaler:")


if __name__ == '__main__':
    main()
