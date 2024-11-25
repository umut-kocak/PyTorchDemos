"""
This script demonstrates training and evaluation of a Spatial Transformer Network (STN) on the
MNIST dataset, including visualization of spatial transformations.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import make_grid

from common.models import STNNet
from common.utils.arg_parser import get_args as get_common_args
from common.utils.helper import select_default_device
from common.utils.train import train_and_evaluate
from common.utils.visualise import convert_image_np

DEFAULT_NUM_WORKERS = 4
OVERRIDE_LEARNING_RATE = 0.01


def get_args():
    """
    Parses command-line arguments for model and training configurations,
    setting defaults where appropriate.

    Returns:
        argparse.Namespace: Parsed arguments with added configuration attributes.
    """
    parser, config = get_common_args(True)
    # Apply configuration overrides
    config.learning_rate = OVERRIDE_LEARNING_RATE

    # Set default values for configuration attributes
    parser.add_argument(
        '--num-workers',
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Number of worker threads for data loading.")

    # Parse the command-line arguments
    args = parser.parse_args()
    args.config = config
    return args


def load_data(args):
    """
    Loads and returns training and test data loaders for MNIST.

    Args:
        args (argparse.Namespace): Parsed arguments including configuration.
    Returns:
        tuple: (train_loader, test_loader)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            root=args.default_data_path,
            train=True,
            download=True,
            transform=transform),
        batch_size=args.config.batch_size, shuffle=True, num_workers=args.num_workers)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            root=args.default_data_path,
            train=False,
            transform=transform),
        batch_size=args.config.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, test_loader


def setup_model(device):
    """
    Initializes the STN model and moves it to the specified device.

    Args:
        device (torch.device): Device to load the model on.

    Returns:
        torch.nn.Module: Initialized model.
    """
    model = STNNet().to(device)
    return model


def visualize_spatial_transform(model, test_loader, device):
    """
    Visualizes the transformation performed by the STN on a batch of test images.

    Args:
        model (torch.nn.Module): Trained STN model.
        test_loader (DataLoader): Test data loader.
        device (torch.device): Device to run the visualization on.
    """
    with torch.no_grad():
        data = next(iter(test_loader))[0].to(device)
        input_tensor = data.cpu()
        transformed_tensor = model.stn(data).cpu()

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        in_grid = convert_image_np(make_grid(input_tensor), mean, std)
        out_grid = convert_image_np(make_grid(transformed_tensor), mean, std)

        # Plot original and transformed images side by side
        fig, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Original Images')
        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')


def main():
    """
    Main function to set up configurations, load data and model, and initiate training.
    """
    args = get_args()
    torch.manual_seed(args.seed)

    # Select device
    device = select_default_device(args)

    # Load data
    train_loader, test_loader = load_data(args)

    # Setup model
    model = setup_model(device)

    # Train and evaluate model
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.config.learning_rate)
    criterion = torch.nn.functional.nll_loss
    train_and_evaluate(
        args,
        model,
        train_loader,
        test_loader,
        optimizer,
        criterion)

    # Visualize the spatial transformer network
    plt.ion()  # Enable interactive mode for live visualization
    visualize_spatial_transform(model, test_loader, device)
    plt.ioff()  # Disable interactive mode
    plt.show()


if __name__ == '__main__':
    main()
