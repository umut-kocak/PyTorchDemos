"""
This script demonstrates training, evaluation, and prediction using a custom classification network
on the FashionMNIST dataset.
"""
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from common.models import ClassificationNet as ClassificationNet
from common.utils.arg_parser import get_args as get_common_args
from common.utils.helper import select_default_device
from common.utils.train import train_and_evaluate


def get_args():
    """
    Parses command-line arguments for model and training configurations.

    Returns:
        argparse.Namespace: Parsed arguments with added configuration attributes.
    """
    parser, config = get_common_args(True)
    args = parser.parse_args()
    args.config = config
    return args


def setup_data_loaders(args) -> Tuple[DataLoader, DataLoader]:
    """
    Sets up data loaders for training and testing.

    Args:
        args (argparse.Namespace): Parsed arguments including configuration.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and testing data loaders.
    """
    # DataLoader configuration
    train_kwargs = {'batch_size': args.config.batch_size}
    test_kwargs = {'batch_size': args.config.test_batch_size}

    device = select_default_device(args)
    if device == torch.device("cuda"):
        cuda_kwargs = {'num_workers': 2, 'pin_memory': True, 'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
        test_kwargs['shuffle'] = False  # Disable shuffle for test loader

    # Download and transform datasets
    transform = ToTensor()
    training_data = datasets.FashionMNIST(
        root=args.default_data_path,
        train=True,
        download=True,
        transform=transform
    )
    test_data = datasets.FashionMNIST(
        root=args.default_data_path,
        train=False,
        download=True,
        transform=transform
    )

    # Initialize data loaders
    train_loader = DataLoader(training_data, **train_kwargs)
    test_loader = DataLoader(test_data, **test_kwargs)

    return train_loader, test_loader


def initialize_model(args) -> torch.nn.Module:
    """
    Initializes the model, loss function, and optimizer.

    Args:
        args (argparse.Namespace): Parsed arguments including configuration.

    Returns:
        torch.nn.Module: The initialized model.
    """
    device = select_default_device(args)
    model = ClassificationNet(
        input_channels=1, input_size=(
            28, 28), hidden_conv_dims=None, hidden_fc_dims=[
            512, 512], num_classes=10).to(device)
    return model


def predict_sample(model, test_data, device, classes):
    """
    Runs a single prediction on a sample from the test dataset.

    Args:
        model (torch.nn.Module): Trained model for inference.
        test_data (Dataset): Test dataset to sample from.
        device (torch.device): Device for model and tensor.
        classes (List[str]): List of class labels.
    """
    model.eval()
    x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        x = x.to(device)
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')


def main():
    """
    Main function to load data, initialize the model, run training, and test the model.
    """
    args = get_args()
    torch.manual_seed(args.seed)

    # Select device and load data
    device = select_default_device(args)
    train_loader, test_loader = setup_data_loaders(args)
    model = initialize_model(args)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.config.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    train_and_evaluate(
        args,
        model,
        train_loader,
        test_loader,
        optimizer,
        criterion)

    # Save the trained model
    if args.save_model:
        torch.save(model.state_dict(), "object_classification.pt")

    # Class labels for FashionMNIST
    classes = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]

    # Perform a sample prediction
    predict_sample(model, test_loader.dataset, device, classes)


if __name__ == '__main__':
    main()
