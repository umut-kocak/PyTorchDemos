"""
This script demonstrates training, evaluation, and prediction using a custom classification network
on the CIFAR-10 dataset.
"""
from typing import Tuple

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from common.models import ClassificationNet
from common.utils.arg_parser import get_common_args
from common.utils.helper import load_config_file, select_default_device
from common.utils.train import train_and_evaluate

OVERRIDE_MOMENTUM = 0.9


def get_args():
    """
    Parses command-line arguments for model and training configurations.

    Returns:
        argparse.Namespace: Parsed arguments with added configuration attributes.
    """
    parser = get_common_args()
    args = parser.parse_args()
    args.config = load_config_file(args.config_path)
    # Apply configuration overrides
    args.config.momentum = OVERRIDE_MOMENTUM
    return args


def setup_data_loaders(args) -> Tuple[DataLoader, DataLoader]:
    """
    Sets up data loaders for CIFAR-10 training and testing.

    Args:
        args (argparse.Namespace): Parsed arguments including configuration.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and testing data loaders.
    """
    # DataLoader configuration
    train_kwargs = {'batch_size': args.config.batch_size}
    test_kwargs = {'batch_size': args.config.test_batch_size}
    device = select_default_device(args)

    if device.type == "cuda":
        cuda_kwargs = {'num_workers': 2, 'pin_memory': True, 'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
        test_kwargs['shuffle'] = False  # Disable shuffle for test loader

    # Define dataset transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Initialize datasets and data loaders
    trainset = torchvision.datasets.CIFAR10(
        root=args.default_data_path,
        train=True,
        download=True,
        transform=transform)
    testset = torchvision.datasets.CIFAR10(
        root=args.default_data_path,
        train=False,
        download=True,
        transform=transform)
    train_loader = DataLoader(trainset, **train_kwargs)
    test_loader = DataLoader(testset, **test_kwargs)

    return train_loader, test_loader


def initialize_model(args) -> torch.nn.Module:
    """
    Initializes and returns the model moved to the appropriate device.

    Args:
        args (argparse.Namespace): Parsed arguments including configuration.

    Returns:
        torch.nn.Module: The initialized model on the correct device.
    """
    device = select_default_device(args)
    model = ClassificationNet(
        input_channels=3, input_size=(
            32, 32), hidden_conv_dims=[
            6, 16], hidden_fc_dims=[
                120, 84], num_classes=10).to(device).to(device)
    return model


def predict_sample(model, device, test_loader, classes):
    """
    Runs a single prediction on a sample from the test dataset.

    Args:
        model (torch.nn.Module): Trained model for inference.
        device (torch.device): Device for model and tensor.
        test_loader (DataLoader): DataLoader for testing data.
        classes (Tuple[str]): Class labels for CIFAR-10.
    """
    # Fetch a batch of test images and labels
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    print('Actual Labels: ', ' '.join(
        f'{classes[labels[j]]:5s}' for j in range(4)))

    # Forward pass through the model
    images, labels = images.to(device), labels.to(device)  # Move to device
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join(
        f'{classes[predicted[j]]:5s}' for j in range(4)))

    # Run single prediction for one image
    model.eval()
    with torch.no_grad():
        x, y = images[0], labels[0]
        pred = model(x.unsqueeze(0))  # Add batch dimension for single image
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')


def main():
    """
    Main function to set up data, initialize model, run training and evaluation,
    and save the model. Finally, performs a sample prediction.
    """
    args = get_args()
    torch.manual_seed(args.seed)

    # Device selection and data loading
    device = select_default_device(args)
    train_loader, test_loader = setup_data_loaders(args)
    model = initialize_model(args)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.config.learning_rate,
        momentum=args.config.momentum)
    criterion = torch.nn.CrossEntropyLoss()
    train_and_evaluate(
        args,
        model,
        train_loader,
        test_loader,
        optimizer,
        criterion)

    # Save model if specified
    if args.save_model:
        torch.save(model.state_dict(), "object_classification.pt")

    # Class labels for CIFAR-10
    classes = (
        'plane',
        'car',
        'bird',
        'cat',
        'deer',
        'dog',
        'frog',
        'horse',
        'ship',
        'truck')

    # Run a sample prediction
    predict_sample(model, device, test_loader, classes)


if __name__ == '__main__':
    main()
