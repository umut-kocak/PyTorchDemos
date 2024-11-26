"""
This script demonstrates training and fine-tuning a ResNet18 model on the Hymenoptera dataset,
including data visualization and predictions.
"""
import os
import time
import urllib.request
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision
from PIL import Image
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

from common.utils.arg_parser import get_common_args
from common.utils.helper import load_config_file
from common.utils.helper import select_default_device
from common.utils.train import train_single_epoch
from common.utils.visualise import display_image

cudnn.benchmark = True

DEFAULT_DATASET = 'hymenoptera_data'
DEFAULT_DATA_URL = "https://download.pytorch.org/tutorial/hymenoptera_data.zip"
DEFAULT_IMAGE = 'val/bees/72100438_73de9f17af.jpg'
OVERRIDE_BATCH_SIZE = 4
OVERRIDE_MOMENTUM = 0.9
DEFAULT_NUM_WORKERS = 4
DEFAULT_STEP_SIZE = 7


def get_args():
    """
    Parses command-line arguments for model and training configurations.
    Adds dataset and image path defaults for use with example data.

    Returns:
        argparse.Namespace: Parsed arguments with added configuration attributes.
    """
    parser = get_common_args()

    parser.add_argument('--default-dataset', type=str, default=DEFAULT_DATASET,
                        help='Default dataset folder within data path')
    parser.add_argument('--default-image', type=str, default=DEFAULT_IMAGE,
                        help='Default example image within dataset')

    parser.add_argument('--num-workers', type=int, default=DEFAULT_NUM_WORKERS,
                        help="Number of worker threads for data loading.")

    parser.add_argument('--step-size', type=int, default=DEFAULT_STEP_SIZE,
                        help="Step size for the optimizer.")

    args = parser.parse_args()
    args.config = load_config_file(args.config_path)
    # Apply configuration overrides
    args.config.batch_size = OVERRIDE_BATCH_SIZE
    args.config.moementum = OVERRIDE_MOMENTUM
    return args


def prepare_data_transforms():
    """Defines data augmentation and normalization for training and validation."""
    return {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }


def load_dataloaders(data_dir, batch_size, num_workers):
    """Creates data loaders for training and validation datasets."""
    data_transforms = prepare_data_transforms()
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
    return {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                           shuffle=True, num_workers=num_workers)
            for x in ['train', 'val']}, image_datasets['train'].classes


def setup_model(args, num_classes, finetune=False):
    """
    Sets up a ResNet18 model with specified output classes and optimizer.
    Parameters:
        args (Namespace): Command-line arguments with training configuration.
        num_classes (int): Number of output classes for final layer.
        finetune (bool): If True, only train the final layer; otherwise train all layers.
    Returns:
        tuple: Model, optimizer, and learning rate scheduler.
    """
    model = models.resnet18(weights='IMAGENET1K_V1')
    if finetune:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=args.config.learning_rate, momentum=args.config.momentum)
    scheduler = lr_scheduler.StepLR(
        optimizer,
        step_size=args.step_size,
        gamma=args.config.learning_rate_gamma)
    return model, optimizer, scheduler


def setup_tensorboard_logging(log_to_tensorboard):
    """Sets up TensorBoard logging if enabled."""
    if log_to_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter()
    return None


def train_model(args, model, dataloaders, optimizer,
                scheduler, device, writer=None):
    """
    Trains the model for a set number of epochs as specified in args.config.
    Parameters:
        args (Namespace): Command-line arguments with training configuration.
        model (torch.nn.Module): Model to train.
        dataloaders (dict): Data loaders for training and validation.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        device (torch.device): Device to run training on.
        writer (SummaryWriter): TensorBoard writer for logging.
    """
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, args.config.epochs + 1):
        train_single_epoch(
            args,
            model,
            dataloaders['train'],
            optimizer,
            criterion,
            device,
            epoch,
            writer=writer)
        scheduler.step()
        if writer:
            writer.flush()
    if writer:
        writer.close()


def visualize_model_predictions(
        model, data_loader, class_names, device, num_images=8):
    """
    Visualizes model predictions on images from a data loader in a single figure window.

    Parameters:
        model (torch.nn.Module): Trained model.
        data_loader (DataLoader): Data loader to get images for prediction.
        class_names (list): Name of the classes to predict
        device (torch.device): Device to run inference on.
        num_images (int): Number of images to show.
    """
    # Store the previous interactive mode state and disable it temporarily
    was_training = model.training
    model.eval()
    previous_ion = plt.isinteractive()
    plt.ioff()  # Turn off interactive mode to avoid closing individual plots

    images_so_far = 0
    fig = plt.figure(figsize=(12, 6))

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size(0)):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(
                    f'Predicted: {class_names[preds[j].item()]}, Actual: {class_names[labels[j].item()]}')

                display_image(inputs.cpu().data[j],
                              mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                              skip_show=True)

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    plt.show()
                    if previous_ion:  # Restore interactive mode if it was previously enabled
                        plt.ion()
                    return

    plt.show()  # Ensure all subplots display at once
    if previous_ion:
        plt.ion()
    model.train(mode=was_training)


def verify_data(args, data_url):
    """
    Verifies if the dataset directory exists, and if not, downloads and extracts the dataset.

    Args:
        args (argparse.Namespace): Parsed arguments containing default data path and dataset name.
        data_url (str): URL to download the dataset zip file if the data directory does not exist.

    Returns:
        bool: True if the data verification is complete (either found or downloaded).
    """
    # Construct paths for the data directory and zip file
    data_dir = os.path.join(args.default_data_path, args.default_dataset)
    data_zip_path = os.path.join(
        args.default_data_path, f"{
            args.default_dataset}.zip")

    # Check if data directory already exists
    if not os.path.exists(data_dir):
        # Download the dataset zip file if it does not exist locally
        if not os.path.exists(data_zip_path):
            print(f"Downloading dataset from {data_url}...")
            urllib.request.urlretrieve(data_url, data_zip_path)
            print("Download complete.")

        # Extract the dataset to the specified data directory
        print(f"Extracting dataset to {data_dir}...")
        with zipfile.ZipFile(data_zip_path, 'r') as zip_ref:
            zip_ref.extractall(args.default_data_path)

        # Remove the zip file after extraction to save space
        os.remove(data_zip_path)

    return True


def main():

    args = get_args()
    torch.manual_seed(args.seed)
    device = select_default_device(args)

    data_dir = os.path.join(args.default_data_path, args.default_dataset)
    if (not verify_data(args, DEFAULT_DATA_URL)):
        return
    dataloaders, class_names = load_dataloaders(
        data_dir, args.config.batch_size, args.num_workers)

    print("Visualizing some sample data:")
    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['val']))
    images = torchvision.utils.make_grid(inputs)
    display_image(images, title=[class_names[x] for x in classes],
                  mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    writer = setup_tensorboard_logging(args.config.log_to_tensorboard)

    # Finetuning the ConvNet
    print("The model is used as a whole")
    model, optimizer, scheduler = setup_model(
        args, num_classes=len(class_names), finetune=False)
    model = model.to(device)
    train_model(args, model, dataloaders, optimizer, scheduler, device, writer)
    visualize_model_predictions(
        model,
        dataloaders['val'],
        class_names,
        device,
        num_images=8)

    # Fixed feature extractor
    print("The model is being fine tuned; only the last layer is trained")
    model_conv, optimizer_conv, exp_lr_scheduler = setup_model(
        args, num_classes=len(class_names), finetune=True)
    model_conv = model_conv.to(device)
    train_model(
        args,
        model_conv,
        dataloaders,
        optimizer_conv,
        exp_lr_scheduler,
        device,
        writer)

    # Visualization
    visualize_model_predictions(
        model_conv,
        dataloaders['val'],
        class_names,
        device,
        num_images=8)


if __name__ == '__main__':
    main()
