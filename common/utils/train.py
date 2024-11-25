import os
import tempfile
import time

import torch
from torch.utils.data import DataLoader

from common.utils.helper import log_to_tensorboard, select_default_device


def device_to_str(device):
    """
    Convert a PyTorch device object to its string representation.

    Args:
        device (torch.device): The PyTorch device object.

    Returns:
        str: A string representing the device ('cuda', 'mps', or 'cpu').
    """
    if device == torch.device("cuda"):
        return 'cuda'
    if device == torch.device("mps"):
        return 'mps'
    else:
        return 'cpu'


def test_model(
    args,
    model,
    test_loader,
    loss_criterion,
    device,
    classification=False,
    writer=None
):
    """
    Evaluate the model on a test set for either general or classification tasks.

    Args:
        args: Command-line arguments, including configurations like `dry_run`.
        model (torch.nn.Module): Model to evaluate.
        test_loader (torch.utils.data.DataLoader): DataLoader for test data.
        loss_criterion (callable): Loss function to compute the loss.
        device (torch.device): Device to perform evaluation on (e.g., 'cuda' or 'cpu').
        classification (bool, optional): Whether the task is classification. Default is False.
        writer (torch.utils.tensorboard.SummaryWriter, optional): TensorBoard writer for logging metrics.
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0  # For classification accuracy
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # Compute loss
            if classification:
                test_loss += loss_criterion(output, target).item()
            else:
                test_loss += loss_criterion(output,
                                            target, reduction='sum').item()

            # Handle predictions and accuracy
            if classification:
                # For classification tasks
                total += target.size(0)
                # _, predicted = torch.max(output.data, 1)
                # correct += (predicted == target).sum().item()
                correct += (output.argmax(1) ==
                            target).type(torch.float).sum().item()
            else:
                # For non-classification tasks
                # get the index of the max log-probability
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

            # Collect predictions and targets if needed (e.g., for debugging or
            # extended metrics)
            all_predictions.append(output.cpu())
            all_targets.append(target.cpu())

    # Normalize loss
    if classification:
        accuracy = 100.0 * correct / total
    else:
        accuracy = 100.0 * correct / len(test_loader.dataset)
    test_loss /= len(test_loader)

    # Print results
    print(
        f"\nTest set: Average loss: {
            test_loss:.4f}, Accuracy: ({
            accuracy:.0f}%)\n")

    # Log to TensorBoard
    log_to_tensorboard(
        writer,
        'Loss/test',
        test_loss,
        args.config.epochs *
        len(test_loader))
    log_to_tensorboard(
        writer,
        'Accuracy/test',
        accuracy,
        args.config.epochs *
        len(test_loader))

    return test_loss, accuracy


def train_single_epoch(args, model, data_source, optimizer, loss_criterion, device, epoch,
                       use_amp=False, scaler=None, writer=None, explicit_data=False):
    """
    Train the model for one epoch using either a DataLoader or explicit data.

    Args:
        args: Command-line arguments containing configurations such as log_interval and dry_run.
        model (torch.nn.Module): Model to train.
        data_source (torch.utils.data.DataLoader or tuple): Data source; DataLoader or a tuple of input and target data.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        loss_criterion (callable): Loss function.
        device (torch.device): Device to perform training on.
        epoch (int): Current epoch number.
        use_amp (bool, optional): Whether to use automatic mixed precision. Default is False.
        scaler (torch.cuda.amp.GradScaler, optional): Gradient scaler for AMP. Default is None.
        writer (torch.utils.tensorboard.SummaryWriter, optional): TensorBoard writer for logging metrics.
        explicit_data (bool, optional): Whether the data_source is explicit input/target data. Default is False.
    """
    model.train()
    if explicit_data:
        input_data, target_data = data_source
        data_iterator = enumerate(zip(input_data, target_data))
        total_batches = len(input_data)
    else:
        data_iterator = enumerate(data_source)
        total_batches = len(data_source)

    for batch_idx, (data, target) in data_iterator:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        with torch.autocast(device_type=device_to_str(device), dtype=torch.float16, enabled=use_amp):
            output = model(data)
            loss = loss_criterion(output, target)

        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Logging
        if batch_idx % args.config.log_interval == 0:
            print(
                f"Train Epoch: {epoch} [{
                    batch_idx * len(data)}/{
                    total_batches * len(data)} "
                f"({100. * batch_idx / total_batches:.0f}%)]\tLoss: {loss.item():.6f}"
            )
        log_to_tensorboard(
            writer,
            'Loss/train',
            loss.item(),
            epoch *
            total_batches +
            batch_idx +
            1)

        if args.config.dry_run:
            break

# Usage
# with data_loader
# train_single_epoch(
#    args,
#    model,
#    data_source=train_loader,
#    optimizer=optimizer,
#    loss_criterion=loss_criterion,
#    device=device,
#    epoch=epoch,
#    use_amp=True,
#    scaler=scaler,
#    writer=writer,
#    explicit_data=False
# )
#
# with explicit data
# train_single_epoch(
#    args,
#    model,
#    data_source=(input_data, target_data),
#    optimizer=optimizer,
#    loss_criterion=loss_criterion,
#    device=device,
#    epoch=epoch,
#    use_amp=True,
#    scaler=scaler,
#    writer=writer,
#    explicit_data=True
# )


def train_and_evaluate(args, model, train_loader: DataLoader,
                       test_loader: DataLoader, optimizer, loss_criterion):
    """
    Runs the training and evaluation loop for the specified number of epochs.

    Args:
        args (argparse.Namespace): Parsed arguments including configuration.
        model (torch.nn.Module): The classification model to train and evaluate.
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for testing data.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        loss_criterion (callable): Loss function.
    """

    device = select_default_device(args)
    writer = None
    best_test_loss = float('inf')
    best_model_path = None

    # Initialize TensorBoard writer if logging is enabled
    if args.config.log_to_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter()

    # Create a temporary file to store the best model weights
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        best_model_path = temp_file.name

    try:
        for epoch in range(1, args.config.epochs + 1):
            train_single_epoch(
                args,
                model,
                train_loader,
                optimizer,
                loss_criterion,
                device,
                epoch,
                writer=writer)

            # Test the model and get loss and accuracy
            test_loss, test_accuracy = test_model(
                args, model, test_loader, loss_criterion, device, True, writer)

            # Check if this is the best model so far
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                # Save the model state_dict to the temporary file
                torch.save(model.state_dict(), best_model_path)

            # Flush TensorBoard writer
            if writer:
                writer.flush()

        # Load the best model weights after training
        if best_model_path:
            model.load_state_dict(
                torch.load(
                    best_model_path,
                    map_location=device,
                    weights_only=True))

    finally:
        # Clean up: close the TensorBoard writer and delete the temporary file
        if writer:
            writer.close()
        if best_model_path and os.path.exists(best_model_path):
            os.remove(best_model_path)
