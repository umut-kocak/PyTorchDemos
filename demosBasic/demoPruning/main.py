"""
This script demonstrates various pruning techniques on a PyTorch LeNet model, including structured,
unstructured, global, iterative, and custom pruning methods, with detailed logging of sparsity and
parameter changes.
"""
import logging

import torch
from torch.nn.utils import prune

from common.models import LeNet
from common.utils import helper
from common.utils.arg_parser import get_common_args


def get_args():
    """
    Parses command-line arguments including model configuration details.

    Returns:
        Namespace: Parsed arguments including configuration settings.
    """
    parser = get_common_args()
    args = parser.parse_args()
    args.config = helper.load_config_file(args.config_path)
    return args


def apply_pruning(model):
    """
    Applies various pruning techniques to the LeNet model's conv1 layer.

    Args:
        model (torch.nn.Module): The model to be pruned.
    """
    module = model.conv1

    logging.info("Before pruning conv1 layer:")
    logging.info(list(module.named_parameters()))

    # Apply random unstructured pruning
    prune.random_unstructured(module, name="weight", amount=0.3)
    logging.info("After random unstructured pruning on conv1 weights:")
    logging.info(list(module.named_parameters()))
    logging.info(list(module.named_buffers()))

    # Prune 3 smallest bias entries in conv1 by L1 norm
    prune.l1_unstructured(module, name="bias", amount=3)
    logging.info("After L1 unstructured pruning on conv1 bias:")
    logging.info(list(module.named_parameters()))
    logging.info(list(module.named_buffers()))

    # Structured pruning based on L2 norm along the 0th dimension
    prune.ln_structured(module, name="weight", amount=0.5, n=2, dim=0)
    logging.info("After structured pruning on conv1 weights (50% channels):")
    logging.info(list(module.named_parameters()))

    # Remove pruning to make it permanent
    prune.remove(module, 'weight')
    logging.info("After making pruning permanent on conv1 weights:")
    logging.info(list(module.named_parameters()))


def apply_global_pruning(model, sparsity=0.3):
    """
    Applies global pruning across multiple layers of the model.

    Args:
        model (torch.nn.Module): The model to be globally pruned.
        sparsity (float): Percentage of total connections to prune.
    """
    parameters_to_prune = (
        (model.conv1, 'weight'),
        (model.conv2, 'weight'),
        (model.fc1, 'weight'),
        (model.fc2, 'weight'),
        (model.fc3, 'weight'),
    )
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=sparsity,
    )
    sparsity_percentage = calculate_global_sparsity(model)
    logging.info(
        f"After global pruning with {sparsity * 100}% sparsity: {sparsity_percentage:.2f}%")


def calculate_global_sparsity(model):
    """
    Calculates the global sparsity of the model after pruning.

    Args:
        model (torch.nn.Module): The pruned model.

    Returns:
        float: Global sparsity as a percentage.
    """
    total_zero = sum(
        torch.sum(
            layer.weight == 0) for layer in [
            model.conv1,
            model.conv2,
            model.fc1,
            model.fc2,
            model.fc3])
    total_elements = sum(
        layer.weight.nelement() for layer in [
            model.conv1,
            model.conv2,
            model.fc1,
            model.fc2,
            model.fc3])
    return 100.0 * float(total_zero) / total_elements


class CustomPruningMethod(prune.BasePruningMethod):
    """Custom pruning method to prune every other entry in a tensor."""
    PRUNING_TYPE = 'unstructured'

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        mask.view(-1)[::2] = 0
        return mask


def custom_unstructured(module, name):
    """
    Applies a custom unstructured pruning to remove every other entry in the tensor.

    Args:
        module (torch.nn.Module): Module containing the tensor to prune.
        name (str): Parameter name within the module to be pruned.
    """
    CustomPruningMethod.apply(module, name)


def main():
    """
    Main function to perform various types of pruning on a LeNet model, demonstrating
    random, structured, iterative, and global pruning, as well as a custom pruning method.
    """
    logging.basicConfig(level=logging.INFO)
    args = get_args()
    torch.manual_seed(args.seed)

    device = helper.select_default_device(args)
    model = LeNet(5).to(device=device)

    # Prune specific layers
    apply_pruning(model)

    # Apply global pruning across model
    apply_global_pruning(LeNet(5))

    # Apply custom pruning method on new model
    new_model = LeNet(5)
    custom_unstructured(new_model.fc3, name='bias')
    logging.info("After custom pruning of bias in fc3 layer:")
    logging.info(new_model.fc3.bias_mask)


if __name__ == '__main__':
    main()
