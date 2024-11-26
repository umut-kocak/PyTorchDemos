import argparse

def get_common_args() -> argparse.ArgumentParser:
    """
    Creates and returns an argument parser for the PyTorch Demo application.

    Returns:
        argparse.ArgumentParser: An ArgumentParser instance with pre-defined arguments 
        for configuring the PyTorch Demo.

    Command-Line Arguments:
        --config-path (str): Path to the configuration JSON file (default: 'DefaultConfig.json').
        --verbose (bool): Enables verbose mode for additional output.
        --seed (int): Random seed for reproducibility (default: 42).
        --default-device (str): Device to use for computation (e.g., 'cuda', 'mps', 'cpu').
        --default-data-path (str): Path to the default dataset directory (default: './data').
        --default-assets-path (str): Path to the default assets directory for standalone data.
        --save-model (bool): Flag to save the current model state.
    """
    parser = argparse.ArgumentParser(description="PyTorch Demo")

    # Common arguments
    parser.add_argument('--config-path', type=str, default='DefaultConfig.json',
                        help='Path to the configuration JSON file(If used) (default: DefaultConfig.json)')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Enable verbose mode for additional output')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--default-device', type=str, default='cuda',
                        help='Device to use for computation (e.g., cuda, mps, cpu)')
    parser.add_argument('--default-data-path', type=str, default='./data',
                        help='Path to the default dataset directory (default: ./data)')
    parser.add_argument('--default-assets-path', type=str, default='./assets',
                        help='Path to the default assets directory for standalone data (default: ./assets)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='Flag to save the current model state (default: False)')

    return parser
