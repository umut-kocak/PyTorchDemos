import argparse
from typing import Optional, Tuple


def get_args(load_config: bool = False) -> Tuple[argparse.ArgumentParser, Optional['Config']]:
    """
    Creates an argument parser for the PyTorch Demo application and optionally loads 
    additional configuration parameters from a JSON file.

    Args:
        load_config (bool): If True, loads a configuration file specified by the 
                            `--config-path` argument.

    Returns:
        Tuple[argparse.ArgumentParser, Optional[Config]]: A tuple containing the 
        ArgumentParser instance with pre-set arguments and an optional `Config` object 
        (if `load_config` is True). If `load_config` is False, returns `None` for the 
        configuration object.
        
    Command-Line Arguments:
        --config-path (str): Path to the configuration JSON file (default: 'DefaultConfig.json').
        --verbose (bool): Enables verbose mode for additional output (default: False).
        --seed (int): Random seed for reproducibility (default: 42).
        --save-model (bool): Flag to save the current model state (default: False).
    """
    parser = argparse.ArgumentParser(description="PyTorch Demo")

    # Common arguments
    parser.add_argument('--config-path', type=str, default='DefaultConfig.json',
                        help='Path to the configuration JSON file')

    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Enable verbose mode')

    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='Random seed (default: 42)')

    parser.add_argument('--default-device', type=str, default='cuda',
                        help='Default Device(cuda, mps, cpu)')

    parser.add_argument('--default-data-path', type=str, default='./data',
                        help='Default data path')

    parser.add_argument('--default-assets-path', type=str, default='./assets',
                        help='Default assets path, for stand alone data not belonging to a dataset.')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For saving the current model')

    # Parse only known arguments to allow for dynamic extension by the caller
    args, _ = parser.parse_known_args()

    # Load configuration if `load_config` is True
    config = None
    if load_config:
        try:
            from common.utils.config import Config  # Import only if needed
            config = Config(args.config_path)
        except ImportError:
            print("Error: Config module could not be imported. Ensure `common.utils.config` is available.")

    return parser, config

# Example usage:
# parser, config = get_args(load_config=True)
# parser.add_argument('--new-arg', type=int, default=5, help='A new custom argument')
# args = parser.parse_args()  # This allows the caller to parse additional arguments
# print(args)
# if config:
#     print(config)
