"""
Utility module for common PyTorch operations including configuration loading, logging, device selection,
timing utilities, and memory usage monitoring.

This module provides the following functionalities:
- Loading configuration files in JSON format.
- Logging metrics to TensorBoard.
- Selecting an appropriate computation device (CPU, CUDA, or MPS) based on availability.
- Measuring execution time and memory usage with optional CUDA synchronization.
- Utilities for clearing cache and managing timers.

Functions:
    - load_config_file: Load and parse a configuration file.
    - log_to_tensorboard: Log scalar values to TensorBoard.
    - select_default_device: Determine the best computation device.
    - start_timer, end_timer_and_print: Manage and log execution timing.
    - timed_function_call: Measure execution time of a function.
    - get_memory_usage: Retrieve peak memory usage by tensors on CUDA.

Constants:
    - START_TIME: Global variable for tracking the start time of a timed operation.

Dependencies:
    - PyTorch (torch), including optional CUDA and MPS support.
    - Optional `Config` class for configuration handling, located in `common.utils.config`.
"""
import gc
import os
import time
from typing import Optional

import torch


def load_config_file(config_path: str) -> Optional[object]:
    """
    Loads configuration parameters from a JSON file.

    Args:
        config_path (str): Path to the configuration JSON file.

    Returns:
        Optional[object]: An instance of the `Config` object if the file is successfully loaded,
        or `None` if the file is not found, invalid, or if the `Config` class cannot be imported.

    Raises:
        FileNotFoundError: If the specified configuration file does not exist.
        Exception: If any other error occurs during configuration loading.

    Note:
        Ensure that the `Config` class is defined in `common.utils.config` and handles
        JSON parsing appropriately.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        from common.utils.config import Config  # Import only when needed
        return Config(config_path)
    except ImportError as e:
        print(
            "Error: Config module could not be imported. Ensure `common.utils.config` is available.")
        print(f"Details: {e}")
    except FileNotFoundError as e:  # Example of a specific exception
        print(f"Error: Configuration file {config_path} not found.")
        print(f"Details: {e}")
    except ValueError as e:  # Example of another specific exception
        print(f"Error: Configuration file {config_path} is invalid.")
        print(f"Details: {e}")

    return None


def log_to_tensorboard(writer, tag, value, step):
    """
    Log metrics to TensorBoard if a writer is provided.

    Args:
        writer (torch.utils.tensorboard.SummaryWriter or None): TensorBoard writer object.
        tag (str): The tag for the metric.
        value (float): The value of the metric.
        step (int): The global step (epoch or batch index).
    """
    if writer is not None:
        writer.add_scalar(tag, value, step)


def select_default_device(args) -> torch.device:
    """
    Selects and sets the computation device based on availability and config settings.

    Args:
        args: args object with `default_device` attribute specifying the desired device
                (options: "cuda", "mps", "cpu").

    Returns:
        torch.device: Selected computation device (CPU, CUDA, or MPS).

    Raises:
        ValueError: If the specified device is not supported or available.
    """
    if args.default_device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.default_device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.default_device == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError(
            f"Unsupported device '{
                args.default_device}' or device not available.")
    return device


# Timing utilities
START_TIME = None


def start_timer(clear_cache: bool = True):
    """
    Starts a timer to measure execution time and optionally clears cache.

    Args:
        clear_cache (bool): If True, clears Python garbage collection and CUDA cache
                            to minimize memory overhead (default: True).
    """
    global START_TIME # pylint: disable=W0603
    if clear_cache:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()
            torch.cuda.synchronize()
    START_TIME = time.time()


def end_timer_and_print(local_msg: str):
    """
    Ends the timer, synchronizes CUDA (if available), and prints elapsed time
    and peak memory usage.

    Args:
        local_msg (str): A message to print before displaying the timing results.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.time()
    print("\n" + local_msg)
    print(f"Total execution time = {end_time - START_TIME:.3f} sec")
    if torch.cuda.is_available():
        print(f"Max memory used by tensors = {torch.cuda.max_memory_allocated()} bytes")


def timed_function_call(fn, no_cuda: bool = False):
    """
    Measures and returns the execution time of a function, optionally using CUDA events.

    Args:
        fn (callable): The function to measure.
        no_cuda (bool): If True, disables CUDA timing even if CUDA is available (default: False).

    Returns:
        tuple: Result of `fn()` and the time taken to execute `fn()` in milliseconds.
    """
    if (not no_cuda) and torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    else:
        start = time.time()

    result = fn()

    if (not no_cuda) and torch.cuda.is_available():
        end.record()
        torch.cuda.synchronize()
        duration = start.elapsed_time(end)  # Duration in milliseconds
    else:
        end = time.time()
        duration = (end - start) * 1000  # Convert seconds to milliseconds

    return result, duration


def get_memory_usage() -> int:
    """
    Returns the maximum memory used by tensors on the CUDA device, if available.

    Returns:
        int: Maximum memory used by tensors in bytes; returns 0 if CUDA is not available.
    """
    return torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
