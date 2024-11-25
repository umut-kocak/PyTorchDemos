import gc
import time

import torch


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
start_time = None


def start_timer(clear_cache: bool = True):
    """
    Starts a timer to measure execution time and optionally clears cache.

    Args:
        clear_cache (bool): If True, clears Python garbage collection and CUDA cache
                            to minimize memory overhead (default: True).
    """
    global start_time
    if clear_cache:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()
            torch.cuda.synchronize()
    start_time = time.time()


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
    print("Total execution time = {:.3f} sec".format(end_time - start_time))
    if torch.cuda.is_available():
        print(
            "Max memory used by tensors = {} bytes".format(
                torch.cuda.max_memory_allocated()))


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
